import numpy as np
import time

def design(problem):
    """
    Differential-evolution search over C4v-symmetric superellipse dielectric rods.
    The design space is (amplitude a, exponent p, aspect r) defining
        |x/a|^p + |y/(a*r)|^p <= 1  (centered at cell center)
    Superellipses include circles (p=2), squares (p->inf), and intermediate shapes.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Plane-wave setup (matches evaluator: 289 PWs, 30 k-points) ---
    N_pw = 17  # -16..16 => 289 PWs
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # (289, 2)
    num_pw = len(G_vectors)

    # 30 k-points covering irreducible BZ (Gamma-M, M-X, Gamma-X)
    k_points = []
    # Gamma to M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    # M to X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # Gamma to X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    k_points = np.array(unique_k)  # (30, 2)

    # Precompute G_diff: G[i] - G[j] for all pairs
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (289, 289, 2)

    def evaluate_cell(eps_grid):
        """
        Computes the TM band gap between bands 4 and 5 using a 289-plane-wave
        Galerkin method. eps_grid is a 32x32 array of permittivity values.
        Returns the gap-midgap ratio.
        """
        inv_eps_grid = 1.0 / eps_grid
        # FFT to get Fourier coefficients of 1/eps
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)

        # Build coefficient matrix: coeff[i,j] = <G_i|1/eps|G_j>
        # = Fourier coefficient of (G_i - G_j)
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors  # (289, 2)
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (289,)

            # TM: M[i,j] = (k+G_i)^2 * <G_i|1/eps|G_j>
            M = coeff_matrix * omega_sq[:, np.newaxis]

            # Symmetrize to ensure Hermitian (fixes non-Hermitian eigenvalue issues)
            M = 0.5 * (M + M.conj().T)

            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(np.real(eigvals))

            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(max(real_eigvals[3], 0))
                omega_5 = np.sqrt(max(real_eigvals[4], 0))

                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)

        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap <= 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    def params_to_cell(params):
        """
        Convert superellipse parameters (a, p, r) to a 32x32 binary cell.
        a: amplitude (half-width), p: exponent, r: aspect ratio
        Superellipse: |x/a|^p + |y/(a*r)|^p <= 1
        """
        a, p, r = params
        a = np.clip(a, 0.05, 0.48)
        p = np.clip(p, 1.0, 20.0)
        r = np.clip(r, 0.3, 1.0)  # aspect ratio, capped at 1 for C4v (square symmetry)

        # Grid coordinates centered at (0, 0)
        x = np.linspace(0.5 / size, 1 - 0.5 / size, size)
        X, Y = np.meshgrid(x - 0.5, x - 0.5, indexing='ij')

        # Superellipse condition
        # Handle p -> infinity (square): use max(|x/a|, |y/(a*r)|) <= 1
        if p > 10:
            # Approximate square
            cond = (np.abs(X) / a) <= 1.0
            cond = cond & (np.abs(Y) / (a * r)) <= 1.0
        else:
            cond = (np.abs(X) / a) ** p + (np.abs(Y) / (a * r)) ** p <= 1.0

        cell = cond.astype(int)
        return cell

    def evaluate_params(params):
        """Evaluate a parameter vector: returns the gap ratio."""
        cell = params_to_cell(params)
        eps_grid = eps_low + (eps_high - eps_low) * cell.astype(float)
        return evaluate_cell(eps_grid)

    # --- Differential Evolution ---
    pop_size = 24
    F = 0.8  # mutation scale
    CR = 0.9  # crossover rate
    bounds = [(0.05, 0.48), (1.0, 20.0), (0.3, 1.0)]

    # Initialize population
    pop = np.zeros((pop_size, 3))
    for i in range(pop_size):
        for d in range(3):
            pop[i, d] = np.random.uniform(bounds[d][0], bounds[d][1])

    # Evaluate initial population
    scores = np.array([evaluate_params(pop[i]) for i in range(pop_size)])
    best_idx = np.argmax(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]

    # DE loop
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        for i in range(pop_size):
            if time.time() > deadline - 10:
                break

            # Mutation: DE/rand/1
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(pop_size, 3, replace=False)

            mutant = pop[a] + F * (pop[b] - pop[c])
            # Clip to bounds
            for d in range(3):
                mutant[d] = np.clip(mutant[d], bounds[d][0], bounds[d][1])

            # Crossover: binomial
            trial = pop[i].copy()
            j_rand = np.random.randint(3)
            for d in range(3):
                if np.random.rand() < CR or d == j_rand:
                    trial[d] = mutant[d]

            # Evaluate trial
            trial_score = evaluate_params(trial)

            # Selection
            if trial_score >= scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_params = trial.copy()

    # --- Finalize: binarize and enforce strict C4v symmetry ---
    final_cell = params_to_cell(best_params)

    # Enforce strict symmetry: average over all 8 symmetry operations
    sym_cell = np.zeros((size, size), dtype=float)
    counts = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            val = float(final_cell[i, j])
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            for pi, pj in positions:
                sym_cell[pi, pj] += val
                counts[pi, pj] += 1

    # Binarize: threshold at 0.5 of the average
    final_binary_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            if counts[i, j] > 0:
                avg = sym_cell[i, j] / counts[i, j]
                row.append(1 if avg >= 0.5 else 0)
            else:
                row.append(0)
        final_binary_cell.append(row)

    return final_binary_cell
