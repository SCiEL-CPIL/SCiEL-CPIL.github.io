import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 unit cell by optimizing a C4v-symmetric Fourier representation
    of the dielectric field to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Parameterization: C4v symmetric Fourier series ---
    # f(x,y) = sum A_{mn} cos(2*pi*m*(x-0.5)) cos(2*pi*n*(y-0.5))
    # Symmetry about x=0.5, y=0.5 is automatic.
    # Symmetry about y=x requires A_{mn} = A_{nm}.
    
    max_harmonic = 4
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)

    # Precompute grid coordinates
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5

    # Precompute basis functions
    basis_funcs = []
    for (m, n) in params_idx:
        if m == 0 and n == 0:
            basis_funcs.append(np.ones((size, size)))
        else:
            cm = np.cos(2 * np.pi * m * Xc)
            cn = np.cos(2 * np.pi * n * Yc)
            basis_funcs.append(cm * cn)
    basis_funcs = np.array(basis_funcs)  # shape (num_params, size, size)

    def get_field(params):
        """Returns the 32x32 epsilon grid from parameters."""
        field = np.tensordot(params, basis_funcs, axes=1)
        eps_grid = 0.5 * (eps_low + eps_high) + 0.5 * (eps_high - eps_low) * np.tanh(field)
        return eps_grid

    def get_binary_cell(params):
        """Returns the 32x32 binary cell from parameters."""
        final_grid = get_field(params)
        threshold = 0.5 * (eps_low + eps_high)
        return (final_grid > threshold).astype(int)

    # --- Plane-wave solver setup (matching evaluator's 289 PWs) ---
    N_pw = 17  # 17x17 = 289 plane waves
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)

    # K points: 30 points in irreducible BZ as specified by evaluator
    # Irreducible BZ of square lattice: 0 <= kx <= ky, kx + ky <= 0.5
    k_points = []
    # Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    # M-X: (0.5, t) for t in [0, 0.5]  
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    
    # Remove duplicates and sort
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 12), round(k[1], 12))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G-G' differences vectorized
    G_diff_x = G_vectors[:, np.newaxis, 0] - G_vectors[np.newaxis, :, 0]  # (num_pw, num_pw)
    G_diff_y = G_vectors[:, np.newaxis, 1] - G_vectors[np.newaxis, :, 1]

    def evaluate_cell_fast(eps_grid):
        """
        Computes the band gap using a fast vectorized plane-wave method.
        Uses 289 PWs and 30 k-points.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)

        # Vectorized coefficient matrix construction
        # coeff_matrix[i,j] = fft_inv_eps[G_diff[i,j] mod size]
        ix = G_diff_x % size
        iy = G_diff_y % size
        coeff_matrix = fft_inv_eps[ix, iy]  # (num_pw, num_pw)

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)

            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]

            # For TM mode, we solve det(M - omega^2 I) = 0
            # M is not Hermitian in general, but the eigenvalues should be real
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))

            if len(real_eigvals) >= 5:
                # Filter out negative eigenvalues (numerical artifacts)
                positive_eigs = real_eigvals[real_eigvals > 1e-10]
                if len(positive_eigs) >= 5:
                    omega_4 = np.sqrt(positive_eigs[3])
                    omega_5 = np.sqrt(positive_eigs[4])
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)

        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap < 1e-10:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    def evaluate_params(params):
        """Evaluate the score for given parameters."""
        eps_grid = get_field(params)
        return evaluate_cell_fast(eps_grid)

    # --- Soft surrogate for smooth optimization ---
    def soft_gap(params, beta=5.0):
        """
        Soft version of the gap objective using log-sum-exp.
        Returns a differentiable approximation of the gap ratio.
        """
        eps_grid = get_field(params)
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)

        ix = G_diff_x % size
        iy = G_diff_y % size
        coeff_matrix = fft_inv_eps[ix, iy]

        # Collect all omega_4 and omega_5 values
        all_omega_4 = []
        all_omega_5 = []

        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            M = coeff_matrix * omega_sq[:, np.newaxis]
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            positive_eigs = real_eigvals[real_eigvals > 1e-10]
            if len(positive_eigs) >= 5:
                all_omega_4.append(np.sqrt(positive_eigs[3]))
                all_omega_5.append(np.sqrt(positive_eigs[4]))

        if not all_omega_4:
            return -1.0

        all_omega_4 = np.array(all_omega_4)
        all_omega_5 = np.array(all_omega_5)

        # Soft max of omega_4 and soft min of omega_5
        max_omega_4 = np.log(np.sum(np.exp(beta * all_omega_4)) / len(all_omega_4)) / beta
        min_omega_5 = -np.log(np.sum(np.exp(-beta * all_omega_5)) / len(all_omega_5)) / beta

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap < 1e-10:
            return -1.0
        return 2 * gap / midgap

    # --- Optimization ---
    best_params = None
    best_score = -np.inf

    # Initial guesses based on known photonic crystal structures
    initial_guesses = []
    
    # Guess 1: Dielectric rods (high A(1,1), low A(0,0))
    g1 = np.zeros(num_params)
    idx_11 = params_idx.index((1, 1))
    idx_00 = params_idx.index((0, 0))
    g1[idx_11] = 2.5
    g1[idx_00] = -1.0
    initial_guesses.append(g1)

    # Guess 2: Air holes in dielectric (low A(1,1), high A(0,0))
    g2 = np.zeros(num_params)
    g2[idx_11] = -2.5
    g2[idx_00] = 1.0
    initial_guesses.append(g2)

    # Guess 3: Square-like pattern
    g3 = np.zeros(num_params)
    idx_22 = params_idx.index((2, 2)) if (2, 2) in params_idx else None
    idx_13 = params_idx.index((1, 3)) if (1, 3) in params_idx else None
    g3[idx_11] = 1.5
    g3[idx_00] = -0.5
    if idx_22 is not None:
        g3[idx_22] = -0.8
    if idx_13 is not None:
        g3[idx_13] = 0.5
    initial_guesses.append(g3)

    # Guess 4: Cross pattern
    g4 = np.zeros(num_params)
    idx_10 = params_idx.index((1, 0)) if (1, 0) in params_idx else None
    idx_01 = params_idx.index((0, 1)) if (0, 1) in params_idx else None
    g4[idx_11] = 2.0
    g4[idx_00] = -0.8
    if idx_10 is not None:
        g4[idx_10] = -1.0
    if idx_01 is not None:
        g4[idx_01] = -1.0
    initial_guesses.append(g4)

    # Evaluate initial guesses
    for guess in initial_guesses:
        score = evaluate_params(guess)
        if score > best_score:
            best_score = score
            best_params = guess.copy()

    # Gradient-based local refinement using finite differences
    def gradient_ascent(params, score_func, lr=0.1, max_iter=50, tol=1e-6):
        """Simple gradient ascent with finite difference gradients."""
        p = params.copy()
        s = score_func(p)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient via central differences
            grad = np.zeros_like(p)
            h = 0.01
            for i in range(len(p)):
                p_plus = p.copy()
                p_minus = p.copy()
                p_plus[i] += h
                p_minus[i] -= h
                sp = score_func(p_plus)
                sm = score_func(p_minus)
                grad[i] = (sp - sm) / (2 * h)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                break
            
            # Update
            p_new = p + lr * grad / grad_norm
            s_new = score_func(p_new)
            
            if s_new > s:
                p = p_new
                s = s_new
            else:
                # Backtrack
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return p, s

    # Phase 1: Random search with restarts
    num_random_attempts = 80
    for attempt in range(num_random_attempts):
        if time.time() > deadline - 30:
            break

        # Generate candidate
        if attempt < 30:
            # Local perturbation of best
            p = best_params + np.random.normal(0, 0.5, num_params)
        elif attempt < 60:
            # Medium exploration
            p = best_params * 0.5 + np.random.uniform(-2, 2, num_params) * 0.5
        else:
            # Random restart
            p = np.random.uniform(-3, 3, num_params)
            # Bias towards rod-like: A(1,1) positive
            if (1, 1) in params_idx:
                idx = params_idx.index((1, 1))
                p[idx] = abs(p[idx]) + 1.5

        score = evaluate_params(p)

        if score > best_score:
            best_score = score
            best_params = p.copy()

    # Phase 2: Gradient-based refinement from best point
    if time.time() < deadline - 20:
        refined_params, refined_score = gradient_ascent(
            best_params, evaluate_params, lr=0.3, max_iter=30, tol=1e-5
        )
        if refined_score > best_score:
            best_score = refined_score
            best_params = refined_params

    # Phase 3: More local search around best
    if time.time() < deadline - 15:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            p_local = best_params + np.random.normal(0, 0.2, num_params)
            score_local = evaluate_params(p_local)
            if score_local > best_score:
                best_score = score_local
                best_params = p_local.copy()
        
        # Another gradient step
        if time.time() < deadline - 10:
            refined_params, refined_score = gradient_ascent(
                best_params, evaluate_params, lr=0.1, max_iter=20, tol=1e-6
            )
            if refined_score > best_score:
                best_score = refined_score
                best_params = refined_params

    if best_params is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x_c = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y_c = (j + 0.5) / size - 0.5
                if x_c * x_c + y_c * y_c < 0.2 ** 2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell

    # Get the binary cell from best parameters
    binary_cell = get_binary_cell(best_params)

    # Enforce strict C4v symmetry
    # The Fourier parameterization should already be symmetric, but let's verify and fix
    sym_cell = np.zeros((size, size), dtype=int)
    count = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            # Original
            sym_cell[i, j] += binary_cell[i, j]
            count[i, j] += 1
            # 180 rotation
            r_i, r_j = size - 1 - i, size - 1 - j
            sym_cell[r_i, r_j] += binary_cell[i, j]
            count[r_i, r_j] += 1
            # Transpose
            t_i, t_j = j, i
            sym_cell[t_i, t_j] += binary_cell[i, j]
            count[t_i, t_j] += 1
            # Reflection across diagonal
            f_i, f_j = size - 1 - j, i
            sym_cell[f_i, f_j] += binary_cell[i, j]
            count[f_i, f_j] += 1

    # Majority vote
    final_sym_cell = (sym_cell / count) >= 0.5
    final_sym_cell = final_sym_cell.astype(int)

    # Final check: ensure strict symmetry by direct assignment from the fundamental domain
    # Fundamental domain: 0 <= i <= 15, 0 <= j <= 15 (upper-left quadrant)
    # Actually for C4v, we need to be more careful
    # Let's just force symmetry by copying from one representative
    for i in range(size):
        for j in range(size):
            val = final_sym_cell[i, j]
            # 180 rotation
            final_sym_cell[size - 1 - i, size - 1 - j] = val
            # Transpose  
            final_sym_cell[j, i] = val
            # Reflection
            final_sym_cell[size - 1 - j, i] = val

    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_sym_cell[i, j]))
        result.append(row)

    return result
