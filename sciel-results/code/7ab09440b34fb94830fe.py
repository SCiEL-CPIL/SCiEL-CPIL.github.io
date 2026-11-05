"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Precompute DFT matrix for fast evaluation on a dense grid.
    M_grid = 16384
    k = np.arange(M_grid)
    n_idx = np.arange(N)
    D = np.exp(-2j * np.pi * np.outer(k, n_idx) / M_grid)

    def get_supnorm_fast(coeffs):
        """Fast sup norm via dense grid (matrix-vector product)."""
        P = D @ coeffs
        return float(np.max(np.abs(P)))

    def get_supnorm_accurate(coeffs):
        """Accurate sup norm with golden-section refinement of top peaks."""
        M = 16384
        c_padded = np.zeros(M, dtype=complex)
        c_padded[:N] = coeffs
        P = np.fft.fft(c_padded)
        mags = np.abs(P)

        K = 40
        idx = np.argsort(mags)[-K:][::-1]
        theta = 2 * np.pi * np.arange(M) / M

        max_val = 0.0
        for i in idx:
            w = 2 * np.pi / M
            a = theta[i] - w
            b = theta[i] + w
            gr = (5 ** 0.5 - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)

            def f(theta_val):
                z_val = np.exp(1j * theta_val)
                p_val = 0.0
                for coeff in reversed(coeffs):
                    p_val = p_val * z_val + coeff
                return np.abs(p_val)

            fc = f(c)
            fd = f(d)
            for _ in range(55):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
            peak_val = max(fc, fd)
            if peak_val > max_val:
                max_val = peak_val
        return max_val

    def local_search(current_c, time_limit):
        """
        Combined 1-flip / 2-flip local search with vectorized evaluation.
        Returns (best_c, best_val).
        """
        current_c = current_c.copy()
        P_curr = D @ current_c
        current_val = float(np.max(np.abs(P_curr)))
        best_c = current_c.copy()
        best_val = current_val

        # Precompute single-flip deltas: flipping index i changes P by -2*c_i*D[:,i]
        single_deltas = [-2.0 * current_c[i] * D[:, i] for i in range(N)]

        max_iters = 400
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break

            # --- 1-flip search ---
            best_flip_val = current_val
            best_flip_idx = -1
            for i in range(N):
                P_new = P_curr + single_deltas[i]
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i

            if best_flip_idx >= 0:
                i = best_flip_idx
                current_c[i] *= -1
                single_deltas[i] *= -1
                P_curr = D @ current_c
                current_val = best_flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue

            # --- 2-flip search (vectorized over j for each i) ---
            best_2flip_val = current_val
            best_2flip = None
            for i in range(N - 1):
                # For fixed i, evaluate all j > i at once
                # P_new = P_curr + delta_i + delta_j for each j
                delta_i = single_deltas[i]
                # Build P_new for all j in range(i+1, N)
                # P_new_j = P_curr + delta_i + delta_j
                # We can compute: P_curr + delta_i is a vector; add delta_j for each j
                base = P_curr + delta_i
                # For each j, max |base + delta_j|
                # delta_j = single_deltas[j], shape (M_grid,)
                for j in range(i + 1, N):
                    P_new = base + single_deltas[j]
                    val = np.max(np.abs(P_new))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)

            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                single_deltas[i] *= -1
                single_deltas[j] *= -1
                P_curr = D @ current_c
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue

            # No 1-flip or 2-flip improvement found
            break

        return best_c, best_val

    # --- Build diverse initial candidates ---
    initial_candidates = []

    # 1. Golay-Rudin-Shapiro (length 16) padded to 24
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs16 = np.array(p[:16], dtype=float)
    # Pad with 8 more coefficients (try several padding patterns)
    for pad in range(4):
        c = grs16.copy()
        # Extend by repeating GRS pattern
        ext = np.array(p[:8] if len(p) >= 8 else [1]*8, dtype=float)
        c = np.concatenate([grs16, ext])[:N]
        initial_candidates.append(c)

    # 2. Known near-optimal candidates from inspiration 3
    known_seeds = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
    ]
    for seed in known_seeds:
        initial_candidates.append(np.array(seed, dtype=float))

    # 3. GRS with deterministic shifts (cyclic rotations of the 16-length GRS + padding)
    for shift in range(8):
        shifted = np.roll(grs16, shift)
        c = np.concatenate([shifted, grs16[:8]])[:N]
        initial_candidates.append(c)

    # 4. Spectral synthesis: various spectrum shapes
    np.random.seed(123)
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        phases = np.random.uniform(0, 2 * np.pi, size=N)
        spectrum = np.exp(1j * phases)
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        initial_candidates.append(c_bin)

        # Cosine-tapered
        taper = np.cos(np.pi * np.arange(N) / (N - 1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        initial_candidates.append(c_bin2)

    # 5. Random starts
    for _ in range(15):
        if time.time() > deadline - 30:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        initial_candidates.append(c_rand)

    # --- Run local search on all candidates ---
    best_overall_c = None
    best_overall_val = float('inf')

    for c0 in initial_candidates:
        if time.time() > deadline - 10:
            break
        c_best, v_best = local_search(c0, time.time() + 1.5)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best

    # --- Simulated annealing with adaptive cooling ---
    T = 0.5
    cooling_rate = 0.995
    min_T = 1e-6

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    triples = [(i, j, k) for i in range(N - 2) for j in range(i + 1, N - 1) for k in range(j + 1, N)]

    sa_best_c = best_overall_c.copy()
    sa_best_val = best_overall_val
    current_c = best_overall_c.copy()
    current_val = best_overall_val

    sa_step = 0
    while T > min_T and time.time() < deadline - 20:
        sa_step += 1
        # Pick random move
        r = np.random.rand()
        if r < 0.5:
            i, j = pairs[np.random.randint(len(pairs))]
            current_c[i] *= -1
            current_c[j] *= -1
            new_score = get_supnorm_fast(current_c)
            delta = new_score - current_val

            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = current_c.copy()
            else:
                current_c[i] *= -1
                current_c[j] *= -1
        elif r < 0.8:
            i = np.random.randint(N)
            current_c[i] *= -1
            new_score = get_supnorm_fast(current_c)
            delta = new_score - current_val

            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = current_c.copy()
            else:
                current_c[i] *= -1
        elif r < 0.95:
            i, j, k = triples[np.random.randint(len(triples))]
            current_c[i] *= -1
            current_c[j] *= -1
            current_c[k] *= -1
            new_score = get_supnorm_fast(current_c)
            delta = new_score - current_val

            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = current_c.copy()
            else:
                current_c[i] *= -1
                current_c[j] *= -1
                current_c[k] *= -1
        else:
            idxs = np.random.choice(N, size=4, replace=False)
            for idx in idxs:
                current_c[idx] *= -1
            new_score = get_supnorm_fast(current_c)
            delta = new_score - current_val

            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = current_c.copy()
            else:
                for idx in idxs:
                    current_c[idx] *= -1

        T *= cooling_rate

        # Periodic local search every ~200 SA steps
        if sa_step % 200 == 0 and time.time() < deadline - 25:
            current_c, current_val = local_search(current_c, time.time() + 1.0)
            if current_val < sa_best_val:
                sa_best_val = current_val
                sa_best_c = current_c.copy()

        # Occasionally restart from best seen
        if sa_step % 500 == 0:
            current_c = sa_best_c.copy()
            current_val = sa_best_val

    # --- Final local search on the best solution ---
    best_overall_c, best_overall_val = local_search(sa_best_c, deadline - 10)

    # --- Try additional perturbed local searches ---
    for _ in range(8):
        if time.time() > deadline - 10:
            break
        perturbed = best_overall_c.copy()
        n_flips = np.random.randint(2, 6)
        for _ in range(n_flips):
            idx = np.random.randint(N)
            perturbed[idx] *= -1
        c_best, v_best = local_search(perturbed, time.time() + 1.5)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best

    # --- Final high-precision polish: more iterations of local search ---
    best_overall_c, best_overall_val = local_search(best_overall_c, deadline - 5)

    return [int(c) for c in best_overall_c]

# EVOLVE-BLOCK-END
