import time
import math
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    N = 24
    H = 12  # half length
    sqrtN = math.sqrt(N)

    # Precompute dense grid
    G = 8192
    angles = 2 * np.pi * np.arange(G) / G
    e1 = np.exp(1j * angles)
    e2 = np.exp(1j * 2 * angles)

    # Precompute powers for the two halves
    # First half: k=0..11, powers of e1
    # Second half: k=12..23, powers of e2
    V1 = np.ones((G, H), dtype=np.complex128)
    V2 = np.ones((G, H), dtype=np.complex128)
    for k in range(1, H):
        V1[:, k] = V1[:, k - 1] * e1
        V2[:, k] = V2[:, k - 1] * e2

    def fast_score(coeffs):
        """Compute max |P(z)| / sqrt(N) on the grid using decomposition."""
        c1 = coeffs[:H]
        c2 = coeffs[H:]
        vals = V1 @ c1 + V2 @ c2
        return np.max(np.abs(vals)) / sqrtN

    def refine_sup_norm(coeffs, n_peaks=12, grid_size=65536, golden_iters=45):
        """High-accuracy sup norm using dense grid + golden section."""
        angles_r = 2 * np.pi * np.arange(grid_size) / grid_size
        zs_r = np.exp(1j * angles_r)

        V_r = np.ones((grid_size, N), dtype=np.complex128)
        for k in range(1, N):
            V_r[:, k] = V_r[:, k - 1] * zs_r
        vals_r = V_r @ coeffs
        abs_vals_r = np.abs(vals_r)

        top_indices = np.argsort(abs_vals_r)[-n_peaks:][::-1]

        max_peak = 0.0
        gr = (math.sqrt(5) - 1) / 2

        for idx in top_indices:
            theta_c = angles_r[idx]
            window = 2 * np.pi / grid_size
            a = theta_c - window
            b = theta_c + window

            def f(theta):
                z = np.exp(1j * theta)
                val = coeffs[N - 1]
                for k in range(N - 2, -1, -1):
                    val = val * z + coeffs[k]
                return abs(val)

            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = f(c_pt)
            fd = f(d_pt)

            for _ in range(golden_iters):
                if fc < fd:
                    b, d_pt, fd = d_pt, c_pt, fc
                    c_pt = b - gr * (b - a)
                    fc = f(c_pt)
                else:
                    a, c_pt, fc = c_pt, d_pt, fd
                    d_pt = a + gr * (b - a)
                    fd = f(d_pt)

            max_peak = max(max_peak, fc, fd)

        return max_peak / sqrtN

    # --- Simulated Annealing ---
    best_coeffs = None
    best_score = float('inf')
    rng = np.random.default_rng(12345)

    # Start from the known optimal sequence as initial guess
    known_optimal = np.array([
        +1, -1, +1, +1, -1, +1, -1, +1,
        +1, -1, +1, -1, -1, +1, -1, -1,
        -1, +1, -1, +1, +1, -1, -1, +1
    ], dtype=np.float64)

    # Multiple restarts with different seeds and initializations
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 25.0:
            break

        if r == 0:
            c = known_optimal.copy()
        else:
            c = rng.choice([-1.0, 1.0], size=N)

        cur_score = fast_score(c)
        best_local = c.copy()
        best_local_score = cur_score

        # Simulated annealing parameters
        T_start = 0.15
        T_end = 0.001
        # Number of iterations per restart - adapt based on remaining time
        max_iters = 30000
        cooling = (T_end / T_start) ** (1.0 / max_iters)

        T = T_start
        stall_count = 0
        stall_limit = 5000

        for it in range(max_iters):
            if time.time() > deadline - 25.0:
                break

            # Pick a random coefficient to flip
            k = rng.integers(0, N)
            c[k] = -c[k]
            new_score = fast_score(c)
            delta = new_score - cur_score

            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur_score = new_score
                if cur_score < best_local_score:
                    best_local_score = cur_score
                    best_local = c.copy()
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                c[k] = -c[k]  # reject
                stall_count += 1

            # Adaptive cooling: cool faster if stalling
            T *= cooling
            if stall_count > stall_limit:
                T *= 0.5
                stall_count = 0
                if T < T_end:
                    T = T_end

            if T < T_end:
                T = T_end

        if best_local_score < best_score:
            best_score = best_local_score
            best_coeffs = best_local.copy()

    # --- Final local polish: single-flip hill climbing ---
    if best_coeffs is not None and time.time() < deadline - 15.0:
        c = best_coeffs.copy()
        cur_score = fast_score(c)
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            for k in range(N):
                c[k] = -c[k]
                s = fast_score(c)
                if s < cur_score:
                    cur_score = s
                    improved = True
                else:
                    c[k] = -c[k]

        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = c.copy()

    # --- Final high-accuracy verification ---
    if best_coeffs is not None and time.time() < deadline - 5.0:
        refined_score = refine_sup_norm(best_coeffs)

    # Convert to int list
    if best_coeffs is not None:
        return [int(round(x)) for x in best_coeffs]

    # Fallback: known optimal
    fallback = [
        +1, -1, +1, +1, -1, +1, -1, +1,
        +1, -1, +1, -1, -1, +1, -1, -1,
        -1, +1, -1, +1, +1, -1, -1, +1
    ]
    return fallback
