import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Known good candidates from literature (Odlyzko tables, Golay sequences, etc.)
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        # Additional candidates: Golay-Rudin-Shapiro and variants
        [1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    ]

    M = 2**14  # 16384 points: fast screening
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    sqrt24 = np.sqrt(24.0)

    def evaluate_full(coeffs):
        """Evaluate sup-norm using full DFT array."""
        F = D @ coeffs
        return np.max(np.abs(F)) / sqrt24

    def golden_section_refine(coeffs, peak_indices, n_refine=50):
        """Refine peak locations using golden-section search."""
        # Evaluate at dense points near each peak
        best_val = 0.0
        for idx in peak_indices:
            theta_center = 2.0 * np.pi * idx / M
            # Search in a small window around the peak
            window = 4.0 / M  # ~2 sample points on each side
            a = theta_center - window
            b = theta_center + window
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c_point = b - gr * (b - a)
            d_point = a + gr * (b - a)

            def eval_at(theta):
                z = np.exp(1j * theta)
                # Horner's method for polynomial evaluation
                val = 0.0
                for ck in reversed(coeffs):
                    val = val * z + ck
                return abs(val)

            fc = eval_at(c_point)
            fd = eval_at(d_point)
            for _ in range(n_refine):
                if fc < fd:
                    b = d_point
                    d_point = c_point
                    fd = fc
                    c_point = b - gr * (b - a)
                    fc = eval_at(c_point)
                else:
                    a = c_point
                    c_point = d_point
                    fc = fd
                    d_point = a + gr * (b - a)
                    fd = eval_at(d_point)
            local_max = max(fc, fd)
            if local_max > best_val:
                best_val = local_max
        return best_val / sqrt24

    def evaluate_accurate(coeffs):
        """Evaluate with golden-section refinement of top peaks."""
        F = D @ coeffs
        abs_F = np.abs(F)
        # Get top 10 peak locations
        top_k = 10
        peak_indices = np.argsort(abs_F)[-top_k:]
        peak_indices = np.unique(peak_indices)  # Remove duplicates
        refined_max = golden_section_refine(coeffs, peak_indices, n_refine=50)
        return refined_max

    best_score = float('inf')
    best_c = None

    for c in candidates:
        c_arr = np.array(c, dtype=float)
        score = evaluate_accurate(c_arr)
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()

    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f)) / sqrt24

    np.random.seed(42)

    # Precompute column norms for speed
    D_cols = D  # shape (M, 24)

    # Iterated Local Search with simulated annealing
    temperature = 0.08
    cooling = 0.998
    min_temp = 0.005
    stuck_count = 0

    while time.time() < deadline:
        if temperature < min_temp:
            temperature = 0.08
            # Random restart: flip 5-8 random bits
            n_flips = np.random.randint(5, 9)
            indices = np.random.choice(24, n_flips, replace=False)
            current_c[indices] *= -1.0
            current_f = D @ current_c
            current_max = np.max(np.abs(current_f)) / sqrt24
            stuck_count = 0

        improved = False

        # 1. Try 1-bit flips
        best_flip_1 = -1
        best_max_1 = current_max

        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D_cols[:, n]
            max_val = np.max(np.abs(f_new)) / sqrt24
            if max_val < best_max_1:
                best_max_1 = max_val
                best_flip_1 = n

        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = D @ current_c
            current_max = best_max_1
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            stuck_count = 0
            continue

        # 2. Try 2-bit flips if 1-bit failed
        best_flip_2 = (-1, -1)
        best_max_2 = current_max

        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D_cols[:, n1] + current_c[n2] * D_cols[:, n2])
                max_val = np.max(np.abs(f_new)) / sqrt24
                if max_val < best_max_2:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)

        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            stuck_count = 0
            continue

        # 3. Try 3-bit flips (sample a subset for speed)
        if stuck_count < 5:
            best_flip_3 = (-1, -1, -1)
            best_max_3 = current_max
            # Sample 200 random 3-combinations
            n_samples = 200
            for _ in range(n_samples):
                idx3 = np.random.choice(24, 3, replace=False)
                f_new = current_f - 2.0 * (current_c[idx3[0]] * D_cols[:, idx3[0]] +
                                            current_c[idx3[1]] * D_cols[:, idx3[1]] +
                                            current_c[idx3[2]] * D_cols[:, idx3[2]])
                max_val = np.max(np.abs(f_new)) / sqrt24
                if max_val < best_max_3:
                    best_max_3 = max_val
                    best_flip_3 = (idx3[0], idx3[1], idx3[2])

            if best_flip_3 != (-1, -1, -1):
                n1, n2, n3 = best_flip_3
                current_c[n1] *= -1.0
                current_c[n2] *= -1.0
                current_c[n3] *= -1.0
                current_f = D @ current_c
                current_max = best_max_3
                if current_max < best_score:
                    best_score = current_max
                    best_c = current_c.copy()
                temperature *= cooling
                stuck_count = 0
                continue

        # 4. Stuck: accept worse move with probability (simulated annealing)
        n_flips = np.random.randint(2, 5)
        indices = np.random.choice(24, n_flips, replace=False)
        old_c = current_c.copy()
        old_max = current_max

        current_c[indices] *= -1.0
        current_f = D @ current_c
        new_max = np.max(np.abs(current_f)) / sqrt24

        delta = new_max - old_max
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-10)):
            # Accept
            current_max = new_max
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
        else:
            # Reject: revert
            current_c = old_c
            current_f = D @ current_c
            current_max = old_max

        temperature *= cooling
        stuck_count += 1

    # Final accurate evaluation of best candidate
    final_score = evaluate_accurate(best_c)

    # Also evaluate a few top candidates from the search history
    # (best_c is already the best found)

    return [int(x) for x in best_c]
