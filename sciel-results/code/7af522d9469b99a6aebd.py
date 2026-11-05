import numpy as np
import time

def _eval_sup_norm_refined(coeffs, N_dft=2**16, n_top=10, n_refine=60):
    """Evaluate sup norm with golden-section refinement of top peaks."""
    n = len(coeffs)
    coeffs = np.array(coeffs, dtype=np.float64)
    
    # Dense DFT sampling
    theta = 2 * np.pi * np.arange(N_dft) / N_dft
    z = np.exp(1j * theta)
    
    # Evaluate P(z) = sum c_k z^k using FFT
    c_padded = np.zeros(N_dft, dtype=np.complex128)
    c_padded[:n] = coeffs
    P = np.fft.ifft(c_padded) * N_dft
    
    abs_P = np.abs(P)
    
    # Find top peaks
    top_idx = np.argsort(abs_P)[-n_top:]
    
    best_val = np.max(abs_P)
    
    # Golden-section refinement around each top peak
    for idx in top_idx:
        theta_c = 2.0 * np.pi * idx / N_dft
        # Search in a window around the peak
        delta = 2.0 * np.pi / N_dft
        a = theta_c - delta
        b = theta_c + delta
        
        # Golden section search
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        
        def f(theta):
            zz = np.exp(1j * theta)
            # Direct evaluation for accuracy
            powers = np.power(zz, np.arange(n))
            return np.abs(np.dot(coeffs, powers))
        
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(n_refine):
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
        
        val = max(f1, f2)
        if val > best_val:
            best_val = val
    
    return best_val / np.sqrt(24.0)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Known good candidates from literature
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
        [1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1],
        [1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1],
        # Golay-Rudin-Shapiro and variants
        [1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    ]
    
    # Use a fine DFT grid for consistent evaluation during search
    M = 2**17  # 131072 points
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    sqrt24 = np.sqrt(24.0)
    
    def fast_eval(c_arr):
        """Fast DFT-based evaluation."""
        f_arr = D @ c_arr
        return np.max(np.abs(f_arr))
    
    def local_search_1flip(current_c, current_f, current_max, max_iters=50):
        """1-flip local search until no improvement or max iterations."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_flip = -1
            best_max_val = current_max
            for n in range(24):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-10:
                    best_max_val = max_val
                    best_flip = n
            if best_flip != -1:
                current_c[best_flip] *= -1.0
                current_f = current_f - 2.0 * current_c[best_flip] * D[:, best_flip]
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max
    
    def local_search_2flip(current_c, current_f, current_max):
        """Try 2-flips to escape local minima."""
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-10:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
            current_max = best_max_2
        return current_c, current_f, current_max
    
    def local_search_3flip(current_c, current_f, current_max):
        """Try 3-flips to escape deeper local minima."""
        best_flip_3 = (-1, -1, -1)
        best_max_3 = current_max
        # Sample a subset of 3-flips for speed
        rng_3 = np.random.RandomState(int(time.time() * 1000) % 10000)
        n_3flips = 200
        for _ in range(n_3flips):
            idx = rng_3.choice(24, 3, replace=False)
            n1, n2, n3 = int(idx[0]), int(idx[1]), int(idx[2])
            f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2] + current_c[n3] * D[:, n3])
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_3 - 1e-10:
                best_max_3 = max_val
                best_flip_3 = (n1, n2, n3)
        if best_flip_3 != (-1, -1, -1):
            n1, n2, n3 = best_flip_3
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_c[n3] *= -1.0
            current_f = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2] + current_c[n3] * D[:, n3])
            current_max = best_max_3
        return current_c, current_f, current_max
    
    best_score = float('inf')
    best_c = None
    
    # Evaluate all candidates and do local search on each
    for c in candidates:
        if time.time() > deadline - 30:  # Leave time for refinement
            break
        c_arr = np.array(c, dtype=float)
        f_arr = D @ c_arr
        max_val = np.max(np.abs(f_arr))
        
        # Do 1-flip local search
        c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        # Try 2-flip and 3-flip escapes
        for _ in range(6):
            if time.time() > deadline - 30:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_3flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        score = max_val / sqrt24
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()
    
    # Now do iterated local search from best_c with random perturbations
    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f))
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 20:  # Leave time for final refinement
        # Try 1-flip
        best_flip_1 = -1
        best_max_1 = current_max
        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_1 - 1e-10:
                best_max_1 = max_val
                best_flip_1 = n
        
        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = current_f - 2.0 * current_c[best_flip_1] * D[:, best_flip_1]
            current_max = best_max_1
            if current_max / sqrt24 < best_score:
                best_score = current_max / sqrt24
                best_c = current_c.copy()
            continue
        
        # Try 2-flip
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-10:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
            current_max = best_max_2
            if current_max / sqrt24 < best_score:
                best_score = current_max / sqrt24
                best_c = current_c.copy()
            continue
        
        # Try 3-flip
        best_flip_3 = (-1, -1, -1)
        best_max_3 = current_max
        n_3flips = 300
        for _ in range(n_3flips):
            idx = rng.choice(24, 3, replace=False)
            n1, n2, n3 = int(idx[0]), int(idx[1]), int(idx[2])
            f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2] + current_c[n3] * D[:, n3])
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_3 - 1e-10:
                best_max_3 = max_val
                best_flip_3 = (n1, n2, n3)
        
        if best_flip_3 != (-1, -1, -1):
            n1, n2, n3 = best_flip_3
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_c[n3] *= -1.0
            current_f = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2] + current_c[n3] * D[:, n3])
            current_max = best_max_3
            if current_max / sqrt24 < best_score:
                best_score = current_max / sqrt24
                best_c = current_c.copy()
            continue
        
        # Stuck: random restart by flipping 2-4 random bits
        n_flips = rng.choice([2, 3, 4], p=[0.35, 0.4, 0.25])
        indices = rng.choice(24, n_flips, replace=False)
        current_c[indices] *= -1.0
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        # Do a quick 1-flip local search after restart
        current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max, max_iters=20)
        if current_max / sqrt24 < best_score:
            best_score = current_max / sqrt24
            best_c = current_c.copy()
    
    # Final polish: do a few more rounds of 1-flip and 2-flip on the best candidate
    if time.time() < deadline - 5:
        current_c = best_c.copy()
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max, max_iters=30)
            current_c, current_f, current_max = local_search_2flip(current_c, current_f, current_max)
            if current_max / sqrt24 < best_score:
                best_score = current_max / sqrt24
                best_c = current_c.copy()
    
    return [int(x) for x in best_c]
