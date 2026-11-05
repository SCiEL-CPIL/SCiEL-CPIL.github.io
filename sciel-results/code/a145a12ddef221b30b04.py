import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Known good candidates from literature and random search
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        # Additional candidates
        [1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
        [1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1],
        [1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1],
    ]
    
    M = 2**15
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def local_search_1flip(current_c, current_f, current_max):
        """Thorough 1-flip local search until no improvement."""
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_max_val = current_max
            for n in range(24):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-12:
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
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
            current_max = best_max_2
        return current_c, current_f, current_max
    
    best_score = float('inf')
    best_c = None
    
    # Evaluate all candidates and do local search on each
    for c in candidates:
        if time.time() > deadline:
            break
        c_arr = np.array(c, dtype=float)
        f_arr = D @ c_arr
        max_val = np.max(np.abs(f_arr))
        
        # Do 1-flip local search
        c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        # Try 2-flip escapes, then re-do 1-flip
        for _ in range(8):
            if time.time() > deadline:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        score = max_val / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()
    
    # Now do iterated local search from best_c with random perturbations
    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f))
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        # Try 1-flip
        best_flip_1 = -1
        best_max_1 = current_max
        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_1 - 1e-12:
                best_max_1 = max_val
                best_flip_1 = n
        
        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = current_f - 2.0 * current_c[best_flip_1] * D[:, best_flip_1]
            current_max = best_max_1
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
            continue
        
        # Try 2-flip
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
            current_max = best_max_2
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
            continue
        
        # Stuck: random restart by flipping 2-4 random bits
        n_flips = rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
        indices = rng.choice(24, n_flips, replace=False)
        current_c[indices] *= -1.0
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        # Do a quick 1-flip local search after restart
        current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max)
        if current_max / np.sqrt(24) < best_score:
            best_score = current_max / np.sqrt(24)
            best_c = current_c.copy()
    
    return [int(x) for x in best_c]
