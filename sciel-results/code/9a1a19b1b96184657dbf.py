"""Low discrepancy on all arithmetic progressions - fast SA + spectral init + steepest descent."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs, n=256, limit=None):
    """Compute max absolute signed sum over any arithmetic progression in [0..n-1].
    signs: list of +1/-1.
    limit: if provided, returns limit as soon as discrepancy >= limit is found.
    """
    max_disc = 0
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                continue
            min_prefix = 0
            max_prefix = 0
            cur_sum = 0
            max_sub = 0
            min_sub = 0
            idx = r
            while idx < n:
                cur_sum += signs[idx]
                diff = cur_sum - min_prefix
                if diff > max_sub:
                    max_sub = diff
                    if limit is not None and max_sub >= limit:
                        return limit
                if cur_sum < min_prefix:
                    min_prefix = cur_sum
                diff_min = cur_sum - max_prefix
                if diff_min < min_sub:
                    min_sub = diff_min
                    if limit is not None and (-min_sub) >= limit:
                        return limit
                if cur_sum > max_prefix:
                    max_prefix = cur_sum
                idx += d
            ap_max = max(max_sub, -min_sub)
            if ap_max > max_disc:
                max_disc = ap_max
                if limit is not None and max_disc >= limit:
                    return limit
    return max_disc


def _spectral_regularize(signs, n, max_freq=16, threshold=0.5):
    """Apply soft-thresholding to DFT coefficients to suppress high-frequency components."""
    x = np.array(signs, dtype=np.float64)
    X = np.fft.rfft(x)
    for k in range(1, len(X)):
        if k > max_freq:
            X[k] = np.sign(X[k]) * max(0.0, abs(X[k]) - threshold)
    x_recon = np.fft.irfft(X, n=n)
    signs_out = np.where(x_recon >= 0, 1, -1).astype(int)
    return list(signs_out)


def _steepest_descent_polish(signs, deadline, max_iter=50):
    """Steepest descent: repeatedly find the best single flip and apply it."""
    n = len(signs)
    current = signs[:]
    current_val = _max_ap_discrepancy(current)
    best = current[:]
    best_val = current_val
    
    for _ in range(max_iter):
        if time.time() > deadline - 3:
            break
        best_flip = -1
        best_flip_val = current_val
        for i in range(n):
            current[i] = -current[i]
            v = _max_ap_discrepancy(current, limit=best_flip_val)
            if v < best_flip_val:
                best_flip_val = v
                best_flip = i
            current[i] = -current[i]
        if best_flip == -1:
            break
        current[best_flip] = -current[best_flip]
        current_val = best_flip_val
        if current_val < best_val:
            best_val = current_val
            best = current[:]
    return best, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    n = 256
    rng = random.Random(42)
    
    # Phase 1: Generate diverse candidates using spectral regularization
    candidates = []
    
    # Thue-Morse
    signs_tm = [0] * n
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
    candidates.append(signs_tm)
    
    # Alternating
    signs_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_alt)
    
    # Random initializations with spectral regularization
    for seed in range(6):
        rng2 = random.Random(seed)
        init = [rng2.choice((-1, 1)) for _ in range(n)]
        candidates.append(init)
        for max_freq in [8, 12, 16]:
            for threshold in [0.3, 0.5]:
                reg = _spectral_regularize(init, n, max_freq, threshold)
                candidates.append(reg)
    
    # Evaluate all candidates and pick the best
    best_signs = None
    best_disc = float('inf')
    for c in candidates:
        d = _max_ap_discrepancy(c)
        if d < best_disc:
            best_disc = d
            best_signs = c[:]
    
    current_signs = best_signs[:]
    current_disc = best_disc
    
    # Phase 2: Simulated Annealing
    T = 8.0
    T_min = 0.05
    cooling = 0.9992
    stall_counter = 0
    max_stalls = 300
    
    while time.time() < deadline - 10:
        i = rng.randrange(n)
        old_val = current_signs[i]
        current_signs[i] = -old_val
        
        # Use limit-based early termination for speed
        limit = current_disc + int(T * 3) + 2
        new_disc = _max_ap_discrepancy(current_signs, n, limit)
        delta = new_disc - current_disc
        
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = current_signs[:]
                stall_counter = 0
            else:
                stall_counter += 1
        else:
            current_signs[i] = old_val
            stall_counter += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        if stall_counter > max_stalls:
            current_signs = best_signs[:]
            current_disc = best_disc
            T = 8.0
            stall_counter = 0
    
    # Phase 3: Steepest descent polish
    if time.time() < deadline - 5:
        polished, polished_val = _steepest_descent_polish(best_signs, deadline, max_iter=80)
        if polished_val < best_disc:
            best_signs = polished
            best_disc = polished_val
    
    return best_signs
# EVOLVE-BLOCK-END
