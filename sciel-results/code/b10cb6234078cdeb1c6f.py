"""Optimized program: Simulated Annealing with FFT-based score evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

def _compute_score_fft(U, fft_cache=None):
    """Compute the score for a given set U using FFT for speed."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = int(U[-1])
    if max_u == 0:
        return 1.0
    
    # Use FFT to compute sumset and diffset sizes efficiently
    # Create indicator array
    size = 2 * max_u + 1
    indicator = np.zeros(size, dtype=np.float64)
    indicator[U] = 1.0
    
    # Sumset: convolution of indicator with itself
    # U+U has values from 0 to 2*max_u
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    f = np.fft.rfft(indicator, n=fft_size)
    conv = np.fft.irfft(f * f, n=fft_size)
    # Sumset elements are indices where conv > 0.5 (threshold for floating point)
    sumset_mask = conv[:2 * max_u + 1] > 0.5
    len_sum = int(np.sum(sumset_mask))
    
    # Diffset: |U-U| = number of distinct differences
    # U-U has values from -max_u to max_u
    # We can compute this as: for each d in [-max_u, max_u], check if exists a,b in U with a-b=d
    # Alternative: use the fact that |U-U| = 2*|U+(-U)| - 1 (since 0 is always in both)
    # Actually, let's just compute it directly for small sets, or use a different FFT trick
    # For diffset: shift U by max_u, then U' = U + max_u, and U'-U' gives same structure
    # Better: just use direct computation for diffset since it's the bottleneck
    # Actually, let's use a smarter approach:
    # |U-U| can be computed via: for each pair, but that's O(n^2)
    # For FFT: consider the autocorrelation
    # The number of pairs (a,b) with a-b = d is the autocorrelation at lag d
    # |U-U| = number of d where autocorrelation[d] > 0
    
    # Autocorrelation via FFT
    g = np.fft.rfft(indicator, n=fft_size)
    autocorr = np.fft.irfft(g * np.conj(g), n=fft_size)
    # autocorr[k] for k=0..max_u gives count of pairs with diff = k
    # autocorr[fft_size-k] gives count of pairs with diff = -k
    diff_counts = np.zeros(2 * max_u + 1, dtype=np.float64)
    diff_counts[max_u] = autocorr[0]  # diff = 0
    for d in range(1, max_u + 1):
        diff_counts[max_u + d] = autocorr[d]  # diff = d
        diff_counts[max_u - d] = autocorr[d]  # diff = -d
    
    len_diff = int(np.sum(diff_counts > 0.5))
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _sa_search(initial_set, max_val, target_size, timeout=30.0):
    """Simulated Annealing to find a good set."""
    start_time = time.time()
    
    # Initialize
    current = set(initial_set)
    if 0 not in current:
        current.add(0)
    current = sorted(current)
    current_score = _compute_score_fft(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.9995  # Cooling rate
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        
        if T < T_min:
            break
        
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element
            new_elem = random.randint(0, max_val)
            if new_elem not in current:
                new_set = list(current) + [new_elem]
            else:
                continue
        elif move_type < 0.7:
            # Remove a random element (not 0)
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set = [x for x in current if x != rem]
        elif move_type < 0.85:
            # Swap: remove one, add another
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            add = random.randint(0, max_val)
            new_set = [x for x in current if x != rem]
            if add not in new_set:
                new_set.append(add)
        else:
            # Perturb: change a random element to a nearby value
            if len(current) <= 1:
                continue
            idx = random.randint(0, len(current) - 1)
            if idx == 0 and 0 in current:
                # Don't perturb 0
                idx = random.randint(1, len(current) - 1) if len(current) > 1 else 0
            old_val = current[idx]
            delta = random.choice([-3, -2, -1, 1, 2, 3, 5, -5])
            new_val = old_val + delta
            if new_val < 0 or new_val > max_val:
                new_val = random.randint(0, max_val)
            new_set = list(current)
            new_set[idx] = new_val
            # Ensure uniqueness
            if len(set(new_set)) != len(new_set):
                continue
        
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
        
        new_score = _compute_score_fft(new_set)
        
        # Acceptance criterion
        if new_score >= current_score or random.random() < np.exp((new_score - current_score) / T):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Start with a reasonable initial set
    best_result = [0, 1, 2, 4]
    best_score = _compute_score_fft(best_result)
    
    # Multiple SA runs with different initializations and parameters
    # Use structured initial sets based on known good constructions
    
    # Initial set 1: Geometric progression
    init1 = [0] + [2**i for i in range(1, 8)]  # 0, 2, 4, 8, 16, 32, 64, 128
    init1 = [x for x in init1 if x <= 200]
    
    # Initial set 2: Sparse arithmetic-like
    init2 = [0, 1, 3, 7, 15, 31, 63, 127]
    
    # Initial set 3: Mixed
    init3 = [0, 1, 2, 5, 10, 20, 40, 80, 160]
    
    # Initial set 4: Random sparse
    init4 = {0}
    for _ in range(15):
        init4.add(random.randint(1, 300))
    init4 = sorted(init4)
    
    # Initial set 5: Based on the paper's construction hints
    init5 = [0, 1, 2, 4, 8, 17, 36, 72, 144, 288]
    
    initial_sets = [init1, init2, init3, init4, init5]
    
    # Run SA with different max_val ranges
    max_vals = [100, 200, 400, 800]
    
    for max_val in max_vals:
        if time.time() > deadline - 10:
            break
        
        for init in initial_sets:
            if time.time() > deadline - 5:
                break
            
            # Scale initial set to fit within max_val
            if init and max(init) > max_val:
                scale = max_val / max(init)
                scaled = sorted(set([int(x * scale) for x in init]))
                if 0 not in scaled:
                    scaled.insert(0, 0)
                init_scaled = scaled
            else:
                init_scaled = list(init)
            
            # Run SA with time budget proportional to remaining time
            remaining = deadline - time.time()
            sa_timeout = min(15.0, remaining * 0.3)
            
            if sa_timeout < 1.0:
                break
            
            result, score = _sa_search(init_scaled, max_val, len(init_scaled), timeout=sa_timeout)
            
            if score > best_score:
                best_score = score
                best_result = result
    
    # Final refinement: run SA from the best result found
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        max_u_best = max(best_result) if best_result else 0
        max_val_refine = min(int(max_u_best * 2) + 50, 1000)
        result, score = _sa_search(best_result, max_val_refine, len(best_result), timeout=min(20.0, remaining - 5))
        if score > best_score:
            best_score = score
            best_result = result
    
    # Final check
    final_score = _compute_score_fft(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END
