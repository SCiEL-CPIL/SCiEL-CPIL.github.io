"""Low discrepancy on all arithmetic progressions - spectral regularization + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs, limit=None):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1.
    limit: if provided, returns limit as soon as discrepancy >= limit is found.
    Returns the max absolute sum.
    """
    n = len(signs)
    max_abs = 0
    
    for d in range(1, n + 1):
        for r in range(d):
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
                    if limit is not None and -min_sub >= limit:
                        return limit
                if cur_sum > max_prefix:
                    max_prefix = cur_sum
                
                idx += d
                
            ap_max = max(max_sub, -min_sub)
            if ap_max > max_abs:
                max_abs = ap_max
                if limit is not None and max_abs >= limit:
                    return limit
                    
    return max_abs

def _spectral_regularize(signs, n, max_freq=16, threshold=0.5):
    """Apply soft-thresholding to DFT coefficients to suppress high-frequency components.
    This promotes smoothness which should reduce local sum discrepancies.
    """
    x = np.array(signs, dtype=np.float64)
    X = np.fft.rfft(x)
    
    # Soft-threshold high frequency components
    for k in range(1, len(X)):
        if k > max_freq:
            # Soft threshold: shrink towards zero
            X[k] = np.sign(X[k]) * max(0, abs(X[k]) - threshold)
    
    # Inverse FFT and round to nearest sign
    x_recon = np.fft.irfft(X, n=n)
    # Round to nearest +1 or -1
    signs_out = np.where(x_recon >= 0, 1, -1).astype(int)
    return list(signs_out)

def _greedy_construct(n):
    """Greedy construction: at each position, choose sign that minimizes worst AP sum so far."""
    signs = [0] * n
    
    for i in range(n):
        # Try both signs
        best_sign = 1
        best_disc = float('inf')
        
        for s in (1, -1):
            signs[i] = s
            # Compute current max discrepancy (only need to check APs ending at or before i)
            # For efficiency, just compute full discrepancy with current partial signs
            # Pad with zeros for positions > i (they won't affect APs ending at i)
            partial = signs[:] + [0] * (n - i - 1)
            disc = _max_ap_discrepancy(partial, limit=best_disc)
            if disc < best_disc:
                best_disc = disc
                best_sign = s
        
        signs[i] = best_sign
    
    return signs

def _local_search(signs, deadline, rng):
    """Steepest descent local search with time budget."""
    n = len(signs)
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs)
    best_signs = current_signs[:]
    best_val = current_val
    
    while time.time() < deadline - 2:
        improved = False
        best_flip = -1
        best_flip_val = current_val
        
        for i in range(n):
            current_signs[i] = -current_signs[i]
            v = _max_ap_discrepancy(current_signs, limit=best_flip_val)
            if v < best_flip_val:
                best_flip_val = v
                best_flip = i
            current_signs[i] = -current_signs[i]  # revert
        
        if best_flip != -1:
            current_signs[best_flip] = -current_signs[best_flip]
            current_val = best_flip_val
            improved = True
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs[:]
        
        if not improved:
            break
    
    return best_signs, best_val

def construct():
    deadline = time.time() + 50  # Safety margin
    n = 256
    rng = random.Random(42)
    
    # Phase 1: Spectral regularization from multiple initializations
    candidates = []
    
    # Random initializations with spectral regularization
    for seed in range(8):
        rng2 = random.Random(seed)
        init = [rng2.choice((-1, 1)) for _ in range(n)]
        
        # Apply spectral regularization
        for max_freq in [8, 12, 16, 24]:
            for threshold in [0.3, 0.5, 0.7]:
                reg_signs = _spectral_regularize(init, n, max_freq, threshold)
                candidates.append(reg_signs)
        
        # Also add the original
        candidates.append(init)
    
    # Greedy construction
    greedy = _greedy_construct(n)
    candidates.append(greedy)
    
    # Phase 2: Local search on top candidates
    best_signs = None
    best_val = float('inf')
    
    # Sort candidates by initial discrepancy
    candidate_vals = []
    for c in candidates:
        v = _max_ap_discrepancy(c)
        candidate_vals.append((v, c))
    candidate_vals.sort()
    
    # Take top 5 candidates and run local search
    for v, c in candidate_vals[:5]:
        if time.time() > deadline - 10:
            break
        ls_signs, ls_val = _local_search(c, deadline, rng)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs
    
    # Phase 3: Additional local search iterations if time permits
    if best_signs is not None and time.time() < deadline - 5:
        ls_signs, ls_val = _local_search(best_signs, deadline, rng)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs
    
    # Phase 4: Random restarts if time permits
    while time.time() < deadline - 8:
        init = [rng.choice((-1, 1)) for _ in range(n)]
        # Apply spectral regularization
        reg_signs = _spectral_regularize(init, n, max_freq=12, threshold=0.5)
        ls_signs, ls_val = _local_search(reg_signs, deadline - 3, rng)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs
    
    if best_signs is None:
        best_signs = greedy
    
    return best_signs
# EVOLVE-BLOCK-END
