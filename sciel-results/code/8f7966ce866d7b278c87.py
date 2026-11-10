import time
import numpy as np
import random

def _fast_check_discrepancy(signs, limit):
    """
    Check if the maximum absolute signed sum over any arithmetic progression
    is <= limit.
    Returns True if max_disc <= limit, False otherwise.
    Uses early exit for speed.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    
    # If limit is very small, we can even check singletons, but limit >= 1 always for n>=1
    # Singletons have sum +/- 1, so if limit < 1, return False. But limit is usually >= 1.
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            
            # Compute prefix sums
            # We can do this in a loop to allow early exit on each element
            # But numpy cumsum is faster for full computation.
            # For early exit, we need to check each prefix sum.
            # Let's use a python loop for the inner check if the sequence is short,
            # or just compute cumsum and check max.
            # For n=256, the longest AP is 256.
            # cumsum + max is very fast in numpy.
            # The overhead is the array slicing arr[r::d].
            
            # Optimization: Only compute if there's a chance to exceed limit?
            # No, we need to know if it exceeds.
            
            prefix_sums = np.cumsum(subseq)
            
            # Check if any prefix sum exceeds limit
            # We can use np.any(np.abs(prefix_sums) > limit)
            if np.any(np.abs(prefix_sums) > limit):
                return False
                
    return True

def _compute_max_discrepancy(signs):
    """
    Compute the exact maximum absolute signed sum over any arithmetic progression.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
                
    return max_disc

def _find_worst_ap(signs):
    """
    Returns (max_disc, start, step, length) for one of the APs achieving max_disc.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    best_start = 0
    best_step = 1
    best_len = 1
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            abs_sums = np.abs(prefix_sums)
            current_max = np.max(abs_sums)
            if current_max > max_disc:
                max_disc = int(current_max)
                best_step = d
                best_start = r
                idx = np.argmax(abs_sums)
                best_len = idx + 1
                
    return max_disc, best_start, best_step, best_len

def _greedy_construct(n):
    """
    Greedy construction: at each step, choose the sign that minimizes the maximum
    absolute prefix sum over all APs ending at the current position.
    """
    signs = [0] * n
    
    for i in range(n):
        best_sign = 1
        best_max_abs = float('inf')
        
        max_abs_pos = 0
        max_abs_neg = 0
        
        for d in range(1, i + 1):
            a = i
            while a >= 0:
                if a == i:
                    prev_sum = 0
                else:
                    prev_sum = 0
                    curr = a
                    while curr < i:
                        prev_sum += signs[curr]
                        curr += d
                
                s_pos = prev_sum + 1
                s_neg = prev_sum - 1
                
                if abs(s_pos) > max_abs_pos:
                    max_abs_pos = abs(s_pos)
                if abs(s_neg) > max_abs_neg:
                    max_abs_neg = abs(s_neg)
                
                a -= d
        
        if max_abs_pos <= max_abs_neg:
            best_sign = 1
            best_max_abs = max_abs_pos
        else:
            best_sign = -1
            best_max_abs = max_abs_neg
            
        signs[i] = best_sign
        
    return signs

def construct():
    n = 256
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # 1. Initial guess: Greedy construction
    signs = _greedy_construct(n)
    
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(42)
    
    # 2. Local Search
    # We try to reduce the discrepancy.
    # If we can't reduce it, we do a random walk within the feasible set (disc <= current_best)
    # to try to find a better configuration later.
    
    while time.time() - start_time < deadline:
        # Find the worst AP in the current configuration
        disc, start, step, length = _find_worst_ap(signs)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
            
        # If we have reached discrepancy 1, we are optimal (for n>=1, min is 1)
        # Actually, can we get 1?
        # If all APs have sum +/-1 or 0?
        # Singletons are +/-1. So max disc is at least 1.
        # If we achieve 1, we are done.
        if best_disc == 1:
            break
        
        # Try to improve: find a flip that reduces max_disc to disc-1
        improved = False
        # Candidate positions: indices in the worst AP
        ap_indices = list(range(start, start + length * step, step))
        
        # Shuffle candidates to randomize
        rng.shuffle(ap_indices)
        
        for idx in ap_indices:
            if time.time() - start_time > deadline:
                break
                
            # Flip
            signs[idx] = -signs[idx]
            
            # Check if we can beat the current best_disc (which is the global best found so far)
            # Actually, we want to see if this flip reduces the discrepancy of the *current* state
            # to something lower than the current state's discrepancy, or specifically lower than best_disc.
            
            # Let's check if the new configuration has max_disc <= best_disc - 1
            # This is the "improvement" check.
            if _fast_check_discrepancy(signs, best_disc - 1):
                # We improved!
                current_disc = _compute_max_discrepancy(signs) # Get exact value
                improved = True
                break
            else:
                # Revert
                signs[idx] = -signs[idx]
        
        if improved:
            continue
        
        # If we couldn't improve the global best, try to move within the feasible set
        # to escape local minima.
        # We accept a flip if it keeps the discrepancy <= best_disc.
        # This is a random walk on the set of solutions with disc <= best_disc.
        
        # Pick a random index from the worst AP (or all indices if AP is small)
        if ap_indices:
            idx = rng.choice(ap_indices)
        else:
            idx = rng.randint(0, n-1)
            
        signs[idx] = -signs[idx]
        
        # Check if we are still within the best known bound
        if _fast_check_discrepancy(signs, best_disc):
            # Accept the move (random walk)
            # Optionally, we could check if this move reduces the "second best" discrepancy
            # but for now, simple random walk is robust.
            pass
        else:
            # Revert
            signs[idx] = -signs[idx]
            
        # Periodically restart from best_signs if stuck?
        # The random walk should explore the basin.
        
    return best_signs
