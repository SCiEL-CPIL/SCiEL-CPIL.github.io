import numpy as np
import time

N = 256
DEADLINE = 85.0  # Safety margin

def _thue_morse(n):
    """Generate the first n terms of the Thue-Morse sequence (-1, 1)."""
    # t(0)=1, t(1)=-1, t(2)=-1, t(3)=1 ... (-1)^{popcount(i)}
    # We want values in {-1, 1}.
    # popcount of 0 is 0 -> 1
    # popcount of 1 is 1 -> -1
    res = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Count set bits
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        res[i] = -1 if c % 2 else 1
    return res

def _fast_max_abs_ap_sum(signs):
    """
    Compute max absolute sum over all APs.
    Uses vectorized prefix sum logic for max/min subarray sum.
    """
    n = len(signs)
    max_disc = 1  # Singletons
    
    # Pre-compute slices? No, dynamic slicing is fine.
    for d in range(2, n):
        # For each residue class modulo d
        for r in range(d):
            seq = signs[r::d]
            if len(seq) < 2:
                continue
            
            # Prefix sums P[0]=0, P[i] = sum(seq[0..i-1])
            P = np.concatenate(([0], np.cumsum(seq)))
            
            # Max subarray sum: max_{i<j} P[j] - P[i]
            # min_prefix[j] = min(P[0..j-1])
            # We need min of P[0..j-1] for each j in 1..len(P)-1
            # So we take P[:-1] and accumulate min.
            min_prefix = np.minimum.accumulate(P[:-1])
            # Differences P[1:] - min_prefix
            # Note: P[1:] has length L, min_prefix has length L.
            max_sub = np.max(P[1:] - min_prefix)
            
            # Min subarray sum: min_{i<j} P[j] - P[i]
            max_prefix = np.maximum.accumulate(P[:-1])
            min_sub = np.min(P[1:] - max_prefix)
            
            cur_max = max(max_sub, -min_sub)
            if cur_max > max_disc:
                max_disc = cur_max
                if max_disc > 4:  # Early exit if already worse than target (target is likely 3)
                    return max_disc
                    
    return max_disc

def _greedy_improve(signs, time_limit=5.0):
    """
    Perform greedy local search: flip the sign that most reduces max discrepancy.
    If multiple, pick random.
    """
    n = len(signs)
    current = signs.copy()
    current_max = _fast_max_abs_ap_sum(current)
    best = current.copy()
    best_max = current_max
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_flip_pos = -1
        best_flip_val = current_max + 1 # Must be strictly better or equal? Let's say strictly better first.
        
        # To speed up, we can check all positions
        for pos in range(n):
            old_sign = current[pos]
            current[pos] = -old_sign
            new_max = _fast_max_abs_ap_sum(current)
            current[pos] = old_sign
            
            if new_max < best_flip_val:
                best_flip_val = new_max
                best_flip_pos = pos
                # If we hit 1, we can't do better (min is 1)
                if new_max == 1:
                    break
        
        if best_flip_pos != -1 and best_flip_val < current_max:
            current[best_flip_pos] = -current[best_flip_pos]
            current_max = best_flip_val
            if current_max < best_max:
                best_max = current_max
                best = current.copy()
            improved = True
        else:
            # Try to find a move that keeps max same but improves "second best" or just random walk?
            # For now, stop if no strict improvement.
            break
            
    return best, best_max

def construct():
    start_time = time.time()
    
    # 1. Initialize with Thue-Morse
    signs = _thue_morse(N)
    
    # 2. Greedy improvement to get into a local minimum
    signs, current_max = _greedy_improve(signs, time_limit=5.0)
    
    # 3. Simulated Annealing
    # Use remaining time
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 2.0 # 2s safety
    
    if remaining > 1.0:
        current_signs = signs.copy()
        current_max = _fast_max_abs_ap_sum(current_signs)
        best_signs = current_signs.copy()
        best_max = current_max
        
        temp = 3.0
        cooling = 0.999
        
        while time.time() - start_time < DEADLINE:
            # Pick random position
            pos = np.random.randint(N)
            old_sign = current_signs[pos]
            new_sign = -old_sign
            current_signs[pos] = new_sign
            
            new_max = _fast_max_abs_ap_sum(current_signs)
            delta = new_max - current_max
            
            if delta <= 0 or np.random.random() < np.exp(-delta / temp):
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_signs = current_signs.copy()
            else:
                current_signs[pos] = old_sign
            
            temp *= cooling
            if temp < 0.01:
                break
                
        signs = best_signs
        current_max = best_max
        
    return [int(s) for s in signs]
