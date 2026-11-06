"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_aps(n=256):
    """Generate all arithmetic progressions within [1, n] (1-indexed positions).
    Returns list of lists of 0-indexed positions."""
    aps = []
    for start in range(n):
        for step in range(1, n - start):
            # length is at least 2 for step>0 in AP definition usually, but task says "including one"
            # Actually, "any start, positive step and length including one"
            # If length is 1, the sum is just the sign. Max abs is 1.
            # If step > 0, length >= 2 usually implies distinct terms.
            # Let's generate all sequences a, a+d, a+2d... <= n-1
            # We include length 1? "including one".
            # A singleton {i} is an AP with step anything? Or just length 1.
            # Sum is s_i. Abs is 1.
            # So the constraint from singletons is max_abs >= 1.
            # We can ignore singletons in optimization as they are constant 1, unless we can make them 0? No, signs are +/-1.
            # So the baseline max is 1.
            # We need to minimize max over all APs of length >= 2?
            # Wait, if an AP has length 1, sum is +/-1.
            # If an AP has length 2, sum is s_i + s_j. Can be -2, 0, 2.
            # We want to minimize the maximum absolute sum.
            
            seq = []
            curr = start
            while curr < n:
                seq.append(curr)
                curr += step
            if len(seq) >= 2:
                aps.append(seq)
    return aps

def _eval_max_abs(signs, aps):
    """Compute the maximum absolute sum over all APs."""
    max_val = 0
    for ap in aps:
        s = 0
        for idx in ap:
            s += signs[idx]
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def _eval_max_abs_fast(signs, aps):
    """Vectorized evaluation if possible, or optimized loop."""
    # Pre-compute indices as arrays?
    # For now, simple loop is safer for correctness, but might be slow.
    # Let's try to speed up by pre-computing AP index lists as numpy arrays.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s safety margin
    
    n = 256
    # Generate all APs
    aps = []
    for start in range(n):
        for step in range(1, n - start):
            seq = list(range(start, n, step))
            if len(seq) >= 2:
                aps.append(seq)
    
    # Convert to numpy arrays for faster summation?
    # Or just keep as lists.
    # Number of APs is roughly n^2/2 = 32k.
    # Each AP length varies.
    # Total operations ~ sum(len(ap)) ~ n^2 log n? No, sum_{d=1}^n (n/d) * n ~ n^2 log n.
    # 256^2 * log(256) ~ 65k * 8 ~ 500k operations per eval.
    # 500k ops in Python is ~0.05-0.1s.
    # So we can do ~1000-2000 evals in 80s.
    
    # Greedy construction
    signs = np.zeros(n, dtype=np.int8)
    
    # Initialize with random or zero?
    # Greedy: pick sign for each position to minimize current max discrepancy.
    
    # Pre-compute which APs include each position?
    # pos_to_aps = {i: [list of ap indices]}
    # This allows local update.
    
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            pos_to_aps[idx].append(i)
    
    # Current sums for each AP
    ap_sums = np.zeros(len(aps), dtype=np.int8)
    # Current max abs
    # We need to maintain max.
    # A heap or just scan? Scan is 32k, cheap.
    
    def current_max():
        return np.max(np.abs(ap_sums))
    
    # Greedy loop
    for i in range(n):
        best_sign = 1
        best_max = 1000
        
        # Try +1
        signs[i] = 1
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] += 1
        max_plus = np.max(np.abs(ap_sums))
        
        # Try -1
        signs[i] = -1
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] -= 1
        max_minus = np.max(np.abs(ap_sums))
        
        if max_plus <= max_minus:
            # Keep +1
            signs[i] = 1
            for ap_idx in pos_to_aps[i]:
                ap_sums[ap_idx] += 1
            # Wait, I set signs[i]=-1 and updated sums to -1 state.
            # If I choose +1, I need to revert to +1 state.
            # Current state is -1.
            # So add 2 to all affected APs.
            for ap_idx in pos_to_aps[i]:
                ap_sums[ap_idx] += 2
            # signs[i] is already 1? No, I set it to -1.
            signs[i] = 1
            # Wait, the logic above was messy. Let's restart the logic block.
            
        # Let's do it cleanly.
        
        # Reset i
        signs[i] = 0
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] -= signs[i] # signs[i] is 0, so no change? 
            # This is getting complicated.
            
    # Let's implement a cleaner greedy.
    
    # Re-initialize
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(len(aps), dtype=np.int8)
    
    for i in range(n):
        # Try +1
        signs[i] = 1
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] += 1
        max_plus = np.max(np.abs(ap_sums))
        
        # Try -1
        # Revert +1
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] -= 1
        signs[i] = -1
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] -= 1
        max_minus = np.max(np.abs(ap_sums))
        
        if max_plus <= max_minus:
            # Choose +1
            signs[i] = 1
            for ap_idx in pos_to_aps[i]:
                ap_sums[ap_idx] += 1
        else:
            # Choose -1 (already set)
            signs[i] = -1
            # Sums already reflect -1
            
    # Local search / Hill Climbing
    current_signs = signs.copy()
    current_max = np.max(np.abs(ap_sums))
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.integers(0, n)
        
        # Flip sign
        old_sign = current_signs[i]
        new_sign = -old_sign
        
        # Update sums
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] += (new_sign - old_sign)
            
        new_max = np.max(np.abs(ap_sums))
        
        if new_max < current_max:
            current_signs[i] = new_sign
            current_max = new_max
        else:
            # Revert
            for ap_idx in pos_to_aps[i]:
                ap_sums[ap_idx] += (old_sign - new_sign)
            current_signs[i] = old_sign
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    return [int(x) for x in current_signs]
# EVOLVE-BLOCK-END
