"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_max_discrepancy_fast(seq):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [0, n-1] (0-indexed), with any start, positive step, and length >= 1.
    """
    n = len(seq)
    arr = np.array(seq, dtype=np.int32)
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            idxs = np.arange(r, n, d)
            if len(idxs) == 0:
                continue
            vals = arr[idxs]
            cumsum = np.cumsum(vals)
            local_max = max(int(np.max(cumsum)), -int(np.min(cumsum)))
            if local_max > max_abs:
                max_abs = local_max
    return max_abs

def _greedy_construct(n, rng):
    """
    Build a sequence greedily: for each position, choose the sign that minimizes
    the maximum absolute discrepancy of the partial sequence.
    """
    seq = [0] * n
    for i in range(n):
        best_sign = 1
        best_cost = float('inf')
        for sign in [1, -1]:
            seq[i] = sign
            # Compute cost for the prefix
            # We only need to check APs within [0, i]
            # To be safe and simple, we compute for the whole prefix
            # Optimization: We can just check APs ending at i or contained in [0, i]
            # But _compute_max_discrepancy_fast works on the whole array.
            # For n=256, this is 256 * 2 * O(n^2) ~ 256 * 2 * 32768 ~ 16M ops.
            # In Python/NumPy, this might be slow.
            # Let's use a faster check for the prefix.
            
            # Actually, let's just use the fast function on the prefix.
            # But the fast function assumes length n.
            # Let's create a helper for prefix.
            
            # For simplicity in the first pass, let's just use the full function 
            # but only on the slice seq[:i+1].
            # Wait, _compute_max_discrepancy_fast takes a list.
            # Let's pass a list of zeros for the rest? No, that changes the APs.
            # We must only consider APs within [0, i].
            
            # Let's write a quick inline check for the prefix.
            # Since i is small initially, this is fast.
            # As i grows, it gets slower.
            
            # Alternative: Just use random initialization if greedy is too slow.
            # But the inspiration suggests greedy is good.
            # Let's optimize the cost check for prefix.
            
            # We can just call a modified version of the fast check that takes n_eff.
            
            cost = _prefix_max_discrepancy(seq, i + 1)
            if cost < best_cost:
                best_cost = cost
                best_sign = sign
        
        seq[i] = best_sign
        
    return seq

def _prefix_max_discrepancy(seq, n_eff):
    """Compute max discrepancy for the first n_eff elements."""
    arr = np.array(seq[:n_eff], dtype=np.int32)
    max_abs = 0
    for d in range(1, n_eff):
        for r in range(d):
            idxs = np.arange(r, n_eff, d)
            if len(idxs) == 0:
                continue
            vals = arr[idxs]
            cumsum = np.cumsum(vals)
            local_max = max(int(np.max(cumsum)), -int(np.min(cumsum)))
            if local_max > max_abs:
                max_abs = local_max
    return max_abs

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin from 90s limit
    
    rng = random.Random(42)
    
    # Phase 1: Greedy Construction
    # This takes some time, but gives a good starting point.
    # Let's see if it fits in the budget.
    # If greedy is too slow, we fall back to random.
    
    # Actually, greedy with full prefix checks is O(n^3) roughly.
    # 256^3 ~ 16M. With NumPy overhead, it might be 1-2 seconds.
    # Let's try it.
    
    seq = _greedy_construct(n, rng)
    current_cost = _compute_max_discrepancy_fast(seq)
    best_seq = list(seq)
    best_cost = current_cost
    
    if time.time() > deadline - 10:
        return best_seq
        
    # Phase 2: Local Search (Simulated Annealing / Hill Climbing)
    # To make this fast, we need to update the cost efficiently.
    # However, implementing an incremental update for "max over all APs" is complex.
    # A simpler approach: Use a "fast enough" cost function and do many iterations.
    # The current _compute_max_discrepancy_fast is O(n^2) ~ 65k operations.
    # In Python, this is ~1-2 ms per call.
    # In 70 seconds, we can do ~35,000 - 70,000 iterations.
    # This is much better than the previous program's ~1000-2000 iterations.
    
    # We will use a simple hill climbing with random restarts if stuck.
    
    T = 2.0
    T_min = 0.1
    decay = 0.999
    
    # Pre-compute indices for each AP to speed up cost calculation?
    # The fast function already does this.
    
    iteration = 0
    while time.time() < deadline:
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        
        # Flip
        seq[pos] = -seq[pos]
        
        # Compute new cost
        new_cost = _compute_max_discrepancy_fast(seq)
        
        # Accept if better, or with probability exp(-(new_cost - current_cost)/T)
        if new_cost <= current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = list(seq)
                # Optional: Reset T if we found a new best?
                # T = max(T, 1.0) 
        else:
            delta = new_cost - current_cost
            prob = np.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < prob:
                current_cost = new_cost
            else:
                # Reject: flip back
                seq[pos] = -seq[pos]
        
        # Cool down
        T *= decay
        if T < T_min:
            T = T_min
            
        iteration += 1
        
        # Periodic restart if stuck?
        # If we haven't improved in a long time, maybe restart from best?
        # For now, simple SA.
        
    return best_seq
# EVOLVE-BLOCK-END
