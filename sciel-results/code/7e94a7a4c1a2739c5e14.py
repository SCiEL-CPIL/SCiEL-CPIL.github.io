"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n: t_k = (-1)^{popcount(k)}"""
    seq = [1] * n
    for i in range(n):
        # popcount of i
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq[i] = 1 if c % 2 == 0 else -1
    return seq

def _compute_max_discrepancy(seq):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [0, n-1] (0-indexed), with any start, positive step, and length >= 1.
    """
    n = len(seq)
    max_abs = 0
    # For each step d from 1 to n-1
    for d in range(1, n):
        # For each start s from 0 to n-1
        for s in range(n):
            curr = 0
            idx = s
            while idx < n:
                curr += seq[idx]
                if curr > max_abs:
                    max_abs = curr
                elif curr < -max_abs:
                    max_abs = -curr
                idx += d
    return max_abs

def _compute_max_discrepancy_fast(seq):
    """
    Faster computation using numpy for the inner loop where possible.
    Still O(n^2 * n/d) total but with vectorized operations.
    """
    n = len(seq)
    arr = np.array(seq, dtype=np.int32)
    max_abs = 0
    for d in range(1, n):
        # For each residue class mod d, compute cumulative sums
        for r in range(d):
            # Indices: r, r+d, r+2d, ...
            idxs = np.arange(r, n, d)
            if len(idxs) == 0:
                continue
            vals = arr[idxs]
            cumsum = np.cumsum(vals)
            local_max = max(int(np.max(cumsum)), -int(np.min(cumsum)))
            if local_max > max_abs:
                max_abs = local_max
    return max_abs

def _flip_effect(seq, pos):
    """
    Compute the change in max discrepancy if we flip position pos.
    Returns (new_max, delta) where delta = new_max - current_max.
    Actually, we need to recompute or estimate. For efficiency in SA,
    we'll just recompute the full cost when needed, but use a faster method.
    
    For now, return None to indicate we need full recomputation.
    """
    pass

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin from 60s limit
    
    # Initialize with Thue-Morse sequence
    seq = _thue_morse(n)
    current_cost = _compute_max_discrepancy_fast(seq)
    best_seq = list(seq)
    best_cost = current_cost
    
    rng = random.Random(12345)
    
    # Simulated Annealing
    # Temperature schedule: start high, decay geometrically
    T_start = 10.0
    T_end = 0.1
    # We'll do iterations until deadline
    iteration = 0
    T = T_start
    
    # Estimate how many iterations we can do
    # Each iteration: pick random pos, flip, compute cost (O(n^2) ~ 65536 * avg length)
    # Let's benchmark: measure time for one cost computation
    t0 = time.time()
    _compute_max_discrepancy_fast(seq)
    t1 = time.time()
    cost_time = t1 - t0
    # Reserve some time
    available_time = deadline - time.time() - 2.0
    max_iters = int(available_time / (cost_time + 0.001)) if cost_time > 0 else 100000
    max_iters = max(max_iters, 100)
    
    # Geometric cooling
    if max_iters > 1:
        decay = (T_end / T_start) ** (1.0 / max_iters)
    else:
        decay = 1.0
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        
        # Flip
        seq[pos] = -seq[pos]
        
        # Compute new cost
        new_cost = _compute_max_discrepancy_fast(seq)
        
        # Accept if better, or with probability exp(-(new_cost - current_cost)/T)
        if new_cost <= current_cost:
            # Accept
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = list(seq)
        else:
            # Maybe accept
            delta = new_cost - current_cost
            prob = np.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < prob:
                current_cost = new_cost
            else:
                # Reject: flip back
                seq[pos] = -seq[pos]
        
        # Cool down
        T *= decay
        if T < T_end:
            T = T_end
    
    return best_seq
# EVOLVE-BLOCK-END
