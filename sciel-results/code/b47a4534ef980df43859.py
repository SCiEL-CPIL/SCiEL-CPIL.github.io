"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
# Precompute all arithmetic progressions (start, step, length) contained in [1..N]
# Represent each AP by its 0-based indices into the sign array.
_AP_INDICES = []

def _build_aps():
    """Build list of index arrays for all APs in [0..N-1]."""
    aps = []
    for start in range(N):
        for step in range(1, N):
            # Length is at least 1
            # indices: start, start+step, start+2*step, ... < N
            idx = start
            while idx < N:
                # We need to consider all sub-lengths? 
                # The task says "ANY arithmetic progression ... any start, positive step and length including one".
                # This implies for a fixed start and step, we consider the AP of length 1, length 2, etc.
                # Actually, an AP is defined by start, step, and length.
                # So for start=0, step=1, we have APs: [0], [0,1], [0,1,2], ...
                # For start=0, step=2, we have APs: [0], [0,2], [0,2,4], ...
                # 
                # Efficient way:
                # Iterate start, step.
                # Generate the full sequence of indices.
                # For each prefix of this sequence, it is an AP.
                #
                # However, storing all prefixes for all start/step might be huge.
                # Total number of APs:
                # Sum_{start=0}^{N-1} Sum_{step=1}^{N-1} floor((N-1-start)/step) + 1
                # This is roughly N^2 log N. For N=256, 256^2 * ~6 = ~400k APs.
                # Each AP has average length ~ N/3? No, small steps have long APs.
                # Storing indices as lists of lists might be memory heavy but 400k lists is okay.
                # But computing max sum requires summing these.
                # Better: Precompute a matrix? No, sparse.
                # Better: For each AP, store the tuple of indices.
                # Then to evaluate a sign vector, we sum x[i] for i in ap.
                
                pass
            # Let's re-structure the loop for efficiency
            pass
    return aps

def _generate_aps():
    aps = []
    for start in range(N):
        for step in range(1, N):
            # The AP is determined by start and step.
            # The possible lengths are 1, 2, ..., L_max where L_max = floor((N-1-start)/step) + 1
            # For each length l, the AP is start, start+step, ..., start+(l-1)*step
            # Note: The AP of length l is a prefix of the AP of length L_max.
            # We can just store the full list of indices for the maximal AP, 
            # and when evaluating, we need sums of prefixes?
            # No, the discrepancy is the max over ALL APs.
            # So we need to check the sum for EVERY prefix of every maximal AP?
            # Yes.
            # 
            # Alternative:
            # For each start, step, generate the indices.
            # For each prefix, add to list of APs.
            # This is O(N^2 log N) APs.
            # 256^2 * 6 = 393k.
            # Each AP is a tuple of indices.
            # Total memory: 393k * avg_len * 8 bytes.
            # Avg len is small for large steps, large for small steps.
            # Sum of lengths = Sum_{start,step} L_max * (L_max+1)/2 ? No, sum of lengths of all prefixes is L_max(L_max+1)/2.
            # Sum_{start,step} L_max^2 / 2.
            # This might be large.
            # Let's estimate:
            # For step=1, L_max ~ N. Sum ~ N * N^2 = N^3 = 16M.
            # For step=2, L_max ~ N/2. Sum ~ N * (N/2)^2 = N^3/4 = 4M.
            # Total ~ N^3 * sum(1/k^2) ~ N^3 * 1.6 = 26M integers.
            # 26M * 4 bytes = 100MB. This is acceptable.
            
            pass
    
    # Efficient generation:
    for start in range(N):
        for step in range(1, N):
            # Generate maximal AP indices
            max_len = (N - 1 - start) // step + 1
            if max_len < 1:
                continue
            # Indices: start, start+step, ...
            # We need to add all prefixes.
            # Instead of storing all prefixes, we can store the maximal AP and 
            # during evaluation, compute prefix sums?
            # But we need to check max absolute sum over ALL APs.
            # If we store maximal APs, we can compute the prefix sums of the sign vector 
            # restricted to that AP.
            # 
            # Strategy:
            # 1. Precompute list of maximal APs (start, step, max_len).
            # 2. During evaluation of a sign vector x:
            #    For each maximal AP:
            #       Compute the sum of x[start], x[start+step], ... for each prefix length 1..max_len.
            #       Track max absolute sum.
            # 
            # This avoids storing 26M integers.
            # We just store ~400k tuples (start, step, max_len).
            
            aps.append((start, step, max_len))
    return aps

_AP_DATA = _generate_aps()

def _eval_discrepancy(x):
    """
    x: numpy array of shape (N,) with values -1, 1.
    Returns max absolute sum over all APs.
    """
    max_disc = 0
    for start, step, max_len in _AP_DATA:
        # Compute sums for prefixes
        # s_1 = x[start]
        # s_2 = s_1 + x[start+step]
        # ...
        # We can do this in a loop.
        # Optimization: Use numpy slicing?
        # x[start : start + step*max_len : step] gives the array of values.
        # Then prefix sums: np.cumsum(arr)
        # Then max(abs(cumsum))
        
        # Slicing creates a view? No, strided slice creates a new array?
        # In NumPy, x[start:stop:step] creates a view if possible, but cumsum creates a new array.
        # Let's try to avoid creating large temporary arrays if possible.
        # But max_len is small on average.
        
        arr = x[start : start + step*max_len : step]
        sums = np.cumsum(arr)
        local_max = np.max(np.abs(sums))
        if local_max > max_disc:
            max_disc = int(local_max)
            # Early exit if we hit a high value? No, we need the max.
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Initial random guess
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N).astype(np.int8)
    
    # Local search: Flip one bit if it reduces discrepancy.
    # To speed up, we can maintain a cache of AP sums?
    # Or just re-evaluate. 
    # _eval_discrepancy takes ~0.1-0.5s?
    # Let's time it.
    
    t0 = time.time()
    d0 = _eval_discrepancy(x)
    t1 = time.time()
    eval_time = t1 - t0
    # print(f"Eval time: {eval_time}")
    
    best_x = x.copy()
    best_d = d0
    
    # Greedy descent
    # Iterate through positions, try flipping, keep if better.
    # Repeat until no improvement or time limit.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            d = _eval_discrepancy(x)
            if d < best_d:
                best_d = d
                best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]  # revert
                
    # Simulated Annealing if time remains
    if time.time() < deadline - 5:
        temp = 10.0
        cooling = 0.95
        x = best_x.copy()
        cur_d = best_d
        while time.time() < deadline - 2 and temp > 0.1:
            i = rng.randint(0, N)
            x[i] = -x[i]
            new_d = _eval_discrepancy(x)
            delta = new_d - cur_d
            if delta < 0 or rng.rand() < np.exp(-delta / temp):
                cur_d = new_d
                if new_d < best_d:
                    best_d = new_d
                    best_x = x.copy()
            else:
                x[i] = -x[i]
            temp *= cooling
            
    # Convert to list of ints
    return list(best_x)

# EVOLVE-BLOCK-END
