import numpy as np
import time

def _init_progressions(n):
    """
    Precompute the structure for incremental discrepancy updates.
    Returns:
    - prefix_sums: A 2D list (or list of lists) where prefix_sums[d][r] is a numpy array of prefix sums for progression (d, r).
    - max_indices: A flat array or list to map (d, r) to an index in progression_maxes.
    - progression_maxes: An array storing the current max abs prefix sum for each progression.
    """
    prefix_sums = [[None for _ in range(d)] for d in range(n)]
    progression_maxes = np.zeros(n * n, dtype=np.int32) # Overallocate, or use a list
    # Actually, number of progressions is sum(d for d in 1..n-1) ~ n^2/2.
    # Let's use a list of arrays for simplicity, and a separate list for maxes.
    
    maxes_list = []
    idx_map = {} # (d, r) -> index in maxes_list
    
    for d in range(1, n):
        for r in range(d):
            # Generate indices
            indices = np.arange(r, n, d)
            # We need the signs to compute initial prefix sums.
            # But signs change. So we can't precompute the values, only the structure.
            # We will initialize prefix_sums[d][r] with zeros and fill them in construct()
            # based on the initial signs.
            prefix_sums[d][r] = np.zeros(len(indices), dtype=np.int32)
            maxes_list.append(0)
            idx_map[(d, r)] = len(maxes_list) - 1
            
    return prefix_sums, maxes_list, idx_map

def _compute_initial_maxes(signs, prefix_sums, maxes_list, idx_map, n):
    """
    Fill prefix_sums and maxes_list based on current signs.
    """
    for d in range(1, n):
        for r in range(d):
            indices = np.arange(r, n, d)
            vals = signs[indices]
            ps = np.cumsum(vals)
            prefix_sums[d][r] = ps
            idx = idx_map[(d, r)]
            maxes_list[idx] = int(np.max(np.abs(ps))) if len(ps) > 0 else 0

def _update_maxes_after_flip(i, old_val, signs, prefix_sums, maxes_list, idx_map, n):
    """
    Update prefix sums and maxes after flipping sign at index i.
    Returns the new global max discrepancy.
    """
    delta = -2 * old_val
    signs[i] = -old_val
    
    for d in range(1, n):
        r = i % d
        if r >= d: # Should not happen
            continue
        # Check if i is actually in this progression
        # i = r + k*d => (i-r) must be divisible by d and >= 0
        if (i - r) < 0:
            continue
        k = (i - r) // d
        arr = prefix_sums[d][r]
        if k < len(arr):
            arr[k:] += delta
            idx = idx_map[(d, r)]
            # Update max
            # Optimization: if delta is positive, max might increase. If negative, min might decrease (abs max might increase).
            # Just recalculate max for this small array.
            # len(arr) is small on average.
            new_local_max = int(np.max(np.abs(arr)))
            maxes_list[idx] = new_local_max
            
    # Find global max
    # maxes_list is a list of ints. Converting to numpy array for max is fast.
    # Or just max(maxes_list).
    return max(maxes_list)

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s margin
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = float('inf')
    
    # Precompute structure
    prefix_sums, maxes_list, idx_map = _init_progressions(n)
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        elif restart == 1:
            signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        else:
            signs = rng.choice([-1, 1], size=n).astype(np.int8)
        
        # Initialize discrepancy structures
        _compute_initial_maxes(signs, prefix_sums, maxes_list, idx_map, n)
        cur_score = max(maxes_list)
        
        if cur_score < best_score:
            best_score = cur_score
            best_signs = signs.copy()
        
        improved = True
        passes = 0
        max_passes = 50 # Increase passes since each step is faster
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            idx_order = rng.permutation(n)
            
            for i in idx_order:
                if time.time() > deadline - 2:
                    break
                
                old_val = signs[i]
                
                # Try flip
                new_score = _update_maxes_after_flip(i, old_val, signs, prefix_sums, maxes_list, idx_map, n)
                
                if new_score < cur_score:
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_signs = signs.copy()
                    improved = True
                else:
                    # Revert
                    # We need to revert the changes to prefix_sums
                    # This is tricky. We modified arr[k:] += delta.
                    # To revert, we do arr[k:] -= delta.
                    # And revert signs[i].
                    signs[i] = old_val
                    # Revert prefix sums
                    for d in range(1, n):
                        r = i % d
                        if (i - r) < 0:
                            continue
                        k = (i - r) // d
                        arr = prefix_sums[d][r]
                        if k < len(arr):
                            arr[k:] -= -2 * old_val # delta was -2*old_val, so subtract delta
                            # Wait, delta = -2 * old_val.
                            # We did arr += delta.
                            # To revert, arr -= delta.
                            # So arr -= (-2 * old_val) => arr += 2 * old_val.
                            # Let's just recalculate the max for this progression to be safe?
                            # No, we just need to revert the array values.
                            # The maxes_list entry is now stale if we don't update it.
                            # We should update maxes_list for reverted progressions too.
                            idx = idx_map[(d, r)]
                            maxes_list[idx] = int(np.max(np.abs(arr)))
                            
            if time.time() > deadline - 2:
                break
                
    if best_signs is None:
        best_signs = np.ones(n, dtype=np.int8)
        
    return [int(x) for x in best_signs]
