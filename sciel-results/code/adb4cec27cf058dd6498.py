import time
import random
import numpy as np

_LCM = 232792560
_WEIGHTS = [_LCM // i for i in range(1, 23)]

def _find_collision(indices):
    """
    Returns (is_dissociated, conflict_indices).
    conflict_indices is a set of indices involved in a collision, or None.
    """
    k = len(indices)
    if k == 0:
        return True, None
    
    # Generate sums and masks
    # Using a dict to map sum -> mask
    # To save memory/time, we can stop early if we find a collision
    sum_to_mask = {0: 0}
    
    # Iterate through elements
    # This is O(2^k).
    # For k=22, this is 4M iterations.
    # Python loop might be slow.
    # Let's use NumPy to generate sums, but we need masks to identify the conflict.
    # Alternatively, we can just check distinctness. If not distinct, we need to find WHICH ones.
    # Finding which ones requires storing them.
    
    # Hybrid:
    # 1. Fast check with NumPy (sort/diff).
    # 2. If collision, do the dict approach to find the specific conflict.
    
    vals = np.array([_WEIGHTS[i-1] for i in indices], dtype=np.int64)
    
    # Fast check
    sums = np.array([0], dtype=np.int64)
    for v in vals:
        sums = np.concatenate((sums, sums + v))
    sums.sort()
    if np.all(np.diff(sums) > 0):
        return True, None
        
    # Collision found. Find it.
    # We need to map sums to masks.
    # Since we know there is a collision, we can use a dict.
    # To speed up, we can use the sorted sums to find duplicates?
    # No, dict is easier.
    
    # Reset dict
    sum_to_mask = {0: 0}
    
    # We need to iterate subsets.
    # 2^k is large.
    # But we only need to find ONE collision.
    # We can iterate masks from 1 to 2^k - 1.
    
    # Optimization: We can stop as soon as we find a collision.
    
    # However, iterating 4M masks in Python is ~0.5s.
    # We might do this 5-10 times. 5s total. Acceptable.
    
    # To make it faster, we can use the fact that we already have the sums array?
    # No, the sums array doesn't have masks.
    
    # Let's just use the dict approach directly, but break early.
    # We can generate sums and masks in parallel?
    
    # Actually, we can just use the dict approach from the start?
    # Dict insertion is slower than NumPy concat.
    # NumPy check: ~0.1s for 4M sums.
    # Dict check: ~0.5s for 4M sums.
    # So do NumPy check first.
    
    # Now, to find the conflict:
    # We can iterate through the sorted sums and find adjacent duplicates?
    # No, duplicates might not be adjacent if there are many.
    # But in a sorted array, duplicates are adjacent.
    # So we can find the duplicate value in the sorted array.
    # Then we need to find which masks produce that sum.
    # This requires re-calculating or storing masks.
    
    # Let's just run the dict approach. It's simpler and robust.
    # We can optimize the dict approach by using a list of sums and then sorting?
    # No, dict is fine.
    
    # To speed up the dict approach, we can use a smaller set if possible?
    # No.
    
    # Let's implement the dict approach carefully.
    
    # Note: indices are 1-based in the problem, but 0-based in the list `indices`.
    # We map index i in `indices` to weight _WEIGHTS[i-1].
    
    # We need to return the original indices (1..22) involved in the conflict.
    
    n = len(indices)
    masks = range(1, 1 << n)
    
    # We can't just iterate 1..2^n if n=22 (4M).
    # It's fine.
    
    for mask in masks:
        s = 0
        # Calculate sum for this mask
        # This is O(n) per mask. 4M * 22 = 88M ops. Too slow in Python.
        
        # We need to generate sums efficiently.
        # Use the recursive/iterative building method but store masks.
        
    # Better:
    # Build sums and masks iteratively.
    sums_masks = [(0, 0)]
    for i, idx in enumerate(indices):
        w = _WEIGHTS[idx-1]
        bit = 1 << i
        # Add w to all existing sums
        # This doubles the list.
        # 4M tuples. Memory might be an issue?
        # 4M * (8 bytes sum + 8 bytes mask) = 64MB. OK.
        
        # But creating new list of 4M tuples is slow in Python.
        
    # Let's stick to the NumPy check for "is it valid".
    # For "find conflict", we can use a randomized subset?
    # No, we need the specific conflict to remove the right element.
    
    # Alternative:
    # If we just want to remove *an* element that breaks *a* collision,
    # we can pick a random subset of the current set, check it, and if it has a collision, remove an element from that collision.
    # But the collision must be in the FULL set.
    
    # Let's try to speed up the dict approach.
    # Use a list of sums, sort, find duplicate value V.
    # Then, iterate through masks again, but only calculate sum if it *could* be V?
    # No, we don't know which mask gives V.
    
    # Let's just do the full dict scan.
    # To make it faster, we can use `itertools` or a simple loop.
    # 4M iterations in Python is ~0.5s.
    # We can afford 50 such scans in 60s.
    
    sum_to_mask = {0: 0}
    
    # We need to generate sums and masks.
    # We can do this iteratively.
    # But storing 4M entries in a dict is slow.
    
    # Let's try a different tactic for finding the conflict:
    # 1. Get sorted sums.
    # 2. Find a value V that appears at least twice.
    # 3. We need to find two masks m1, m2 such that sum(m1) = sum(m2) = V.
    # 4. We can iterate through masks, calculate sum, and if sum == V, store the mask.
    # 5. We stop when we have two masks.
    
    # Step 2: Find V.
    # We have the sorted `sums` array from the NumPy check.
    # We can find the first duplicate in the sorted array.
    diffs = np.diff(sums)
    # Find index where diff == 0
    # There might be multiple. Pick the first one.
    dup_indices = np.where(diffs == 0)[0]
    if len(dup_indices) == 0:
        return True, None # Should not happen if we checked before
        
    dup_idx = dup_indices[0]
    V = sums[dup_idx]
    
    # Step 3 & 4: Find masks for V.
    # Iterate through all masks, calculate sum, compare to V.
    # This is still 4M iterations.
    # But we can stop as soon as we find 2 masks.
    
    n = len(indices)
    masks_found = []
    
    # We can optimize sum calculation using DP or Gray code?
    # For simplicity, let's just iterate.
    # 4M iterations * 22 ops = 88M ops. ~1s in Python.
    # This might be too slow if we do it many times.
    
    # Optimization:
    # We can precompute sums for all masks using a list?
    # `sums_arr` is already computed.
    # But we need to map sum -> mask.
    # The `sums` array from NumPy is just the values, not the masks.
    # The order in `sums` corresponds to the order of generation.
    # If we generated `sums` by `sums = np.concatenate((sums, sums + v))`,
    # the order is:
    # 0
    # v0, v0 (from 0+v0) -> No, `sums` starts with [0].
    # Iter 1: [0, v0]
    # Iter 2: [0, v0, v1, v0+v1]
    # Iter 3: [0, v0, v1, v0+v1, v2, v0+v2, v1+v2, v0+v1+v2]
    # The index in the array corresponds to the mask!
    # Index 0: mask 0
    # Index 1: mask 1 (bit 0)
    # Index 2: mask 2 (bit 1)
    # Index 3: mask 3 (bit 0, 1)
    # ...
    # Index k: mask k.
    
    # So, if we have the sorted `sums` array, we lose the mask information.
    # BUT, we can create an array of masks `masks_arr = np.arange(2**n)`.
    # And we can sort `masks_arr` based on `sums`?
    # `order = np.argsort(sums)`
    # `sorted_sums = sums[order]`
    # `sorted_masks = masks_arr[order]`
    
    # Then we can find duplicates in `sorted_sums` and get the corresponding `sorted_masks`.
    
    # This is efficient!
    
    # Let's implement this.
    
    n = len(indices)
    # Generate sums in the specific order that matches mask index
    # The iterative concat method produces sums in the order of mask index?
    # Let's verify.
    # Start: [0] (mask 0)
    # Add v0: [0, v0] (masks 0, 1)
    # Add v1: [0, v0, v1, v0+v1] (masks 0, 1, 2, 3)
    # Yes! The i-th element in the list corresponds to mask i.
    
    # So:
    sums = np.array([0], dtype=np.int64)
    for v in [_WEIGHTS[i-1] for i in indices]:
        sums = np.concatenate((sums, sums + v))
        
    # Now sums[i] is the sum for mask i.
    
    # Sort
    order = np.argsort(sums)
    sorted_sums = sums[order]
    sorted_masks = order # The indices in the original array are the masks
    
    # Find duplicate
    diffs = np.diff(sorted_sums)
    dup_indices = np.where(diffs == 0)[0]
    
    if len(dup_indices) == 0:
        return True, None
        
    # Take the first duplicate
    idx = dup_indices[0]
    m1 = sorted_masks[idx]
    m2 = sorted_masks[idx+1]
    
    # m1 and m2 are the masks (in the local index space of `indices`)
    # We need to map them back to the original indices (1..22)
    
    # The bits in m1/m2 correspond to the positions in `indices`
    # indices[0] is the first element added, etc.
    
    conflict_local_indices = []
    # Symmetric difference of masks
    diff_mask = m1 ^ m2
    
    for i in range(n):
        if (diff_mask >> i) & 1:
            conflict_local_indices.append(indices[i])
            
    return False, set(conflict_local_indices)

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    all_elements = list(range(1, 23))
    best = []
    best_len = 0
    
    # Strategies for removal
    # 0: min weight
    # 1: max weight
    # 2: random
    # 3: max index
    
    strategies = [0, 1, 2, 3]
    
    for strat in strategies:
        for trial in range(10): # 10 trials per strategy
            if time.time() > deadline:
                break
                
            S = set(all_elements)
            
            while time.time() < deadline:
                is_diss, conflict = _find_collision(list(S))
                
                if is_diss:
                    if len(S) > best_len:
                        best = list(S)
                        best_len = len(S)
                    break # Found a valid set for this trial
                    
                if conflict is None:
                    break # Error?
                    
                # Choose element to remove
                if strat == 0:
                    # Min weight
                    x = min(conflict, key=lambda i: _WEIGHTS[i-1])
                elif strat == 1:
                    # Max weight
                    x = max(conflict, key=lambda i: _WEIGHTS[i-1])
                elif strat == 2:
                    # Random
                    x = random.choice(list(conflict))
                else:
                    # Max index
                    x = max(conflict)
                    
                S.remove(x)
                
            # If we exited due to deadline, check if current S is valid
            if time.time() >= deadline:
                if len(S) > 0:
                    is_diss, _ = _find_collision(list(S))
                    if is_diss and len(S) > best_len:
                        best = list(S)
                        best_len = len(S)
                        
    return best
