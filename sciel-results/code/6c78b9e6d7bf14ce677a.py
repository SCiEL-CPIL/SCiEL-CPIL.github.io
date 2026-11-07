"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def _count_comparable_pairs_masks(masks):
    """Count unordered pairs where one mask is a strict subset of the other."""
    m = len(masks)
    cnt = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                cnt += 1
    return cnt

def _initial_solution_greedy(n=12, m=40, time_limit=5.0):
    """
    Greedy construction: start with empty and full set, then add the set
    that maximizes new comparable pairs.
    """
    start = time.time()
    deadline = start + time_limit
    N = n
    NUM_SUBSETS = 1 << N
    all_masks = list(range(NUM_SUBSETS))
    
    F_masks = []
    used = [False] * NUM_SUBSETS
    
    # Start with empty set (0) and full set (NUM_SUBSETS - 1)
    if m >= 1:
        F_masks.append(0)
        used[0] = True
    if m >= 2:
        F_masks.append(NUM_SUBSETS - 1)
        used[NUM_SUBSETS - 1] = True
    
    # Greedily add remaining sets
    for _ in range(m - len(F_masks)):
        if time.time() > deadline:
            break
        
        best_score = -1
        best_mask = -1
        
        for msk in all_masks:
            if used[msk]:
                continue
            
            score = 0
            for f in F_masks:
                if (msk & f) == msk:  # msk subset f
                    score += 1
                elif (msk & f) == f:  # f subset msk
                    score += 1
            
            if score > best_score:
                best_score = score
                best_mask = msk
        
        if best_mask == -1:
            break
            
        F_masks.append(best_mask)
        used[best_mask] = True
        
        if time.time() > deadline:
            break
    
    # Pad if needed (should not happen with 5s limit for 40 items)
    while len(F_masks) < m:
        for msk in all_masks:
            if not used[msk]:
                F_masks.append(msk)
                used[msk] = True
                break
    
    return F_masks[:m]

def _local_search_bitset(masks, time_limit=80.0):
    """
    Local search with O(1) delta calculation using precomputed bitsets.
    masks: list of 40 integers (bitmasks)
    """
    start = time.time()
    deadline = start + time_limit
    n = 12
    m = 40
    NUM_SUBSETS = 1 << n
    
    current = masks[:]
    current_score = _count_comparable_pairs_masks(current)
    
    # Precompute for each mask:
    # - list of masks that are subsets of it
    # - list of masks that are supersets of it
    # But we only care about the 40 masks in current + all possible new masks.
    
    # Actually, for delta calculation:
    # delta = new_comparable - old_comparable
    # old_comparable = count of i != idx such that current[idx] < current[i] or current[i] < current[idx]
    # new_comparable = count of i != idx such that new_mask < current[i] or current[i] < new_mask
    
    # To make this fast, we can precompute for the current set of masks:
    # For each position i, the "contribution" of current[i] to the total score.
    # But since pairs are counted once, we can just compute the delta directly.
    
    # Optimization: Precompute subset/superset relationships for all 4096 masks?
    # That's 4096 * 4096 = 16M booleans, which is ~2MB. Feasible.
    # But checking (a & b) == a is O(1) in Python (big int ops are fast for 12 bits).
    
    # Let's use a different strategy:
    # Maintain a set of current masks for O(1) duplicate check.
    current_set = set(current)
    
    # Precompute all possible masks
    all_masks = list(range(NUM_SUBSETS))
    
    # To speed up the inner loop, we can precompute for each mask in current,
    # the number of other masks in current that are comparable to it.
    # But this changes when we swap.
    
    # Better: Just compute delta on the fly, but efficiently.
    # delta = (comparables of new with others) - (comparables of old with others)
    
    improved = True
    # We will do multiple passes until no improvement or time limit.
    
    # To avoid getting stuck, we can randomize the order of indices or masks.
    
    while improved and (time.time() - start) < deadline:
        improved = False
        
        # Randomize order of indices to escape local optima
        indices = list(range(m))
        random.shuffle(indices)
        
        for idx in indices:
            if (time.time() - start) > deadline:
                break
            
            old_mask = current[idx]
            
            # Try replacing with each other mask
            # We can iterate all_masks and skip if in current_set
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                if new_mask in current_set:
                    continue
                
                # Calculate delta
                # old_comparable: how many other masks are comparable to old_mask
                old_comp = 0
                for i in range(m):
                    if i == idx:
                        continue
                    other = current[i]
                    if (old_mask & other) == old_mask or (old_mask & other) == other:
                        old_comp += 1
                
                # new_comparable: how many other masks are comparable to new_mask
                new_comp = 0
                for i in range(m):
                    if i == idx:
                        continue
                    other = current[i]
                    if (new_mask & other) == new_mask or (new_mask & other) == other:
                        new_comp += 1
                
                delta = new_comp - old_comp
                
                if delta > 0:
                    # Accept swap
                    current_set.remove(old_mask)
                    current[idx] = new_mask
                    current_set.add(new_mask)
                    current_score += delta
                    improved = True
                    # Move to next index? Or restart?
                    # Restarting from beginning might be better for convergence,
                    # but moving on is faster. Let's move on for speed.
                    break
    
    return current, current_score

def construct():
    start = time.time()
    total_deadline = 85.0  # Safety margin before 90s kill
    
    # Step 1: Greedy initial solution
    t0 = time.time()
    init_masks = _initial_solution_greedy(n=12, m=40, time_limit=5.0)
    t1 = time.time()
    
    # Step 2: Local search
    remaining_time = total_deadline - (time.time() - start) - 1.0  # 1s buffer for conversion
    if remaining_time > 1.0:
        final_masks, final_score = _local_search_bitset(init_masks, time_limit=remaining_time)
    else:
        final_masks = init_masks
        final_score = _count_comparable_pairs_masks(final_masks)
    
    # Convert masks to lists of ints
    result = []
    for msk in final_masks:
        s = []
        for i in range(12):
            if msk & (1 << i):
                s.append(i)
        result.append(s)
    
    return result

# EVOLVE-BLOCK-END
