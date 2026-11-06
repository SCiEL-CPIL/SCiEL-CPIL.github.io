"""Program for reciprocal weight of three-progression-free sets on [1,256]."""
import time
import random
import math

N = 256
RECIPS = [0.0] + [1.0 / i for i in range(1, N + 1)]

def _is_3ap_free_add(mask, x):
    """
    Check if adding x to set represented by mask keeps it 3-AP free.
    x is 1-indexed. mask is int where bit i-1 is set if i is in set.
    Returns True if it is safe to add x.
    """
    if (mask >> (x - 1)) & 1:
        return True
    
    m = mask
    while m:
        lsb = m & -m
        a = lsb.bit_length()  # 1-indexed
        m ^= lsb
        
        # Case 1: x is the middle term. a, x, c
        # 2x = a + c  => c = 2x - a
        c = 2 * x - a
        if c > 0 and c <= N and c != a and c != x:
            if (mask >> (c - 1)) & 1:
                return False
        
        # Case 2: x is the end term. a, b, x
        # a + x = 2b => b = (a + x) / 2
        if (x + a) % 2 == 0:
            b = (x + a) // 2
            if b > 0 and b <= N and b != a and b != x:
                if (mask >> (b - 1)) & 1:
                    return False
                    
    return True

def _get_sum(mask):
    s = 0.0
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        s += RECIPS[i]
        m ^= lsb
    return s

def _get_elements(mask):
    """Return list of elements in mask."""
    result = []
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        result.append(i)
        m ^= lsb
    return result

def _greedy_init():
    """
    Greedy construction: add elements in order of descending weight (ascending index)
    if they don't create a 3-AP.
    """
    mask = 0
    # We want to maximize sum of 1/i, so we prefer smaller i.
    # Sort by i ascending.
    for x in range(1, N + 1):
        if _is_3ap_free_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def _local_search(mask, deadline):
    """
    Perform local search:
    1. Try adding any element if valid (improves score).
    2. Try swapping an element x in S with y not in S if valid and 1/y > 1/x.
    3. If stuck, perform a random kick (remove random elements) to escape local optimum.
    """
    best_mask = mask
    best_score = _get_sum(mask)
    
    elements = _get_elements(best_mask)
    element_set = set(elements)
    
    # Precompute candidates for swaps: sorted by weight descending (index ascending)
    # We want to replace large x (low weight) with small y (high weight)
    
    kick_count = 0
    no_improvement_count = 0
    
    while time.time() < deadline:
        improved = False
        current_elements = _get_elements(best_mask)
        current_set = set(current_elements)
        current_score = _get_sum(best_mask)
        
        # Phase 1: Try adding any element
        # We can just iterate through all 1..N
        for y in range(1, N + 1):
            if y in current_set:
                continue
            if _is_3ap_free_add(best_mask, y):
                new_mask = best_mask | (1 << (y - 1))
                new_score = current_score + RECIPS[y]
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    improved = True
        if improved:
            continue
        
        # Phase 2: Try swaps
        # For each x in current set, try replacing with y not in set
        # To be efficient, sort potential y's by weight (ascending index)
        # and stop early if possible? No, just check all.
        
        current_elements = _get_elements(best_mask)
        current_set = set(current_elements)
        
        # Optimization: Only consider swapping x if there exists a y < x not in set that is valid
        # Or just iterate all pairs? 100 * 150 = 15000 checks. Each check is O(100). 1.5M ops. Fine.
        
        # Sort current elements by index ascending (so we check removing small ones first? 
        # No, we want to remove LARGE indices (small weight) to replace with SMALL indices (large weight).
        # So sort current_elements descending by index.
        current_elements_desc = sorted(current_elements, reverse=True)
        
        for x in current_elements_desc:
            # Try replacing x with y
            # y should be < x to improve score (since 1/y > 1/x)
            # If y > x, score decreases. So only y < x is useful for improvement.
            # Wait, is it always better to swap large x for small y?
            # Yes, 1/y > 1/x if y < x.
            # So we only need to check y < x.
            
            # Iterate y from 1 to x-1
            for y in range(1, x):
                if y in current_set:
                    continue
                # Check if y can be added if x is removed
                mask_without_x = best_mask ^ (1 << (x - 1))
                if _is_3ap_free_add(mask_without_x, y):
                    new_mask = mask_without_x | (1 << (y - 1))
                    # Score improvement is guaranteed if y < x
                    # new_score = current_score - RECIPS[x] + RECIPS[y]
                    # Since y < x, 1/y > 1/x, so new_score > current_score
                    best_mask = new_mask
                    best_score += RECIPS[y] - RECIPS[x]
                    current_elements = _get_elements(best_mask)
                    current_set = set(current_elements)
                    improved = True
                    break # Restart inner loop? Or just break and update set?
                    # Breaking inner loop and restarting outer loop is safer for consistency
            if improved:
                break
        
        if improved:
            continue
            
        # Phase 3: Stuck. Perform a kick.
        no_improvement_count += 1
        kick_count += 1
        if kick_count > 20:
            break
            
        # Random kick: remove 1 to 3 random elements
        num_removals = random.randint(1, 3)
        kick_elements = random.sample(current_elements, min(num_removals, len(current_elements)))
        for x in kick_elements:
            best_mask ^= (1 << (x - 1))
            
        # Update best if score went up? No, kick lowers score.
        # We just reset the local search state to this new mask
        current_elements = _get_elements(best_mask)
        current_set = set(current_elements)

    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Greedy Initialization
    best_mask = _greedy_init()
    best_score = _get_sum(best_mask)
    
    # 2. Local Search
    best_mask, best_score = _local_search(best_mask, deadline)
    
    # 3. Convert to list
    result = _get_elements(best_mask)
    return sorted(result)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import sys
    t0 = time.time()
    res = construct()
    t1 = time.time()
    
    # Verification
    def has_3ap(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a == b: continue
                c = 2*b - a
                if c in S_set and c != a and c != b:
                    return (a, b, c)
        return None
        
    violation = has_3ap(res)
    if violation:
        print(f"ERROR: 3-AP found: {violation}")
    else:
        print(f"OK: No 3-AP found.")
        
    print(f"Time: {t1-t0:.2f}s")
    print(f"Set size: {len(res)}")
    score = sum(1.0/x for x in res)
    print(f"Sum: {score:.6f}")
    print(f"First 20 elements: {res[:20]}")
