import time
import random
import numpy as np

N = 256
DEADLINE = 55.0

# Precompute 3-APs and 4-APs
THREE_APS = []
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        c = b + d
        if c > N:
            break
        THREE_APS.append((a - 1, b - 1, c - 1))

FOUR_APS = []
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        c = b + d
        e = c + d
        if e > N:
            break
        FOUR_APS.append((a - 1, b - 1, c - 1, e - 1))

# Precompute adjacency lists for fast delta calculation
# elem_to_3ap[i] = list of indices of 3-APs containing element i
# elem_to_4ap[i] = list of indices of 4-APs containing element i
elem_to_3ap = [[] for _ in range(N)]
for idx, (a, b, c) in enumerate(THREE_APS):
    elem_to_3ap[a].append(idx)
    elem_to_3ap[b].append(idx)
    elem_to_3ap[c].append(idx)

elem_to_4ap = [[] for _ in range(N)]
for idx, (a, b, c, d) in enumerate(FOUR_APS):
    elem_to_4ap[a].append(idx)
    elem_to_4ap[b].append(idx)
    elem_to_4ap[c].append(idx)
    elem_to_4ap[d].append(idx)

def _bit_set(mask, pos):
    return (mask[pos >> 6] >> (pos & 63)) & 1

def count_3aps(mask):
    cnt = 0
    for a, b, c in THREE_APS:
        if _bit_set(mask, a) and _bit_set(mask, b) and _bit_set(mask, c):
            cnt += 1
    return cnt

def count_4aps(mask):
    cnt = 0
    for a, b, c, d in FOUR_APS:
        if _bit_set(mask, a) and _bit_set(mask, b) and _bit_set(mask, c) and _bit_set(mask, d):
            cnt += 1
    return cnt

def get_delta_3(mask, pos, is_adding):
    """
    Calculate change in 3-AP count if we toggle 'pos'.
    Returns the delta.
    """
    delta = 0
    for ap_idx in elem_to_3ap[pos]:
        a, b, c = THREE_APS[ap_idx]
        # Check if the other two elements are in the mask
        others = [x for x in (a, b, c) if x != pos]
        if _bit_set(mask, others[0]) and _bit_set(mask, others[1]):
            if is_adding:
                delta += 1
            else:
                delta -= 1
    return delta

def is_4ap_conflict(mask, pos, is_adding):
    """
    Check if adding 'pos' creates a 4-AP.
    If is_adding is False, it always returns False (removing cannot create).
    """
    if not is_adding:
        return False
        
    for ap_idx in elem_to_4ap[pos]:
        a, b, c, d = FOUR_APS[ap_idx]
        others = [x for x in (a, b, c, d) if x != pos]
        if _bit_set(mask, others[0]) and _bit_set(mask, others[1]) and _bit_set(mask, others[2]):
            return True
    return False

def greedy_init():
    """
    Greedy initialization: Add elements that maximize 3-AP gain without creating 4-APs.
    """
    mask = [0, 0, 0, 0]
    # Start with a random permutation to break symmetry
    order = list(range(N))
    random.shuffle(order)
    
    # We can try multiple passes or just one strong pass.
    # Let's do a pass where we pick the best available element at each step? 
    # That's O(N^2 * cost). N=256, N^2=65k. Cost is small.
    # But simple random order with acceptance is faster and often good enough for ILS.
    
    for pos in order:
        if is_4ap_conflict(mask, pos, True):
            continue
        
        delta = get_delta_3(mask, pos, True)
        # Accept if it adds 3-APs or randomly with small prob to ensure size
        if delta > 0 or (delta == 0 and random.random() < 0.1):
             mask[pos >> 6] |= (1 << (pos & 63))
             
    return mask

def local_search(mask, time_limit):
    """
    Hill climbing: Accept any move that increases 3-APs and maintains 4-AP-freeness.
    """
    start_time = time.time()
    cur_n3 = count_3aps(mask)
    improved = True
    
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Randomize the order of checking moves to avoid cycles
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for pos in candidates:
            is_set = _bit_set(mask, pos)
            
            if is_set:
                # Try removing
                delta = get_delta_3(mask, pos, False)
                if delta > 0:
                    # Removing increases 3-APs? No, delta is negative if we remove a contributor.
                    # Wait. get_delta_3(mask, pos, False) returns -1 for each 3-AP broken.
                    # So delta is negative. We want delta > 0 for improvement.
                    # So removing only helps if it breaks 4-APs? But we are in 4-AP free set.
                    # Removing an element never creates a 4-AP.
                    # Does removing an element ever INCREASE 3-APs? No.
                    # So we should never remove in pure maximization unless we are stuck and want to perturb.
                    # In local search, we only add.
                    pass
            else:
                # Try adding
                if is_4ap_conflict(mask, pos, True):
                    continue
                
                delta = get_delta_3(mask, pos, True)
                if delta > 0:
                    # Accept
                    mask[pos >> 6] |= (1 << (pos & 63))
                    cur_n3 += delta
                    improved = True
                    # Restart shuffle? Or continue? 
                    # Restarting ensures we always take the best available? 
                    # No, we just found an improvement. Let's continue checking others in this pass.
                    
        # If no improvement in a full pass, we are at a local optimum.
        
    return mask, cur_n3

def iterated_local_search(time_budget):
    best_mask = None
    best_n3 = 0
    
    start_time = time.time()
    
    # Initial greedy
    mask = greedy_init()
    mask, cur_n3 = local_search(mask, 1.0)
    
    if cur_n3 > best_n3:
        best_n3 = cur_n3
        best_mask = list(mask)
    
    while time.time() - start_time < time_budget - 5.0:
        # Perturb
        # Remove k random elements from current best
        k = random.randint(2, 6)
        mask = list(best_mask)
        
        # Get list of set bits
        set_bits = []
        for i in range(N):
            if _bit_set(mask, i):
                set_bits.append(i)
        
        if len(set_bits) < k:
            break
            
        to_remove = random.sample(set_bits, k)
        for pos in to_remove:
            mask[pos >> 6] &= ~(1 << (pos & 63))
            
        # Local Search
        # Give it a small time slice, e.g., 0.5s or until convergence
        mask, cur_n3 = local_search(mask, 0.5)
        
        if cur_n3 > best_n3:
            best_n3 = cur_n3
            best_mask = list(mask)
            
    return best_mask, best_n3

def construct():
    start_time = time.time()
    time_budget = DEADLINE - (time.time() - start_time)
    
    # Run ILS
    best_mask, best_n3 = iterated_local_search(time_budget)
    
    # Final verification and cleanup
    # Ensure no 4-APs (should be true by construction, but safety check)
    n4 = count_4aps(best_mask)
    if n4 > 0:
        # Fallback to safe construction if something went wrong
        # Simple base-3 construction
        mask = [0, 0, 0, 0]
        for x in range(1, 257):
            y = x - 1
            valid = True
            while y > 0:
                if y % 3 == 2:
                    valid = False
                    break
                y //= 3
            if valid:
                pos = x - 1
                mask[pos >> 6] |= (1 << (pos & 63))
        best_mask = mask
    
    # Convert to list
    result = []
    for pos in range(N):
        if _bit_set(best_mask, pos):
            result.append(pos + 1)
            
    return sorted(result)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    print(f"First 10: {res[:10]}")
