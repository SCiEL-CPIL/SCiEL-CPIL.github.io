import random
import time
import sys

N = 512

# Precompute all 4-term arithmetic progressions
APs = []
for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        APs.append(ap)

# Build conflict graph using bitmasks for each element
# conflict_mask[i] is a bitmask where bit j is set if elements i+1 and j+1 are in some 4-AP together
# We use 0-indexed internally: element x corresponds to index x-1
conflict_mask = [0] * N
# For each AP, all pairs of elements in that AP conflict
for ap in APs:
    a, b, c, d_elem = ap
    # Convert to 0-indexed
    a0, b0, c0, d0 = a-1, b-1, c-1, d_elem-1
    # All 6 pairs conflict
    pairs = [(a0,b0), (a0,c0), (a0,d0), (b0,c0), (b0,d0), (c0,d0)]
    for i, j in pairs:
        conflict_mask[i] |= (1 << j)
        conflict_mask[j] |= (1 << i)

# Precompute degree for each element
degree = [bin(conflict_mask[i]).count('1') for i in range(N)]

def is_valid_bitmask(mask):
    """Check if the set represented by mask contains no 4-AP."""
    for ap in APs:
        a, b, c, d_elem = ap
        a0, b0, c0, d0 = a-1, b-1, c-1, d_elem-1
        if (mask >> a0) & 1 and (mask >> b0) & 1 and (mask >> c0) & 1 and (mask >> d0) & 1:
            return False
    return True

def can_add_to_mask(mask, idx):
    """Check if adding element idx (0-indexed) to mask keeps it 4-AP free."""
    # Check all APs containing this element
    for ap in APs:
        if (ap[0]-1 == idx or ap[1]-1 == idx or ap[2]-1 == idx or ap[3]-1 == idx):
            a0, b0, c0, d0 = ap[0]-1, ap[1]-1, ap[2]-1, ap[3]-1
            # Count how many of the other 3 are in mask
            count = 0
            for other in [a0, b0, c0, d0]:
                if other != idx and (mask >> other) & 1:
                    count += 1
            if count == 3:
                return False
    return True

def greedy_max_independent_set(deadline, initial_mask=0):
    """
    Greedy approach to find a large independent set (4-AP-free set).
    Uses bitmask representation for fast conflict checking.
    """
    # Start with a random greedy initialization
    order = list(range(N))
    random.shuffle(order)
    
    mask = initial_mask
    
    for idx in order:
        if (mask >> idx) & 1:
            continue
        if can_add_to_mask(mask, idx):
            mask |= (1 << idx)
    
    return mask

def local_search_improve(mask, deadline):
    """
    Try to improve the independent set by swaps.
    Returns improved mask.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find all elements not in the set
        not_in_set = [i for i in range(N) if not (mask >> i) & 1]
        if not not_in_set:
            break
        
        random.shuffle(not_in_set)
        
        # Try to add elements
        added = False
        for x in not_in_set[:30]:
            if can_add_to_mask(mask, x):
                mask |= (1 << x)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try swaps: replace one element with another
        in_set = [i for i in range(N) if (mask >> i) & 1]
        random.shuffle(not_in_set)
        random.shuffle(in_set)
        
        for x in not_in_set[:20]:
            # Find conflicts: which elements in the set would conflict with x
            conflicting_in_set = [y for y in in_set if (conflict_mask[x] >> y) & 1]
            if not conflicting_in_set:
                continue
            
            # Try removing one conflicting element and adding x
            random.shuffle(conflicting_in_set)
            for y in conflicting_in_set[:5]:
                # Check if mask without y plus x is valid
                new_mask = (mask & ~(1 << y)) | (1 << x)
                # Quick check: just verify no new 4-APs are formed
                valid = True
                for ap in APs:
                    a0, b0, c0, d0 = ap[0]-1, ap[1]-1, ap[2]-1, ap[3]-1
                    if (new_mask >> a0) & 1 and (new_mask >> b0) & 1 and (new_mask >> c0) & 1 and (new_mask >> d0) & 1:
                        valid = False
                        break
                if valid:
                    mask = new_mask
                    improved = True
                    break
            if improved:
                break
    
    return mask

def branch_and_bound_4ap_free(deadline):
    """
    Branch and bound to find a maximum 4-AP-free set.
    Uses bitmask representation and pruning.
    """
    best_size = 0
    best_mask = 0
    
    # Upper bound: for any set of available elements, the max independent set
    # size is bounded by the number of available elements
    # We use a simple branch and bound with ordering by degree
    
    def bb(available_mask, current_mask, current_size):
        nonlocal best_size, best_mask, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning: if current_size + popcount(available) <= best_size, prune
        avail_count = bin(available_mask).count('1')
        if current_size + avail_count <= best_size:
            return
        
        # If no more elements available
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Choose an element to branch on
        # Pick element with highest degree among available for better pruning
        avail_list = [i for i in range(N) if (available_mask >> i) & 1]
        if not avail_list:
            return
        
        # Sort by degree descending
        avail_list.sort(key=lambda i: degree[i], reverse=True)
        pivot = avail_list[0]
        
        # Branch 1: Include pivot
        if can_add_to_mask(current_mask, pivot):
            new_current = current_mask | (1 << pivot)
            # Remove pivot and all its neighbors from available
            new_available = available_mask & ~(1 << pivot) & ~conflict_mask[pivot]
            bb(new_available, new_current, current_size + 1)
        
        # Branch 2: Exclude pivot
        new_available2 = available_mask & ~(1 << pivot)
        bb(new_available2, current_mask, current_size)
    
    bb((1 << N) - 1, 0, 0)
    
    return best_mask, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_size = 0
    best_mask = 0
    
    # Phase 1: Multiple random greedy initializations
    while time.time() < deadline - 10.0:
        mask = greedy_max_independent_set(deadline)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
        
        # Local search improvement
        mask = local_search_improve(mask, deadline - 5.0)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Try branch and bound on a restricted problem
    if time.time() < deadline - 5.0:
        # Use the best found so far as a lower bound
        # Try B&B on a subset of elements
        # Pick elements that are in the current best or have low degree
        in_best = [i for i in range(N) if (best_mask >> i) & 1]
        not_in_best = [i for i in range(N) if not (best_mask >> i) & 1]
        
        # Combine some from each
        candidates = in_best + not_in_best[:20]
        random.shuffle(candidates)
        
        # Build available mask from candidates
        avail_mask = 0
        for idx in candidates:
            avail_mask |= (1 << idx)
        
        # Run B&B with the current best as initial bound
        old_best_size = best_size
        bb_result_mask, bb_result_size = branch_and_bound_4ap_free(deadline - 2.0)
        
        if bb_result_size > best_size:
            best_size = bb_result_size
            best_mask = bb_result_mask
    
    # Convert best_mask to list
    result = [i+1 for i in range(N) if (best_mask >> i) & 1]
    
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S_set = set(res)
    valid = True
    for ap in APs:
        if all(x in S_set for x in ap):
            valid = False
            break
    print(f"Valid: {valid}")
