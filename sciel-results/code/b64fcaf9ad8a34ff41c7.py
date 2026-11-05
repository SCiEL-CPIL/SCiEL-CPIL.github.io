import time
import random
import sys

N = 256

# Precomputation
def _precompute():
    four_aps = []
    four_ap_masks = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            four_aps.append(ap)
            mask = 0
            for x in ap:
                mask |= (1 << (x - 1))
            four_ap_masks.append(mask)
    
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            ap = (a, a + d, a + 2 * d)
            three_aps.append(ap)
            
    # Map element to list of 3-AP indices
    elem_to_3ap_idx = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3ap_idx[x].append(idx)
            
    # Map element to list of 4-AP indices (for repair/validity checks if needed, 
    # but we use masks for validity)
    
    return four_aps, four_ap_masks, three_aps, elem_to_3ap_idx

(_FOUR_APS, _FOUR_AP_MASKS, _THREE_APS, _ELEM_TO_3AP_IDX) = _precompute()
_NUM_4APS = len(_FOUR_AP_MASKS)
_NUM_3APS = len(_THREE_APS)

def _is_valid(A):
    """Check if A contains no 4-AP."""
    for mask in _FOUR_AP_MASKS:
        if (A & mask) == mask:
            return False
    return True

def _count_3aps(A):
    """Count 3-APs in A."""
    count = 0
    for ap in _THREE_APS:
        a, b, c = ap
        if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1) and ((A >> (c-1)) & 1):
            count += 1
    return count

def _delta_add(x, A):
    """Calculate increase in 3-APs if x is added to A."""
    delta = 0
    for ap_idx in _ELEM_TO_3AP_IDX[x]:
        a, b, c = _THREE_APS[ap_idx]
        # Check if the other two are in A
        # We know x is one of a,b,c.
        # Construct mask for the other two
        # Faster: check bits directly
        if a == x:
            if ((A >> (b-1)) & 1) and ((A >> (c-1)) & 1):
                delta += 1
        elif b == x:
            if ((A >> (a-1)) & 1) and ((A >> (c-1)) & 1):
                delta += 1
        elif c == x:
            if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
                delta += 1
    return delta

def _delta_remove(x, A):
    """Calculate decrease in 3-APs if x is removed from A."""
    # Same logic as add, but we are removing x.
    # The 3-APs lost are those where x is present and the other two are present.
    return _delta_add(x, A)

def _creates_4ap(x, A):
    """Check if adding x to A creates a 4-AP."""
    # We need to check all 4-APs containing x.
    # To make this fast, we should precompute which 4-APs contain x.
    # But we didn't do that in precompute. Let's do it on the fly or precompute.
    # Precomputing is better.
    pass # Placeholder, will use mask check in local search directly if possible?
    # Actually, checking all 4-APs for validity is O(N_4AP). 
    # N_4AP is roughly sum(N-3d) ~ N^2/6 ~ 10k.
    # 10k checks per move is slow.
    # We need elem_to_4ap_idx.

# Re-precompute with elem_to_4ap_idx
def _precompute_full():
    four_aps = []
    four_ap_masks = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            four_aps.append(ap)
            mask = 0
            for x in ap:
                mask |= (1 << (x - 1))
            four_ap_masks.append(mask)
            
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            ap = (a, a + d, a + 2 * d)
            three_aps.append(ap)
            
    elem_to_3ap_idx = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3ap_idx[x].append(idx)
            
    elem_to_4ap_idx = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4ap_idx[x].append(idx)
            
    return four_aps, four_ap_masks, three_aps, elem_to_3ap_idx, elem_to_4ap_idx

(_FOUR_APS, _FOUR_AP_MASKS, _THREE_APS, _ELEM_TO_3AP_IDX, _ELEM_TO_4AP_IDX) = _precompute_full()
_NUM_4APS = len(_FOUR_AP_MASKS)
_NUM_3APS = len(_THREE_APS)

def _check_add_valid(x, A):
    """Check if adding x to A is valid (no 4-AP created)."""
    for ap_idx in _ELEM_TO_4AP_IDX[x]:
        mask = _FOUR_AP_MASKS[ap_idx]
        # The other 3 elements must be in A
        # mask has 4 bits. One is x.
        # Check if (A & (mask ^ (1 << (x-1)))) == (mask ^ (1 << (x-1)))
        other_mask = mask ^ (1 << (x - 1))
        if (A & other_mask) == other_mask:
            return False
    return True

def _local_search(A, time_limit=0.5):
    """Hill climbing with 1-swaps and adds."""
    start_time = time.time()
    current_score = _count_3aps(A)
    
    # We represent A as an int bitmask.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try Adds
        for x in range(1, N + 1):
            if (A >> (x-1)) & 1:
                continue
            if not _check_add_valid(x, A):
                continue
            delta = _delta_add(x, A)
            if delta > 0:
                A |= (1 << (x-1))
                current_score += delta
                improved = True
                # Restart loop to find more adds? Or continue?
                # Restarting is safer for "best improvement" but slower.
                # "First improvement" is faster. Let's do first improvement.
                break
        
        if improved:
            continue
            
        # Try 1-Swaps
        # Remove x, Add y
        # To be fast, we can iterate x in A, y not in A.
        # This is O(N^2).
        # Optimization: Only try swaps if we are stuck.
        
        swapped = False
        for x in range(1, N + 1):
            if not ((A >> (x-1)) & 1):
                continue
            # Calculate cost of removing x
            # We need to know which 3-APs are lost.
            # This is _delta_remove(x, A)
            # But we also need to know if removing x allows adding y.
            
            for y in range(1, N + 1):
                if (A >> (y-1)) & 1:
                    continue
                if y == x:
                    continue
                
                # Check if adding y to (A \ {x}) is valid
                # This requires checking 4-APs containing y.
                # For each 4-AP containing y, check if other 3 are in A \ {x}.
                valid = True
                for ap_idx in _ELEM_TO_4AP_IDX[y]:
                    mask = _FOUR_AP_MASKS[ap_idx]
                    other_mask = mask ^ (1 << (y - 1))
                    # Check if other_mask is subset of A, but excluding x if x is in other_mask
                    # If x is in other_mask, we need (A & other_mask) == (other_mask ^ (1 << (x-1)))
                    # If x is not in other_mask, we need (A & other_mask) == other_mask
                    
                    # Simplified: 
                    # Temp A = A ^ (1 << (x-1))
                    # Check if Temp A contains other_mask
                    # But creating Temp A is expensive in inner loop.
                    
                    # Logic:
                    # other_mask has 3 bits.
                    # If x is one of them, then we need the other 2 bits to be in A.
                    # If x is not one of them, we need all 3 bits to be in A.
                    
                    if other_mask & (1 << (x-1)):
                        # x is in the 4-AP with y
                        # We need the other 2 elements to be in A
                        remaining_mask = other_mask ^ (1 << (x-1))
                        if (A & remaining_mask) != remaining_mask:
                            valid = False
                            break
                    else:
                        # x is not in the 4-AP with y
                        # We need all 3 elements to be in A
                        if (A & other_mask) != other_mask:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Calculate delta
                # Lost 3-APs from removing x
                # Gained 3-APs from adding y to (A \ {x})
                
                # Lost: 3-APs containing x where other 2 are in A.
                # Note: If a 3-AP contains both x and y, it is lost by removing x, 
                # but is it gained by adding y?
                # If 3-AP is (x, y, z). 
                # Before: x in A, y not in A. 3-AP not counted.
                # After: x not in A, y in A. 3-AP not counted.
                # So 3-APs containing both x and y do not change status (0 -> 0).
                # 3-APs containing x but not y: Lost if other 2 in A.
                # 3-APs containing y but not x: Gained if other 2 in A.
                
                delta = 0
                # Lost
                for ap_idx in _ELEM_TO_3AP_IDX[x]:
                    a, b, c = _THREE_APS[ap_idx]
                    # Check if y is in this 3-AP
                    if (a == y) or (b == y) or (c == y):
                        continue
                    # Check if other 2 are in A
                    # Since x is in A, and we are removing x.
                    # The other 2 are in A if their bits are set.
                    # We don't need to check x because x is the one being removed.
                    # Wait, the 3-AP is (x, u, v). u,v are the others.
                    # We need u,v in A.
                    # Since we are swapping, we check against current A.
                    # If u,v in A, then this 3-AP is currently active. Removing x deactivates it.
                    mask_ap = 0
                    for z in (a, b, c):
                        if z != x:
                            mask_ap |= (1 << (z-1))
                    if (A & mask_ap) == mask_ap:
                        delta -= 1

                # Gained
                for ap_idx in _ELEM_TO_3AP_IDX[y]:
                    a, b, c = _THREE_APS[ap_idx]
                    # Check if x is in this 3-AP
                    if (a == x) or (b == x) or (c == x):
                        continue
                    # Check if other 2 are in A (since x is removed, they must be in A, not just A\{x})
                    # If other 2 are in A, and y is added, 3-AP becomes active.
                    mask_ap = 0
                    for z in (a, b, c):
                        if z != y:
                            mask_ap |= (1 << (z-1))
                    if (A & mask_ap) == mask_ap:
                        delta += 1

                if delta > 0:
                    # Perform swap
                    A ^= (1 << (x-1))
                    A |= (1 << (y-1))
                    current_score += delta
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                break
        
        if not improved:
            break
            
    return A, current_score

def _greedy_init(rng):
    """Greedy construction."""
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if _check_add_valid(x, A):
            A |= (1 << (x-1))
    return A

def _repair(A):
    """Remove elements to make A valid (no 4-APs)."""
    while True:
        # Find a 4-AP
        found_ap_idx = -1
        for i, mask in enumerate(_FOUR_AP_MASKS):
            if (A & mask) == mask:
                found_ap_idx = i
                break
        
        if found_ap_idx == -1:
            return A
        
        # Remove one element from this 4-AP
        # Choose the one with lowest contribution to 3-APs
        ap = _FOUR_APS[found_ap_idx]
        best_x = ap[0]
        min_contrib = float('inf')
        
        for x in ap:
            # Calculate contribution of x to 3-APs in current A
            # This is _delta_remove(x, A)
            # But we just need to pick one. 
            # Approximation: pick random or first.
            # Better: pick the one that breaks the most 3-APs? No, we want to keep 3-APs.
            # So remove the one that contributes LEAST to 3-APs.
            contrib = _delta_remove(x, A)
            if contrib < min_contrib:
                min_contrib = contrib
                best_x = x
        
        A ^= (1 << (best_x - 1))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Phase 1: Initial Population
    pop_size = 50
    population = []
    scores = []
    
    rng = random.Random(42)
    
    for _ in range(pop_size):
        A = _greedy_init(rng)
        A = _repair(A)
        # Local search
        A, score = _local_search(A, time_limit=1.0)
        population.append(A)
        scores.append(score)
        
    best_A = population[0]
    best_score = scores[0]
    if scores[0] > best_score:
        best_score = scores[0]
        
    # Sort population by score
    # Keep best
    
    # Phase 2: Genetic Algorithm
    # Crossover: Uniform
    # Mutation: Bit flip
    
    while time.time() < deadline - 5.0:
        # Selection: Tournament
        p1_idx = rng.randrange(pop_size)
        p2_idx = rng.randrange(pop_size)
        if scores[p1_idx] > scores[p2_idx]:
            p1 = population[p1_idx]
        else:
            p1 = population[p2_idx]
            
        if scores[p1_idx] > scores[p2_idx]:
            p2 = population[p2_idx]
        else:
            p2 = population[p1_idx]
            
        # Crossover
        child = 0
        for i in range(N):
            if rng.random() < 0.5:
                if (p1 >> i) & 1:
                    child |= (1 << i)
            else:
                if (p2 >> i) & 1:
                    child |= (1 << i)
                    
        # Mutation
        for _ in range(5): # 5 bit flips
            x = rng.randrange(N)
            child ^= (1 << x)
            
        # Repair
        child = _repair(child)
        
        # Local Search (limited)
        child, score = _local_search(child, time_limit=2.0)
        
        # Replace worst if better
        worst_idx = scores.index(min(scores))
        if score > scores[worst_idx]:
            population[worst_idx] = child
            scores[worst_idx] = score
            
        if score > best_score:
            best_score = score
            best_A = child
            
    # Final Local Search on best
    best_A, best_score = _local_search(best_A, time_limit=10.0)
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if (best_A >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)

# Ensure construct is the entry point
# The task requires construct() -> nonempty subset

if __name__ == "__main__":
    # Simple test
    pass
