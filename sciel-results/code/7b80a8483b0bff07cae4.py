"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
# EVOLVE-BLOCK-START

def _count_3aps_bitmask(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    elements = [i+1 for i in range(256) if (mask >> i) & 1]
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i+1, n):
            b = elements[j]
            c = 2*b - a
            if c > 256:
                break
            if (mask >> (c-1)) & 1:
                count += 1
    return count

def _delta_3aps_add(mask, x):
    """Compute how many new 3-APs are formed by adding element x to the set."""
    delta = 0
    # x as third term: a, b, x where 2b - a = x => a = 2b - x
    # Need a < b < x, a in set, b in set
    for b in range(1, x):
        if (mask >> (b-1)) & 1:
            a = 2*b - x
            if a >= 1 and a < b and (mask >> (a-1)) & 1:
                delta += 1
    # x as middle term: a, x, b where a + b = 2x
    # Need a < x < b, a in set, b in set
    for a in range(1, x):
        if (mask >> (a-1)) & 1:
            b = 2*x - a
            if b > x and b <= 256 and (mask >> (b-1)) & 1:
                delta += 1
    # x as first term: x, b, c where b = x+d, c = x+2d
    # Need x < b < c, b in set, c in set
    for b in range(x+1, 257):
        if (mask >> (b-1)) & 1:
            c = 2*b - x
            if c > b and c <= 256 and (mask >> (c-1)) & 1:
                delta += 1
    return delta

def _delta_3aps_remove(mask, x):
    """Compute how many 3-APs are lost by removing element x from the set."""
    delta = 0
    # x as third term: a, b, x where 2b - a = x => a = 2b - x
    for b in range(1, x):
        if (mask >> (b-1)) & 1:
            a = 2*b - x
            if a >= 1 and a < b and (mask >> (a-1)) & 1:
                delta += 1
    # x as middle term: a, x, b where a + b = 2x
    for a in range(1, x):
        if (mask >> (a-1)) & 1:
            b = 2*x - a
            if b > x and b <= 256 and (mask >> (b-1)) & 1:
                delta += 1
    # x as first term: x, b, c where b = x+d, c = x+2d
    for b in range(x+1, 257):
        if (mask >> (b-1)) & 1:
            c = 2*b - x
            if c > b and c <= 256 and (mask >> (c-1)) & 1:
                delta += 1
    return delta

def _has_4ap_fast(mask):
    """Check if mask contains any 4-term AP. Returns True if it does."""
    # A 4-AP is a, a+d, a+2d, a+3d.
    # We can check for each d, if there exists a such that all 4 bits are set.
    # This is equivalent to checking if (mask & (mask >> d) & (mask >> 2d) & (mask >> 3d)) != 0
    # But we must ensure a, a+d, a+2d, a+3d are within [1, 256].
    # The bit shift handles the "within bounds" implicitly if we mask out higher bits, 
    # but we need to be careful with the start 'a'.
    # If bit k is set in the intersection, it means a=k+1 is the start of a 4-AP with diff d.
    # We need a+3d <= 256. So k+1+3d <= 256 => k <= 255-3d.
    # So we only care about bits 0 to 255-3d.
    
    for d in range(1, 86): # max d such that 1+3d <= 256 => 3d <= 255 => d <= 85
        # Shifts: d, 2d, 3d
        # Intersection of mask and shifted masks
        # We want to know if there is a bit set in mask AND mask>>d AND mask>>2d AND mask>>3d
        # Let M = mask
        # We check if (M & (M >> d) & (M >> 2*d) & (M >> 3*d)) has any bits set in the valid range.
        # Valid range for start 'a' (0-indexed bit pos k): k+3d < 256 => k < 256-3d
        # So we mask the result with (1 << (256 - 3*d)) - 1
        
        inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
        valid_mask = (1 << (256 - 3*d)) - 1
        if inter & valid_mask:
            return True
    return False

def _construct_greedy():
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    # We iterate through numbers 1 to 256.
    # To maximize 3-APs, we might want a denser set.
    # Standard greedy: add x if it doesn't create a 4-AP.
    # To bias towards more 3-APs, we can try adding x if it creates many 3-APs, 
    # but greedy addition order matters. 
    # Let's try a few different orders or just standard increasing order first.
    
    # A known good strategy for no 4-AP is to take numbers with only 0s and 1s in base 3? No, that's for no 3-AP.
    # For no 4-AP, we can take numbers that don't have certain patterns.
    # Let's stick to greedy: add x if safe.
    
    # To get a better starting point, we can run multiple greedy passes with different random perturbations or orders.
    # But let's start with a deterministic greedy that prioritizes adding elements that form 3-APs with existing ones.
    
    # Actually, simple increasing greedy often yields ~100-120 elements.
    # Let's try to be smarter: 
    # We want to maximize 3-APs. 
    # Let's just do a standard greedy first, then local search.
    
    for x in range(1, 257):
        # Check if adding x creates a 4-AP
        new_mask = mask | (1 << (x-1))
        if not _has_4ap_fast(new_mask):
            mask = new_mask
    return mask

def _local_search_sa(mask, time_limit):
    """Simulated annealing to improve the set."""
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    
    def get_elements(m):
        return [i+1 for i in range(256) if (m >> i) & 1]
    
    current_mask = mask
    current_score = best_score
    elements = get_elements(current_mask)
    not_in_set = [x for x in range(1, 257) if not (mask >> (x-1)) & 1]
    
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        if T < T_min:
            T = T_min
        
        # Choose move type
        move_type = random.random()
        if move_type < 0.4:
            # Add an element
            if not not_in_set:
                continue
            x = random.choice(not_in_set)
            new_mask = current_mask | (1 << (x-1))
            if _has_4ap_fast(new_mask):
                continue
            delta = _delta_3aps_add(current_mask, x)
            new_score = current_score + delta
            # Accept if better or with probability exp(-delta/T) if worse
            if delta >= 0 or random.random() < math.exp(-delta / T):
                current_mask = new_mask
                current_score = new_score
                elements = get_elements(current_mask)
                not_in_set = [x for x in range(1, 257) if not (current_mask >> (x-1)) & 1]
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        elif move_type < 0.7:
            # Remove an element
            if not elements:
                continue
            x = random.choice(elements)
            new_mask = current_mask & ~(1 << (x-1))
            delta = -_delta_3aps_remove(current_mask, x)
            new_score = current_score + delta
            if delta >= 0 or random.random() < math.exp(-delta / T):
                current_mask = new_mask
                current_score = new_score
                elements = get_elements(current_mask)
                not_in_set = [x for x in range(1, 257) if not (current_mask >> (x-1)) & 1]
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Swap: remove one, add another
            if not elements or not not_in_set:
                continue
            x_remove = random.choice(elements)
            x_add = random.choice(not_in_set)
            new_mask = (current_mask & ~(1 << (x_remove-1))) | (1 << (x_add-1))
            if _has_4ap_fast(new_mask):
                continue
            delta = _delta_3aps_add(new_mask & ~(1 << (x_add-1)), x_add) - _delta_3aps_remove(current_mask, x_remove)
            # More accurate: compute delta for the swap
            # Remove x_remove from current, then add x_add
            temp_mask = current_mask & ~(1 << (x_remove-1))
            delta = _delta_3aps_add(temp_mask, x_add) - _delta_3aps_remove(current_mask, x_remove)
            new_score = current_score + delta
            if delta >= 0 or random.random() < math.exp(-delta / T):
                current_mask = new_mask
                current_score = new_score
                elements = get_elements(current_mask)
                not_in_set = [x for x in range(1, 257) if not (current_mask >> (x-1)) & 1]
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        T *= cooling_rate
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5 second safety margin
    
    # Initial construction
    # Try a few different greedy strategies to get a good start
    best_initial_mask = 0
    best_initial_score = 0
    
    # Strategy 1: Standard increasing greedy
    mask1 = _construct_greedy()
    score1 = _count_3aps_bitmask(mask1)
    
    # Strategy 2: Random order greedy (multiple times)
    for _ in range(5):
        if time.time() > deadline - 10: break
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        mask = 0
        for x in random_order:
            new_mask = mask | (1 << (x-1))
            if not _has_4ap_fast(new_mask):
                mask = new_mask
        score = _count_3aps_bitmask(mask)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_mask = mask
    
    if score1 > best_initial_score:
        best_initial_score = score1
        best_initial_mask = mask1
        
    # Local search with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        best_mask = _local_search_sa(best_initial_mask, remaining_time - 1)
    else:
        best_mask = best_initial_mask
        
    # Final check and convert to list
    if _has_4ap_fast(best_mask):
        # Fallback: repair by removing elements involved in 4-APs
        # This should rarely happen if local search is correct
        mask = best_mask
        while _has_4ap_fast(mask):
             # Find a 4-AP and remove one element
             # Simple approach: iterate d, find a, remove a
             found = False
             for d in range(1, 86):
                 inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
                 valid_mask = (1 << (256 - 3*d)) - 1
                 inter &= valid_mask
                 if inter:
                     # Find the lowest set bit
                     k = (inter & -inter).bit_length() - 1
                     a = k + 1
                     mask &= ~(1 << (a-1))
                     found = True
                     break
             if not found: break
        best_mask = mask

    result = [i+1 for i in range(256) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
