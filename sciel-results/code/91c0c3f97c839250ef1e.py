"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

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

def _has_4ap_fast(mask):
    """Check if mask contains any 4-term AP. Returns True if it does."""
    for d in range(1, 86):
        inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
        valid_mask = (1 << (256 - 3*d)) - 1
        if inter & valid_mask:
            return True
    return False

def _construct_greedy():
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    for x in range(1, 257):
        new_mask = mask | (1 << (x-1))
        if not _has_4ap_fast(new_mask):
            mask = new_mask
    return mask

def _delta_3ap(mask, x, is_add):
    """
    Calculate the change in the number of 3-APs if we add or remove x.
    mask: current bitmask
    x: element to add/remove (1-indexed)
    is_add: True if adding, False if removing
    """
    # If removing, x is currently in mask. If adding, x is not.
    # We want to count 3-APs formed/destroyed involving x.
    # A 3-AP is (a, b, c) with a < b < c and b-a = c-b.
    # x can be a, b, or c.
    
    delta = 0
    # Case 1: x is the start (a)
    # (x, x+d, x+2d). Need x+d, x+2d in A.
    # If adding: count how many d such that x+d, x+2d in A.
    # If removing: count how many d such that x+d, x+2d in A (these APs are destroyed).
    # Note: The other two elements must be in the *current* mask for the AP to exist.
    # If adding, the other two are in A. If removing, the other two are in A\{x} (which is just A since x!=x+d).
    
    # Case 2: x is the middle (b)
    # (x-d, x, x+d). Need x-d, x+d in A.
    
    # Case 3: x is the end (c)
    # (x-2d, x-d, x). Need x-2d, x-d in A.
    
    # We iterate d.
    # Max d for x as start: x+2d <= 256 => d <= (256-x)//2
    # Max d for x as middle: x+d <= 256 and x-d >= 1 => d <= min(x-1, 256-x)
    # Max d for x as end: x-2d >= 1 => d <= (x-1)//2
    
    # For efficiency, we can just iterate d up to max possible (128) and check bounds.
    # But checking bounds is cheap.
    
    # Optimization: Precompute nothing, just loop.
    # We use bit checks.
    
    # Start case
    max_d_start = (256 - x) // 2
    for d in range(1, max_d_start + 1):
        b = x + d
        c = x + 2*d
        if (mask >> (b-1)) & 1:
            if (mask >> (c-1)) & 1:
                delta += 1
                
    # Middle case
    max_d_mid = min(x - 1, 256 - x)
    for d in range(1, max_d_mid + 1):
        a = x - d
        c = x + d
        if (mask >> (a-1)) & 1:
            if (mask >> (c-1)) & 1:
                delta += 1
                
    # End case
    max_d_end = (x - 1) // 2
    for d in range(1, max_d_end + 1):
        a = x - 2*d
        b = x - d
        if (mask >> (a-1)) & 1:
            if (mask >> (b-1)) & 1:
                delta += 1
                
    if is_add:
        return delta
    else:
        return -delta

def _creates_4ap(mask, x):
    """
    Check if adding x to the set represented by mask creates a 4-AP.
    x is NOT in mask.
    """
    # x can be part of a 4-AP in 4 positions.
    # 1. x, x+d, x+2d, x+3d
    # 2. x-d, x, x+d, x+2d
    # 3. x-2d, x-d, x, x+d
    # 4. x-3d, x-2d, x-d, x
    
    # 1. x as start
    max_d = (256 - x) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x+d-1)) & 1) and ((mask >> (x+2*d-1)) & 1) and ((mask >> (x+3*d-1)) & 1):
            return True
            
    # 2. x as second
    # x-d >= 1 => d <= x-1. x+2d <= 256 => d <= (256-x)//2
    max_d = min(x - 1, (256 - x) // 2)
    for d in range(1, max_d + 1):
        if ((mask >> (x-d-1)) & 1) and ((mask >> (x+d-1)) & 1) and ((mask >> (x+2*d-1)) & 1):
            return True
            
    # 3. x as third
    # x-2d >= 1 => d <= (x-1)//2. x+d <= 256 => d <= 256-x
    max_d = min((x - 1) // 2, 256 - x)
    for d in range(1, max_d + 1):
        if ((mask >> (x-2*d-1)) & 1) and ((mask >> (x-d-1)) & 1) and ((mask >> (x+d-1)) & 1):
            return True
            
    # 4. x as fourth
    # x-3d >= 1 => d <= (x-1)//3
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x-3*d-1)) & 1) and ((mask >> (x-2*d-1)) & 1) and ((mask >> (x-d-1)) & 1):
            return True
            
    return False

def _local_search_sa(mask, time_limit):
    """Simulated Annealing local search."""
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    current_mask = mask
    current_score = best_score
    
    # Initial temperature
    T = 10.0
    cooling = 0.99999
    
    # Elements in set
    elements = [i+1 for i in range(256) if (current_mask >> i) & 1]
    
    while time.time() < start_time + time_limit:
        # Random move: Add, Remove, or Swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            # Pick random x not in set
            # To speed up, pick random x in [1, 256] and check
            x = random.randint(1, 256)
            if (current_mask >> (x-1)) & 1:
                continue
            if _creates_4ap(current_mask, x):
                continue
            new_mask = current_mask | (1 << (x-1))
            delta = _delta_3ap(current_mask, x, True)
            new_score = current_score + delta
            # SA accept
            if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / T):
                current_mask = new_mask
                current_score = new_score
                elements.append(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        elif move_type < 0.6:
            # Remove
            if not elements:
                continue
            idx = random.randint(0, len(elements)-1)
            x = elements[idx]
            # Remove x from elements list
            elements[idx] = elements[-1]
            elements.pop()
            new_mask = current_mask & ~(1 << (x-1))
            delta = _delta_3ap(current_mask, x, False)
            new_score = current_score + delta
            if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Swap
            if not elements:
                continue
            idx = random.randint(0, len(elements)-1)
            x_remove = elements[idx]
            x_add = random.randint(1, 256)
            if x_add == x_remove:
                continue
            if (current_mask >> (x_add-1)) & 1:
                continue
            
            # Check if swap is valid
            # Temporarily remove x_remove
            temp_mask = current_mask & ~(1 << (x_remove-1))
            if _creates_4ap(temp_mask, x_add):
                continue
            
            # Calculate delta
            # Remove x_remove: delta1
            delta1 = _delta_3ap(current_mask, x_remove, False)
            # Add x_add to (current_mask without x_remove): delta2
            # Note: _delta_3ap uses the mask passed. 
            # The APs formed by x_add might involve x_remove? 
            # If x_add and x_remove form a 3-AP with some z, that AP is destroyed by removal and created by addition?
            # Let's be precise.
            # Score_new = Score_old - APs_involving_x_remove + APs_involving_x_add_in_new_set
            # APs_involving_x_remove: calculated on current_mask.
            # APs_involving_x_add_in_new_set: calculated on temp_mask.
            # But wait, if x_add and x_remove are in a 3-AP with z, 
            # e.g., (x_remove, z, x_add). 
            # In current_mask, this AP exists. 
            # In new_set, x_remove is gone, so this AP is gone. 
            # In temp_mask, x_remove is gone. 
            # _delta_3ap(temp_mask, x_add, True) counts APs formed by adding x_add to temp_mask.
            # Since x_remove is not in temp_mask, the AP (x_remove, z, x_add) is NOT counted as "formed".
            # And _delta_3ap(current_mask, x_remove, False) counts it as "destroyed".
            # So the net change is correct.
            
            delta2 = _delta_3ap(temp_mask, x_add, True)
            delta = delta1 + delta2
            new_score = current_score + delta
            new_mask = temp_mask | (1 << (x_add-1))
            
            if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / T):
                current_mask = new_mask
                current_score = new_score
                elements[idx] = x_add
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor to allow some exploration

    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5 second safety margin
    
    # Initial construction
    best_initial_mask = 0
    best_initial_score = 0
    
    # Strategy 1: Standard increasing greedy
    mask1 = _construct_greedy()
    score1 = _count_3aps_bitmask(mask1)
    best_initial_mask = mask1
    best_initial_score = score1
    
    # Strategy 2: Random order greedy (multiple times)
    for _ in range(10):
        if time.time() > deadline - 20: break
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
            
    # Local search
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        best_mask = _local_search_sa(best_initial_mask, remaining_time - 1)
    else:
        best_mask = best_initial_mask
        
    # Final check and convert to list
    if _has_4ap_fast(best_mask):
        # Fallback: repair
        mask = best_mask
        while _has_4ap_fast(mask):
             found = False
             for d in range(1, 86):
                 inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
                 valid_mask = (1 << (256 - 3*d)) - 1
                 inter &= valid_mask
                 if inter:
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
