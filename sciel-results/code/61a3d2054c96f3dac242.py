<<<<<<< SEARCH
def local_search_swap(best_mask, best_len, rng, deadline):
    """
    Try swap moves: remove one point, add another.
    Returns (new_mask, new_len, improved)
    """
    pts_in_set = mask_to_list(best_mask)
    pts_not_in_set = [i for i in range(NUM_PTS) if not (best_mask & (1 << i))]
    
    rng.shuffle(pts_in_set)
    rng.shuffle(pts_not_in_set)
    
    for p_remove in pts_in_set:
        if time.time() > deadline:
            return best_mask, best_len, False
        
        for p_add in pts_not_in_set:
            if time.time() > deadline:
                return best_mask, best_len, False
            
            # Try swapping p_remove with p_add
            temp_mask = best_mask & ~(1 << p_remove)
            temp_mask |= (1 << p_add)
            
            # Check if valid (no isosceles) and maximal
            # First check validity
            valid = True
            m = temp_mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                m ^= lsb
                if temp_mask & iso_mask[p_add][i]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check maximality
            if is_maximal_fast(temp_mask):
                new_len = bin(temp_mask).count('1')
                if new_len < best_len:
                    return temp_mask, new_len, True
    
    return best_mask, best_len, False
=======
def local_search_swap(best_mask, best_len, rng, deadline):
    """
    Try swap moves: remove one point, add another.
    Returns (new_mask, new_len, improved)
    """
    pts_in_set = mask_to_list(best_mask)
    pts_not_in_set = [i for i in range(NUM_PTS) if not (best_mask & (1 << i))]
    
    rng.shuffle(pts_in_set)
    rng.shuffle(pts_not_in_set)
    
    for p_remove in pts_in_set:
        if time.time() > deadline:
            return best_mask, best_len, False
        
        temp_mask_without = best_mask & ~(1 << p_remove)
        
        # Find which points can be added to temp_mask_without
        addable = []
        for p_add in pts_not_in_set:
            if can_add(p_add, temp_mask_without):
                addable.append(p_add)
        
        if not addable:
            continue
            
        # Try each addable point
        for p_add in addable:
            if time.time() > deadline:
                return best_mask, best_len, False
            
            # Try swapping p_remove with p_add
            temp_mask = temp_mask_without | (1 << p_add)
            
            # Check maximality
            if is_maximal_fast(temp_mask):
                new_len = bin(temp_mask).count('1')
                if new_len < best_len:
                    return temp_mask, new_len, True
    
    return best_mask, best_len, False
>>>>>>> REPLACE

