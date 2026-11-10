import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    """
    Precompute data structures for fast incremental updates.
    Returns:
    ap3_partners: list of lists of tuples (j, k) for each i in 0..N-1
    ap4_partners: list of lists of tuples (j, k, l) for each i in 0..N-1
    all_3ap_masks: list of integer masks for each 3-AP
    all_4ap_masks: list of integer masks for each 4-AP
    """
    ap3_partners = [[] for _ in range(N)]
    ap4_partners = [[] for _ in range(N)]
    all_3ap_masks = []
    all_4ap_masks = []
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(N - 2 * d):
            b = a + d
            c = a + 2 * d
            # Update partners for a, b, c
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            mask = (1 << a) | (1 << b) | (1 << c)
            all_3ap_masks.append(mask)
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Update partners for a, b, c, e
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            all_4ap_masks.append(mask)
            
    return ap3_partners, ap4_partners, all_3ap_masks, all_4ap_masks

def _count_3aps(mask, all_3ap_masks):
    """Count 3-APs in a bitmask."""
    count = 0
    for m in all_3ap_masks:
        if (mask & m) == m:
            count += 1
    return count

def _count_3aps_delta(mask, x, ap3_partners):
    """Count 3-APs created by adding x to mask."""
    count = 0
    for j, k in ap3_partners[x]:
        if (mask & (1 << j)) and (mask & (1 << k)):
            count += 1
    return count

def _has_4ap(mask, x, ap4_partners):
    """Check if adding x to mask creates a 4-AP."""
    for j, k, l in ap4_partners[x]:
        if (mask & (1 << j)) and (mask & (1 << k)) and (mask & (1 << l)):
            return True
    return False

def _greedy_add_construct(rng, ap3_partners, ap4_partners):
    """Greedy construction: add elements that maximize 3-AP delta."""
    mask = 0
    remaining = list(range(N))
    rng.shuffle(remaining)
    
    while remaining:
        best_x = -1
        best_delta = -1
        for x in remaining:
            if not _has_4ap(mask, x, ap4_partners):
                delta = _count_3aps_delta(mask, x, ap3_partners)
                if delta > best_delta:
                    best_delta = delta
                    best_x = x
        
        if best_x == -1:
            break
            
        mask |= (1 << best_x)
        remaining.remove(best_x)
        
    return mask

def _greedy_remove_construct(rng, ap3_partners, ap4_partners, all_4ap_masks):
    """Greedy removal: start with all elements, remove to break 4-APs."""
    mask = (1 << N) - 1
    
    while True:
        # Find all active 4-APs
        active_aps = []
        for m in all_4ap_masks:
            if (mask & m) == m:
                active_aps.append(m)
                
        if not active_aps:
            break
            
        # Count involvement of each element in active 4-APs
        counts = [0] * N
        for m in active_aps:
            for x in range(N):
                if (m >> x) & 1:
                    counts[x] += 1
                    
        # Find element with max count
        max_count = max(counts)
        candidates = [i for i in range(N) if counts[i] == max_count]
        
        # Randomly pick one to break symmetry
        x = rng.choice(candidates)
        mask &= ~(1 << x)
        
    return mask

def _local_search(mask, deadline, rng, ap3_partners, ap4_partners, all_3ap_masks):
    """
    Local search with add, remove, and swap moves.
    Returns (best_score, best_mask).
    """
    current_score = _count_3aps(mask, all_3ap_masks)
    best_mask = mask
    best_score = current_score
    
    selected = [i for i in range(N) if (mask >> i) & 1]
    unselected = [i for i in range(N) if not ((mask >> i) & 1)]
    
    T = 10.0
    T_min = 0.05
    cooling = 0.9998
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.5 and selected and unselected:
            # Swap
            x_idx = random.randrange(len(selected))
            y_idx = random.randrange(len(unselected))
            x = selected[x_idx]
            y = unselected[y_idx]
            
            # Check 4-AP validity for adding y to mask without x
            temp_mask = mask & ~(1 << x)
            if _has_4ap(temp_mask, y, ap4_partners):
                T *= cooling
                continue
            
            delta_rem = _count_3aps_delta(mask, x, ap3_partners)
            delta_add = _count_3aps_delta(temp_mask, y, ap3_partners)
            delta = delta_add - delta_rem
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                mask = temp_mask | (1 << y)
                selected[x_idx] = y
                unselected[y_idx] = x
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                    
        elif r < 0.75 and unselected:
            # Add
            y_idx = random.randrange(len(unselected))
            y = unselected[y_idx]
            
            if _has_4ap(mask, y, ap4_partners):
                T *= cooling
                continue
                
            delta = _count_3aps_delta(mask, y, ap3_partners)
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                mask |= (1 << y)
                selected.append(y)
                unselected.pop(y_idx)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                    
        elif selected:
            # Remove
            x_idx = random.randrange(len(selected))
            x = selected[x_idx]
            
            delta = -_count_3aps_delta(mask, x, ap3_partners)
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                mask &= ~(1 << x)
                selected.pop(x_idx)
                unselected.append(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                    
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_score, best_mask

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    ap3_partners, ap4_partners, all_3ap_masks, all_4ap_masks = _precompute()
    
    best_score = -1
    best_mask = 0
    
    # Phase 1: Multiple greedy initializations
    remaining_time = DEADLINE - (time.time() - start_time)
    
    # Greedy add
    if remaining_time > 10:
        mask1 = _greedy_add_construct(rng, ap3_partners, ap4_partners)
        score1 = _count_3aps(mask1, all_3ap_masks)
        if score1 > best_score:
            best_score = score1
            best_mask = mask1
    
    # Greedy remove
    if time.time() - start_time < DEADLINE - 10:
        mask2 = _greedy_remove_construct(rng, ap3_partners, ap4_partners, all_4ap_masks)
        score2 = _count_3aps(mask2, all_3ap_masks)
        if score2 > best_score:
            best_score = score2
            best_mask = mask2
    
    # Phase 2: Local search from best initial
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        ls_deadline = start_time + remaining_time - 2
        ls_score, ls_mask = _local_search(best_mask, ls_deadline, rng, ap3_partners, ap4_partners, all_3ap_masks)
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
    
    # Phase 3: More local search from other initializations
    remaining_time = DEADLINE - (time.time() - start_time)
    num_extra = min(3, int(remaining_time // 5)) if remaining_time > 5 else 0
    
    for _ in range(num_extra):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        # New greedy initialization
        greedy_mask = _greedy_add_construct(rng, ap3_partners, ap4_partners)
        
        # Local search from greedy
        ls_deadline = start_time + (DEADLINE - (time.time() - start_time) - 2)
        ls_score, ls_mask = _local_search(greedy_mask, ls_deadline, rng, ap3_partners, ap4_partners, all_3ap_masks)
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
    
    # Convert mask to list of 1-based indices
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
    
    return result
