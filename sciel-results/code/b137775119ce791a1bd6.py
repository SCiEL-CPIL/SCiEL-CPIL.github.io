import time
import random

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    """
    n = P['n']
    m = len(sets)
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                if mi != mj:
                    count += 1
    return count

def _greedy_initial_solution(comp_bitset, deadline):
    """Build a strong initial solution by greedily adding the mask with highest compatibility."""
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    selected = []
    selected_mask = 0
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in range(total_masks):
            if (selected_mask >> mask) & 1:
                continue
            score = (comp_bitset[mask] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask == -1:
            break
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)
    
    # Fill remaining with random if needed
    while len(selected) < m:
        mask = random.randint(0, total_masks - 1)
        if not ((selected_mask >> mask) & 1):
            selected.append(mask)
            selected_mask |= (1 << mask)
    
    return selected

def _local_search(initial_masks, comp_bitset, deadline):
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    selected = list(initial_masks)
    selected_mask = 0
    for mask in selected:
        selected_mask |= (1 << mask)
    
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selected = list(selected)
    
    # Local search: single swaps with full scan
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Full scan to find the best replacement
            best_new_mask = -1
            best_gain = -1
            
            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                if delta > best_gain:
                    best_gain = delta
                    best_new_mask = new_mask
            
            if best_new_mask != -1 and best_gain > 0:
                selected[idx] = best_new_mask
                selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                current_score += best_gain
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
                improved = True
                break  # restart scan
    
    return best_score, best_selected

def construct():
    deadline = time.time() + 50.0  # 50s safety margin
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    all_ones = total_masks - 1
    
    # Precompute comp_bitset for all 4096 masks
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
        if i != 0:
            sub = (i - 1) & i
            while True:
                ci |= (1 << sub)
                if sub == 0:
                    break
                sub = (sub - 1) & i
        # Enumerate all proper supermasks of i
        comp = all_ones ^ i
        if comp != 0:
            sub = (comp - 1) & comp
            while True:
                s = i | sub
                if s != i:
                    ci |= (1 << s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        comp_bitset[i] = ci
    
    best_overall_score = -1
    best_overall_masks = None
    
    # Greedy initialization
    greedy_masks = _greedy_initial_solution(comp_bitset, deadline)
    score, masks = _local_search(greedy_masks, comp_bitset, deadline)
    if score > best_overall_score:
        best_overall_score = score
        best_overall_masks = masks
    
    # Multiple random restarts
    while time.time() < deadline:
        # Random initial solution
        chosen = set()
        while len(chosen) < m:
            mask = random.randint(0, total_masks - 1)
            chosen.add(mask)
        init_masks = list(chosen)
        
        score, masks = _local_search(init_masks, comp_bitset, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_masks = masks
    
    # Convert best_overall_masks to lists of integers
    result = []
    for mask in best_overall_masks:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
