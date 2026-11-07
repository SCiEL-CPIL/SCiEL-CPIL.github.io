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

def _greedy_construct_fast(deadline):
    """
    Fast greedy construction using potential array.
    """
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Random start
    start_mask = random.randint(0, total_masks - 1)
    selected = [start_mask]
    is_selected = [False] * total_masks
    is_selected[start_mask] = True
    
    # potential[i] = number of selected sets comparable to i
    potential = [0] * total_masks
    
    # Initialize potential for the start mask
    # Actually, potential for future picks depends on the current set.
    # When we pick a new set, we update potential for all others.
    # But we need potential for the *next* pick to be based on the *current* set.
    # So, after picking `start_mask`, we update `potential` for all `j` != `start_mask`.
    
    for j in range(total_masks):
        if j == start_mask:
            continue
        if (start_mask & j) == start_mask or (start_mask & j) == j:
            potential[j] = 1
        # else 0
            
    selected.append(start_mask) # Wait, I added it above.
    # Let's restructure.
    
    selected = []
    is_selected = [False] * total_masks
    potential = [0] * total_masks
    
    # Step 1: Pick random start
    start_mask = random.randint(0, total_masks - 1)
    selected.append(start_mask)
    is_selected[start_mask] = True
    
    # Update potential based on start_mask
    for j in range(total_masks):
        if j != start_mask:
            if (start_mask & j) == start_mask or (start_mask & j) == j:
                potential[j] = 1
                
    # Step 2..40
    for step in range(1, m):
        # Find max potential among unselected
        best_val = -1
        best_idx = -1
        for j in range(total_masks):
            if not is_selected[j]:
                if potential[j] > best_val:
                    best_val = potential[j]
                    best_idx = j
                    
        if best_idx == -1:
            # Should not happen
            break
            
        selected.append(best_idx)
        is_selected[best_idx] = True
        
        # Update potential for all unselected j
        # If best_idx is comparable to j, potential[j] += 1
        for j in range(total_masks):
            if not is_selected[j]:
                if (best_idx & j) == best_idx or (best_idx & j) == j:
                    potential[j] += 1
                    
        # Time check
        if step % 10 == 0 and time.time() > deadline:
            break
            
    if len(selected) < m:
        # Fill remaining
        while len(selected) < m:
            mask = random.randint(0, total_masks - 1)
            if not is_selected[mask]:
                selected.append(mask)
                is_selected[mask] = True
                # No need to update potential if we are just filling and returning
        
    # Calculate score
    score = 0
    for i in range(m):
        for j in range(i + 1, m):
            if (selected[i] & selected[j]) == selected[i] or (selected[i] & selected[j]) == selected[j]:
                if selected[i] != selected[j]:
                    score += 1
                    
    return selected, score

def construct():
    deadline = time.time() + 55.0
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    best_score = -1
    best_masks = None
    
    restarts = 0
    while time.time() < deadline:
        masks, score = _greedy_construct_fast(deadline)
        
        if score > best_score:
            best_score = score
            best_masks = masks
            
        restarts += 1
        
        # If we have a very high score, stop?
        # Max is 780.
        if best_score > 650 and restarts > 100:
             break

    if best_masks is None:
        best_masks = []
        used = set()
        while len(best_masks) < m:
            mask = random.randint(0, total_masks - 1)
            if mask not in used:
                best_masks.append(mask)
                used.add(mask)
                
    result = []
    for mask in best_masks:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
