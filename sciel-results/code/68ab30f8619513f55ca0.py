import time
import random
import math

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

def _precompute_comp_bitset(n):
    total_masks = 1 << n
    all_ones = total_masks - 1
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
    return comp_bitset

def _greedy_initial_solution(comp_bitset, n, m):
    """Build a strong initial solution by greedily adding the mask with highest compatibility."""
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

def _chain_initial_solution(n, m):
    """
    Generate an initial solution based on random chains.
    This often yields a higher starting score than pure random or simple greedy.
    """
    total_masks = 1 << n
    selected = set()
    
    # Generate random chains to fill the set
    # Max chain length is n+1 = 13.
    # We want m=40 sets.
    # Let's generate 4 chains of length 10.
    chains_needed = m // 10 + (1 if m % 10 > 0 else 0)
    
    for _ in range(chains_needed):
        perm = list(range(n))
        random.shuffle(perm)
        cur = 0
        chain = []
        for p in perm:
            cur |= (1 << p)
            chain.append(cur)
        
        # Add a random subset of this chain to selected
        # To ensure we get exactly m, we just add until full
        random.shuffle(chain)
        for mask in chain:
            if len(selected) >= m:
                break
            if mask not in selected:
                selected.add(mask)
                
    # If still not full, fill with random
    while len(selected) < m:
        mask = random.randint(0, total_masks - 1)
        selected.add(mask)
        
    return list(selected)

def _local_search_sa(initial_masks, comp_bitset, n, m, deadline, initial_temp=10.0):
    """
    Local search with Simulated Annealing to escape local optima.
    """
    total_masks = 1 << n
    
    selected = list(initial_masks)
    selected_mask = 0
    for mask in selected:
        selected_mask |= (1 << mask)
    
    # Calculate current score
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selected = list(selected)
    
    temp = initial_temp
    cooling = 0.95
    
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
            second_best_new_mask = -1
            second_best_gain = -1
            
            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                if delta > best_gain:
                    if best_gain > second_best_gain:
                        second_best_gain = best_gain
                        second_best_new_mask = best_new_mask
                    best_gain = delta
                    best_new_mask = new_mask
                elif delta > second_best_gain:
                    second_best_gain = delta
                    second_best_new_mask = new_mask
            
            if best_new_mask != -1:
                # Accept best if it improves
                if best_gain > 0:
                    selected[idx] = best_new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                    current_score += best_gain
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    improved = True
                    break  # restart scan
                else:
                    # If best is not improving, consider accepting a worse move with probability
                    # This helps escape local minima
                    # Use best_gain (which is <= 0) for the delta
                    delta = -best_gain # positive value representing the penalty
                    if random.random() < math.exp(-delta / temp):
                        selected[idx] = best_new_mask
                        selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                        current_score += best_gain
                        improved = True
                        break  # restart scan
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1

    return best_score, best_selected

def construct():
    deadline = time.time() + 80.0  # 80s safety margin
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Precompute comp_bitset for all 4096 masks
    comp_bitset = _precompute_comp_bitset(n)
    
    best_overall_score = -1
    best_overall_masks = None
    
    # 1. Greedy initialization
    greedy_masks = _greedy_initial_solution(comp_bitset, n, m)
    score, masks = _local_search_sa(greedy_masks, comp_bitset, n, m, deadline)
    if score > best_overall_score:
        best_overall_score = score
        best_overall_masks = masks
    
    # 2. Chain-based initialization
    chain_masks = _chain_initial_solution(n, m)
    score, masks = _local_search_sa(chain_masks, comp_bitset, n, m, deadline)
    if score > best_overall_score:
        best_overall_score = score
        best_overall_masks = masks
        
    # 3. Multiple random restarts with SA
    while time.time() < deadline:
        # Random initial solution
        chosen = set()
        while len(chosen) < m:
            mask = random.randint(0, total_masks - 1)
            chosen.add(mask)
        init_masks = list(chosen)
        
        score, masks = _local_search_sa(init_masks, comp_bitset, n, m, deadline)
        
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
