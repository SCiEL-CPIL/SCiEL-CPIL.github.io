# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    all_ones = total_masks - 1  # 0xFFF
    
    comp_bitset = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
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
    
    # Greedy construction with smart ordering
    selected = []
    selected_mask = 0
    
    # Start with empty set
    selected.append(0)
    selected_mask |= (1 << 0)
    
    # Greedily add masks that maximize comparability
    for _ in range(m - 1):
        best_mask = -1
        best_score = -1
        for k in range(total_masks):
            if (selected_mask >> k) & 1:
                continue
            score = (comp_bitset[k] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = k
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)
    
    # Calculate current score
    def calc_score(sel, sel_mask):
        s = 0
        for i in range(m):
            s += (comp_bitset[sel[i]] & sel_mask).bit_count()
        return s // 2
    
    current_score = calc_score(selected, selected_mask)
    best_score = current_score
    best_selected = list(selected)
    
    # Local search: single swaps with simulated annealing
    improved = True
    temp = 5.0  # initial temperature
    cooling = 0.9995
    
    while improved and time.time() < deadline:
        improved = False
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                
                # Accept if better, or with probability exp(-delta/temp) if worse
                accept = False
                if delta > 0:
                    accept = True
                elif delta < 0 and temp > 0.01:
                    if random.random() < (2.718281828 ** (-delta / temp)):
                        accept = True
                
                if accept:
                    selected[idx] = new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    improved = True
                    break  # restart scan
        
        temp *= cooling
    
    # Multi-swap local search: try swapping 2 elements at a time
    if time.time() < deadline:
        ms_improved = True
        while ms_improved and time.time() < deadline:
            ms_improved = False
            for i in range(m - 1):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() >= deadline:
                        break
                    mi = selected[i]
                    mj = selected[j]
                    sel_without = selected_mask & ~(1 << mi) & ~(1 << mj)
                    loss = ((comp_bitset[mi] & sel_without) + (comp_bitset[mj] & sel_without)).bit_count()
                    # Also need to account for mi-mj comparability
                    if (comp_bitset[mi] >> mj) & 1:
                        loss -= 1  # They were comparable, now they're removed
                    
                    best_pair = (-1, -1)
                    best_delta = 0
                    for ni in range(total_masks):
                        if (selected_mask >> ni) & 1:
                            continue
                        gain_i = (comp_bitset[ni] & sel_without).bit_count()
                        for nj in range(ni + 1, total_masks):
                            if (selected_mask >> nj) & 1 or nj == ni:
                                continue
                            gain_j = (comp_bitset[nj] & sel_without).bit_count()
                            # Check if ni and nj are comparable
                            if (comp_bitset[ni] >> nj) & 1:
                                gain_j += 1
                            delta = (gain_i + gain_j) - loss
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (ni, nj)
                    
                    if best_delta > 0:
                        ni, nj = best_pair
                        selected[i] = ni
                        selected[j] = nj
                        selected_mask = (selected_mask & ~(1 << mi) & ~(1 << mj)) | (1 << ni) | (1 << nj)
                        current_score += best_delta
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
                        ms_improved = True
                        break
                if ms_improved:
                    break
    
    # Final polish: exhaustive single-swap improvement
    if time.time() < deadline:
        polish_improved = True
        while polish_improved and time.time() < deadline:
            polish_improved = False
            for idx in range(m):
                if time.time() >= deadline:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > 0:
                        selected[idx] = new_mask
                        selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
                        polish_improved = True
                        break
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure proper types
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
# EVOLVE-BLOCK-END
