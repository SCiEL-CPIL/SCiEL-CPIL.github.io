# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            sj = sets[j]
            if si and sj:
                if len(si) < len(sj):
                    is_sub = True
                    s_j_set = set(sj)
                    for x in si:
                        if x not in s_j_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
                elif len(sj) < len(si):
                    is_sub = True
                    s_i_set = set(si)
                    for x in sj:
                        if x not in s_i_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
            elif si and not sj:
                cnt += 1
            elif not si and sj:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s (or 90s if evaluator kills at 90)
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Precompute comparability bitsets
    # comp_bitset[i] is an integer where bit j is 1 if mask i and mask j are comparable
    comp_bitset = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j and ((i & j) == i or (i & j) == j):
                ci |= (1 << j)
        comp_bitset[i] = ci

    def score_mask_set(mask_set):
        # mask_set is a list of 40 masks
        # We need to count pairs (i, j) with i < j such that comparable
        # Using bitsets is faster if we maintain a global selected_mask
        # But for simplicity in restarts, let's use a helper that takes a list
        # Actually, the main loop maintains current_score. This is for initial evaluation.
        s_mask = 0
        for k in mask_set:
            s_mask |= (1 << k)
        cnt = 0
        for i in range(m):
            cnt += (comp_bitset[mask_set[i]] & s_mask).bit_count()
        return cnt // 2

    best_overall_score = -1
    best_overall_selected = []
    
    # Multiple restarts to escape local optima
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Initialization
        selected = []
        selected_mask = 0
        
        # Randomize order of masks for diversity in restarts
        mask_order = list(range(total_masks))
        random.shuffle(mask_order)
        
        for _ in range(m):
            best_mask = -1
            best_gain = -1
            for k in mask_order:
                if (selected_mask >> k) & 1:
                    continue
                gain = (comp_bitset[k] & selected_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_mask = k
            if best_mask == -1:
                # Should not happen as total_masks > m
                break
            selected.append(best_mask)
            selected_mask |= (1 << best_mask)
        
        current_score = score_mask_set(selected)
        
        # 2. Simulated Annealing / Local Search
        temperature = 10.0
        cooling_rate = 0.95
        min_temp = 0.1
        no_improve_counter = 0
        max_no_improve = 1000
        
        while temperature > min_temp and time.time() < deadline:
            improved = False
            
            # Try Single Swap
            idx = random.randint(0, m - 1)
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Pick a random new mask not in selected
            new_mask = -1
            attempts = 0
            while new_mask == -1 and attempts < 100:
                cand = random.randint(0, total_masks - 1)
                if not ((selected_mask >> cand) & 1):
                    new_mask = cand
                attempts += 1
            
            if new_mask != -1:
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                    selected[idx] = new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                    current_score += delta
                    improved = True
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            
            # Try Double Swap (occasionally to escape deeper minima)
            if not improved and random.random() < 0.2:
                idx1 = random.randint(0, m - 1)
                idx2 = random.randint(0, m - 1)
                if idx1 == idx2:
                    continue
                    
                old_mask1 = selected[idx1]
                old_mask2 = selected[idx2]
                
                if old_mask1 == old_mask2:
                    continue
                
                sel_without_both = selected_mask & ~(1 << old_mask1) & ~(1 << old_mask2)
                
                loss1 = (comp_bitset[old_mask1] & sel_without_both).bit_count()
                loss2 = (comp_bitset[old_mask2] & sel_without_both).bit_count()
                # Check if old_mask1 and old_mask2 were comparable to each other
                if (comp_bitset[old_mask1] >> old_mask2) & 1:
                    total_loss = loss1 + loss2 + 1
                else:
                    total_loss = loss1 + loss2
                
                # Pick two random new masks
                new_mask1 = -1
                new_mask2 = -1
                attempts = 0
                while (new_mask1 == -1 or new_mask2 == -1) and attempts < 200:
                    cand1 = random.randint(0, total_masks - 1)
                    if not ((selected_mask >> cand1) & 1) and cand1 != old_mask1 and cand1 != old_mask2:
                        if new_mask1 == -1:
                            new_mask1 = cand1
                        elif cand1 != new_mask1:
                            new_mask2 = cand1
                    cand2 = random.randint(0, total_masks - 1)
                    if not ((selected_mask >> cand2) & 1) and cand2 != old_mask1 and cand2 != old_mask2:
                        if new_mask1 == -1:
                            new_mask1 = cand2
                        elif cand2 != new_mask1:
                            new_mask2 = cand2
                    attempts += 1
                
                if new_mask1 != -1 and new_mask2 != -1:
                    gain1 = (comp_bitset[new_mask1] & sel_without_both).bit_count()
                    gain2 = (comp_bitset[new_mask2] & sel_without_both).bit_count()
                    if (comp_bitset[new_mask1] >> new_mask2) & 1:
                        total_gain = gain1 + gain2 + 1
                    else:
                        total_gain = gain1 + gain2
                    
                    delta = total_gain - total_loss
                    
                    if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                        selected[idx1] = new_mask1
                        selected[idx2] = new_mask2
                        selected_mask = (selected_mask & ~(1 << old_mask1) & ~(1 << old_mask2)) | (1 << new_mask1) | (1 << new_mask2)
                        current_score += delta
                        improved = True
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
            
            if improved:
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_selected = list(selected)
            else:
                temperature *= cooling_rate
                
            if no_improve_counter > max_no_improve:
                break

    # Final Polish: Greedy Hill Climbing on the best found
    if best_overall_selected and time.time() < deadline:
        selected = list(best_overall_selected)
        selected_mask = 0
        for k in selected:
            selected_mask |= (1 << k)
        current_score = best_overall_score
        
        # Strict hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(m):
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                best_new_mask = -1
                best_delta = 0
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = new_mask
                
                if best_new_mask != -1:
                    selected[idx] = best_new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                    current_score += best_delta
                    improved = True
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_overall_selected = list(selected)

    # Convert best_overall_selected to lists of integers
    result = []
    for mask in best_overall_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure types are correct
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
        
    return final_result
# EVOLVE-BLOCK-END
