import numpy as np
import time
import random

def construct():
    n = 12
    num_sets = 40
    max_mask = 1 << n
    masks = np.arange(max_mask, dtype=np.int32)
    
    # Precompute comparability matrix
    a = masks[:, None]
    b = masks[None, :]
    subset_ab = (a & b) == a
    subset_ba = (a & b) == b
    comp = (subset_ab | subset_ba).astype(np.uint8)
    
    # Initial selection: Greedy based on a random chain
    order = list(range(n))
    random.shuffle(order)
    current = 0
    chain_masks = [0]
    for i in order:
        current |= (1 << i)
        chain_masks.append(current)
        
    selected_set = set(chain_masks)
    
    # Greedy fill
    # Use numpy for speed
    selected_list = list(selected_set)
    sel_mask = np.zeros(max_mask, dtype=bool)
    for m in selected_list:
        sel_mask[m] = True
        
    for _ in range(num_sets - len(selected_list)):
        # scores of each candidate against current selected
        scores = comp[:, selected_list].sum(axis=1)
        scores[sel_mask] = -1
        best_idx = np.argmax(scores)
        selected_list.append(best_idx)
        sel_mask[best_idx] = True
        selected_set.add(best_idx)
        
    selected_arr = np.array(selected_list, dtype=np.int32)
    
    # Local Search
    start_time = time.time()
    time_limit = 55.0
    
    def get_total_score(sel):
        sub = comp[np.ix_(sel, sel)]
        return np.triu(sub, 1).sum()
        
    current_score = get_total_score(selected_arr)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i_idx in range(len(selected_arr)):
            i_val = selected_arr[i_idx]
            
            # sums[j] = sum_{y in S} comp[j, y]
            sums = comp[:, selected_arr].sum(axis=1)
            # We want sum_{y in S, y != i_val} comp[j, y]
            # which is sums[j] - comp[j, i_val]
            sums_excl_i = sums - comp[:, i_val]
            
            # Find best j not in S
            scores = sums_excl_i.copy()
            scores[sel_mask] = -1
            
            best_j = np.argmax(scores)
            best_score_j = scores[best_j]
            
            # Current contribution of i_val
            current_contrib = sums_excl_i[i_val]
            
            if best_score_j > current_contrib:
                # Swap
                sel_mask[i_val] = False
                sel_mask[best_j] = True
                selected_arr[i_idx] = best_j
                current_score = get_total_score(selected_arr)
                improved = True
                break
        
    result = []
    for mask in selected_arr:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
