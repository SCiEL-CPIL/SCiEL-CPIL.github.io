import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    k = 4
    all_masks = []
    mask_to_idx = {}
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            for l in range(j + 1, n + 1):
                for m in range(l + 1, n + 1):
                    mask = (1 << (i - 1)) | (1 << (j - 1)) | (1 << (l - 1)) | (1 << (m - 1))
                    idx = len(all_masks)
                    all_masks.append(mask)
                    mask_to_idx[mask] = idx
                    
    total_masks = len(all_masks)
    
    def get_conflicts_for_pair(i, j):
        mi = all_masks[i]
        mj = all_masks[j]
        common = mi & mj
        forbidden = (mi | mj) & ~common
        
        available = (~forbidden) & ((1 << 16) - 1)
        free_bits = available & ~common
        
        c_size = bin(common).count('1')
        need = 4 - c_size
        if need < 0:
            return []
        
        free_indices = []
        for b in range(16):
            if free_bits & (1 << b):
                free_indices.append(b)
                
        if len(free_indices) < need:
            return []
            
        res = []
        for comb in combinations(free_indices, need):
            new_mask = common
            for b in comb:
                new_mask |= (1 << b)
            idx = mask_to_idx.get(new_mask)
            if idx is not None and idx != i and idx != j:
                res.append(idx)
        return res

    conflicts = [[None for _ in range(total_masks)] for _ in range(total_masks)]
    
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            conf = get_conflicts_for_pair(i, j)
            conflicts[i][j] = conf
            conflicts[j][i] = conf
            
    best_size = 0
    best_family = []
    
    # Pre-shuffle indices for random restarts
    indices = list(range(total_masks))
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        selected = []
        is_avail = [True] * total_masks
        # We don't need to maintain a list of available, just iterate and check is_avail
        
        # To speed up iteration, we can maintain a list of available indices
        # and remove from it. But removal is O(N).
        # Given N=1820, O(N) removal is ~1820 ops.
        # Iterating all N to find best is 1820 ops.
        # So maintaining list is similar cost.
        # Let's use a list `avail_list` for faster iteration of only available items.
        
        avail_list = indices[:]
        
        while avail_list:
            if time.time() > deadline:
                break
                
            # Pick best c from avail_list
            # Heuristic: Minimize new blocks
            # If avail_list is large, sample it?
            # Let's just pick the first one? No, greedy min-degree is better.
            
            # For small avail_list, check all.
            # For large avail_list, check all might be slow if selected is large.
            
            best_c = -1
            min_cost = float('inf')
            
            # Optimization: If selected is empty, cost is 0 for all. Pick first.
            if not selected:
                best_c = avail_list[0]
            else:
                # To save time, if len(avail_list) > 100 and len(selected) > 20:
                # Sample 20 candidates
                sample = avail_list
                # if len(avail_list) > 50:
                #     sample = random.sample(avail_list, min(50, len(avail_list)))
                
                for c in sample:
                    cost = 0
                    for a in selected:
                        for d in conflicts[c][a]:
                            if is_avail[d]:
                                cost += 1
                                if cost >= min_cost:
                                    break
                        if cost >= min_cost:
                            break
                    if cost < min_cost:
                        min_cost = cost
                        best_c = c
                        
            if best_c == -1:
                break
                
            selected.append(best_c)
            
            # Remove best_c from avail_list
            # Find index of best_c in avail_list
            # This is O(N).
            # We can use a dict? Or just list.remove?
            # list.remove is O(N).
            try:
                avail_list.remove(best_c)
            except ValueError:
                pass # Should not happen
            
            is_avail[best_c] = False
            
            # Update blocks
            for a in selected[:-1]:
                for d in conflicts[best_c][a]:
                    if is_avail[d]:
                        is_avail[d] = False
                        # Remove d from avail_list
                        try:
                            avail_list.remove(d)
                        except ValueError:
                            pass
                            
        if len(selected) > best_size:
            best_size = len(selected)
            best_family = selected[:]
            
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(s)
        
    return result

construct()
