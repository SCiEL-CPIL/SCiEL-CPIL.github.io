import time
import sys
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability using NumPy
    masks = np.arange(total_masks, dtype=np.int32)
    comp = np.zeros((total_masks, total_masks), dtype=np.int8)
    
    chunk_size = 512
    for start in range(0, total_masks, chunk_size):
        end = min(start + chunk_size, total_masks)
        i_chunk = masks[start:end, np.newaxis]
        j_all = masks[np.newaxis, :]
        subset_ij = ((i_chunk & ~j_all) == 0)
        subset_ji = ((j_all & ~i_chunk) == 0)
        comp[start:end, :] = (subset_ij | subset_ji).astype(np.int8)
    
    def count_comparables(fam):
        fam_arr = np.array(fam, dtype=np.int32)
        cnt = 0
        L = len(fam_arr)
        for i in range(L):
            cnt += int(np.sum(comp[fam_arr[i], fam_arr[i+1:]]))
        return cnt
    
    def local_search(family, time_limit):
        """Run local search on family until time_limit or no improvement."""
        best_family = family[:]
        
        while time.time() < time_limit:
            improved = False
            fam_arr = np.array(best_family, dtype=np.int32)
            
            # contrib[i] = sum of comp[fam[i]][fam[j]] for j != i
            # comp[i][i] = 1, so sum(comp[fam[i]]) includes self
            contribs = np.array([int(np.sum(comp[fam_arr[i], fam_arr])) - 1 for i in range(m)], dtype=np.int32)
            
            fam_set = set(best_family)
            
            # For each candidate s, compute sum_comp[s] = sum of comp[s][fam[j]] for j in family
            # We can compute this as comp[s, fam_arr].sum()
            # But doing this for all 4096 s each iteration is 4096*40 = 163k ops, fine
            
            all_sums = np.array([int(np.sum(comp[s, fam_arr])) for s in range(total_masks)], dtype=np.int32)
            
            # Find best swap
            best_delta = 0
            best_i = -1
            best_s = -1
            
            for i in range(m):
                fi = best_family[i]
                ci = contribs[i]
                # For each s not in fam_set, delta = (all_sums[s] - comp[s, fi]) - ci
                # We want max delta > 0
                for s in range(total_masks):
                    if s in fam_set:
                        continue
                    val = all_sums[s] - comp[s, fi]
                    delta = val - ci
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_s = s
            
            if best_delta > 0 and best_i != -1 and best_s != -1:
                best_family[best_i] = best_s
                improved = True
            
            if not improved:
                break
        
        return best_family
    
    def greedy_init():
        family = []
        cur = 0
        family.append(cur)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
        
        used = set(family)
        
        for _ in range(m - len(family)):
            best_gain = -1
            best_mask = None
            
            for s in range(total_masks):
                if s in used:
                    continue
                gain = int(np.sum(comp[s, family]))
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        
        return family
    
    # Multi-start local search
    deadline = time.time() + 85
    
    best_family = None
    best_score = -1
    
    # Start 1: greedy init
    family = greedy_init()
    ls_family = local_search(family, deadline)
    score = count_comparables(ls_family)
    if score > best_score:
        best_score = score
        best_family = ls_family
    
    # Start 2: different greedy init (start with full set and remove)
    if time.time() < deadline - 10:
        family2 = []
        cur = (1 << n) - 1
        family2.append(cur)
        for i in range(n):
            cur &= ~(1 << i)
            family2.append(cur)
        
        used = set(family2)
        for _ in range(m - len(family2)):
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in used:
                    continue
                gain = int(np.sum(comp[s, family2]))
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family2.append(best_mask)
            used.add(best_mask)
        
        ls_family2 = local_search(family2, deadline)
        score2 = count_comparables(ls_family2)
        if score2 > best_score:
            best_score = score2
            best_family = ls_family2
    
    # Start 3: random init with local search
    if time.time() < deadline - 5:
        np.random.seed(42)
        all_masks = list(range(total_masks))
        np.random.shuffle(all_masks)
        family3 = all_masks[:m]
        ls_family3 = local_search(family3, deadline)
        score3 = count_comparables(ls_family3)
        if score3 > best_score:
            best_score = score3
            best_family = ls_family3
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
