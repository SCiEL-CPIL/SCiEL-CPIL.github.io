import time
import sys
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability matrix using NumPy for speed
    # comp[i][j] = 1 if i is subset of j or j is subset of i
    i_arr = np.arange(total_masks)
    i_exp = i_arr[:, None]
    j_exp = i_arr[None, :]
    
    # i is subset of j: (i & ~j) == 0
    # j is subset of i: (j & ~i) == 0
    i_subset_j = ((i_exp & ~j_exp) == 0)
    j_subset_i = ((j_exp & ~i_exp) == 0)
    comp = (i_subset_j | j_subset_i).astype(np.int8)
    
    # 1. Initialize with a good family using greedy
    family = []
    cur = 0
    family.append(cur)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    used = set(family)
    
    # Greedily add sets that maximize comparable pairs
    for _ in range(m - len(family)):
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used:
                continue
            gain = sum(comp[s][f] for f in family)
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family.append(best_mask)
        used.add(best_mask)
    
    def count_comparables(fam):
        arr = np.array(fam, dtype=np.int32)
        return int(np.sum(comp[np.ix_(arr, arr)]) // 2)
    
    def local_search(fam_list, deadline):
        """Run local search on a family, returning improved family."""
        fam = list(fam_list)
        current_score = count_comparables(fam)
        
        while time.time() < deadline:
            improved = False
            
            # Compute contribs: contrib[i] = sum of comp[fam[i]][fam[j]] for j != i
            fam_arr = np.array(fam, dtype=np.int32)
            contribs = np.array([
                int(np.sum(comp[fam[i], fam_arr]) - 1) if i in fam else 0
                for i in fam
            ], dtype=np.int32)
            # Actually let's compute properly
            contribs = np.zeros(m, dtype=np.int32)
            for i in range(m):
                contribs[i] = int(np.sum(comp[fam[i], fam_arr])) - 1  # exclude self (comp[i][i]=1)
            
            # Precompute sum_comp for all masks not in family
            fam_set = set(fam)
            all_masks = np.arange(total_masks, dtype=np.int32)
            not_in_fam = np.array([s for s in all_masks if s not in fam_set], dtype=np.int32)
            
            if len(not_in_fam) == 0:
                break
            
            # sum_comp[s] = sum of comp[s][fam[j]] for all j in family
            # Use matrix-vector product
            sum_comps = np.array([
                int(np.sum(comp[s, fam_arr]))
                for s in not_in_fam
            ], dtype=np.int32)
            
            # Find best swap: maximize (sum_comp[s] - comp[s][fam[i]] - contribs[i])
            best_delta = 0
            best_i = -1
            best_s = -1
            
            for i in range(m):
                fi = fam[i]
                # val = sum_comp[s] - comp[s][fi]
                # We need comp[not_in_fam, fi]
                comp_col = comp[not_in_fam, fi]
                vals = sum_comps - comp_col
                max_idx = int(np.argmax(vals))
                max_val = int(vals[max_idx])
                delta = max_val - int(contribs[i])
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_s = int(not_in_fam[max_idx])
            
            if best_delta > 0 and best_i != -1 and best_s != -1:
                fam[best_i] = best_s
                improved = True
                current_score = count_comparables(fam)
            
            if not improved:
                break
        
        return fam
    
    # Run local search with multiple random restarts
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    best_family = local_search(family, deadline)
    best_score = count_comparables(best_family)
    
    # Try a few random restarts if time permits
    import random
    rng = random.Random(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Create a random-ish initial family
        random_fam = []
        random_used = set()
        # Start with some random masks
        for _ in range(m):
            mask = rng.randint(0, total_masks - 1)
            while mask in random_used:
                mask = rng.randint(0, total_masks - 1)
            random_fam.append(mask)
            random_used.add(mask)
        
        # Quick greedy improvement
        for _ in range(3):
            for i in range(m):
                best_gain = count_comparables(random_fam)
                fi = random_fam[i]
                for s in range(total_masks):
                    if s in random_used:
                        continue
                    random_fam[i] = s
                    new_gain = count_comparables(random_fam)
                    if new_gain > best_gain:
                        best_gain = new_gain
                    else:
                        random_fam[i] = fi
        
        improved_fam = local_search(random_fam, deadline)
        score = count_comparables(improved_fam)
        if score > best_score:
            best_score = score
            best_family = improved_fam
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
