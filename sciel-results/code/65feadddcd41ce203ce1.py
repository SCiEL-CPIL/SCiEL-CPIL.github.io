import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Precompute comparability matrix
    i_vals = np.arange(total_masks, dtype=np.int32)[:, None]
    j_vals = np.arange(total_masks, dtype=np.int32)[None, :]
    
    cond1 = (i_vals & j_vals) == i_vals
    cond2 = (i_vals & j_vals) == j_vals
    comp = cond1 | cond2
    np.fill_diagonal(comp, False)
    
    def calc_deg(fam):
        fam_arr = np.array(fam, dtype=np.int32)
        deg = comp[:, fam_arr].sum(axis=1).astype(np.int32)
        return deg

    def get_score(fam, deg):
        return int(np.sum(deg) / 2)

    def local_search(fam):
        fam = list(fam)
        deg = calc_deg(fam)
        score = get_score(fam, deg)
        fam_set = set(fam)
        
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam_mask[f] = True
        not_in_fam_mask = ~in_fam_mask
        
        T = 10.0
        cooling_rate = 0.95
        max_no_improve = 50
        no_improve_count = 0
        best_score_so_far = score
        best_fam_so_far = list(fam)
        
        while time.time() < deadline - 3:
            improved = False
            best_delta = 0
            best_old = None
            best_new = None
            
            deg_candidates = deg.copy()
            deg_candidates[in_fam_mask] = -1
            
            K = 100
            if K > total_masks: K = total_masks
            top_indices = np.argpartition(deg_candidates, -K)[-K:]
            
            for m_new in top_indices:
                if not_in_fam_mask[m_new]:
                    for m_old in fam:
                        c = 1 if comp[m_new, m_old] else 0
                        delta = deg[m_new] - c - deg[m_old]
                        if delta > best_delta:
                            best_delta = delta
                            best_old = m_old
                            best_new = m_new
            
            if best_delta > 0:
                deg -= comp[:, best_old]
                deg += comp[:, best_new]
                
                fam[fam.index(best_old)] = best_new
                fam_set.remove(best_old)
                fam_set.add(best_new)
                
                in_fam_mask[best_old] = False
                in_fam_mask[best_new] = True
                not_in_fam_mask[best_old] = True
                not_in_fam_mask[best_new] = False
                
                score += best_delta
                improved = True
                no_improve_count = 0
                
                if score > best_score_so_far:
                    best_score_so_far = score
                    best_fam_so_far = list(fam)
                    
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                
                if T > 1.0:
                    idxs = np.random.choice(m, size=2, replace=False)
                    old1 = fam[idxs[0]]
                    old2 = fam[idxs[1]]
                    
                    cand_indices = top_indices[np.random.choice(len(top_indices), size=2, replace=False)]
                    new1 = cand_indices[0]
                    new2 = cand_indices[1]
                    
                    if new1 in fam_set or new2 in fam_set or new1 == new2:
                        T *= cooling_rate
                        continue
                        
                    deg -= comp[:, old1]
                    deg -= comp[:, old2]
                    deg += comp[:, new1]
                    deg += comp[:, new2]
                    
                    fam[idxs[0]] = new1
                    fam[idxs[1]] = new2
                    fam_set.remove(old1)
                    fam_set.remove(old2)
                    fam_set.add(new1)
                    fam_set.add(new2)
                    
                    in_fam_mask[old1] = False
                    in_fam_mask[old2] = False
                    in_fam_mask[new1] = True
                    in_fam_mask[new2] = True
                    not_in_fam_mask[old1] = True
                    not_in_fam_mask[old2] = True
                    not_in_fam_mask[new1] = False
                    not_in_fam_mask[new2] = False
                    
                    score = get_score(fam, deg)
                    if score > best_score_so_far:
                        best_score_so_far = score
                        best_fam_so_far = list(fam)
                    
                    T *= cooling_rate
            T *= cooling_rate
        
        return best_fam_so_far, best_score_so_far

    best_score = 0
    best_fam = None
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 2:
        fam = rng.choice(total_masks, size=m, replace=False).tolist()
        cur_fam, cur_score = local_search(fam)
        
        if cur_score > best_score:
            best_score = cur_score
            best_fam = cur_fam
            
        if best_score > 700:
             break

    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
