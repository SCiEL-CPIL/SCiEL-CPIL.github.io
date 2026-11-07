import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

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

    def greedy_init():
        fam = []
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        gain = np.zeros(total_masks, dtype=np.int32)
        
        for _ in range(m):
            gain_masked = gain.copy()
            gain_masked[in_fam_mask] = -1
            v = int(np.argmax(gain_masked))
            
            fam.append(v)
            in_fam_mask[v] = True
            
            gain += comp[:, v]
            
        return fam

    def local_search(fam, rng):
        fam = list(fam)
        deg = calc_deg(fam)
        score = get_score(fam, deg)
        
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam_mask[f] = True
        not_in_fam_mask = ~in_fam_mask
        
        T = 20.0
        cooling_rate = 0.99
        max_no_improve = 500
        no_improve_count = 0
        best_score_so_far = score
        best_fam_so_far = list(fam)
        
        fam_arr = np.array(fam, dtype=np.int32)
        
        while time.time() < deadline - 2:
            improved = False
            
            # Update candidate list
            deg_candidates = deg.copy()
            deg_candidates[in_fam_mask] = -np.inf
            
            # Pick top K candidates
            K = 2000
            if K > total_masks: K = total_masks
            top_indices = np.argpartition(deg_candidates, -K)[-K:]
            
            # Vectorized 1-swap evaluation
            # Filter out any that might be in_fam (just in case, though -inf should handle it)
            valid_top = [i for i in top_indices if not in_fam_mask[i]]
            if len(valid_top) == 0:
                break
                
            valid_top_arr = np.array(valid_top, dtype=np.int32)
            
            comp_top_fam = comp[valid_top_arr, fam_arr] # (K_valid, m)
            deg_fam = deg[fam_arr] # (m,)
            
            costs = deg_fam[None, :] + comp_top_fam # (K_valid, m)
            min_costs = np.min(costs, axis=1) # (K_valid,)
            deltas = deg[valid_top_arr] - min_costs # (K_valid,)
            
            max_idx = np.argmax(deltas)
            best_delta = deltas[max_idx]
            
            if best_delta > 0:
                best_new = valid_top_arr[max_idx]
                min_cost_idx = np.argmin(costs[max_idx])
                best_old = fam_arr[min_cost_idx]
                
                # Update deg
                deg -= comp[:, best_old]
                deg += comp[:, best_new]
                
                # Update fam
                idx = np.where(fam_arr == best_old)[0][0]
                fam[idx] = best_new
                fam_arr[idx] = best_new
                
                in_fam_mask[best_old] = False
                in_fam_mask[best_new] = True
                not_in_fam_mask[best_old] = True
                not_in_fam_mask[best_new] = False
                
                score += int(best_delta)
                improved = True
                no_improve_count = 0
                
                if score > best_score_so_far:
                    best_score_so_far = score
                    best_fam_so_far = list(fam)
                    
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                
                # Simulated Annealing: accept worse move
                if T > 0.5 and len(valid_top_arr) > 0:
                    rand_idx = rng.integers(0, len(valid_top_arr))
                    m_new = valid_top_arr[rand_idx]
                    
                    # Pick random m_old
                    m_old_idx = rng.integers(0, m)
                    m_old = fam_arr[m_old_idx]
                    
                    c = 1 if comp[m_new, m_old] else 0
                    delta = deg[m_new] - deg[m_old] - c
                    
                    if delta >= 0 or np.random.random() < np.exp(delta / T):
                        deg -= comp[:, m_old]
                        deg += comp[:, m_new]
                        
                        fam[m_old_idx] = m_new
                        fam_arr[m_old_idx] = m_new
                        
                        in_fam_mask[m_old] = False
                        in_fam_mask[m_new] = True
                        not_in_fam_mask[m_old] = True
                        not_in_fam_mask[m_new] = False
                        
                        score += int(delta)
                        if score > best_score_so_far:
                            best_score_so_far = score
                            best_fam_so_far = list(fam)
                        
                        no_improve_count = 0
                        improved = True
                
                T *= cooling_rate
            
        return best_fam_so_far, best_score_so_far

    best_score = 0
    best_fam = None
    
    rng = np.random.default_rng(42)
    
    # Run greedy init
    fam = greedy_init()
    cur_fam, cur_score = local_search(fam, rng)
    
    if cur_score > best_score:
        best_score = cur_score
        best_fam = cur_fam
        
    # Random restarts
    while time.time() < deadline - 2:
        fam = rng.choice(total_masks, size=m, replace=False).tolist()
        cur_fam, cur_score = local_search(fam, rng)
        
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
