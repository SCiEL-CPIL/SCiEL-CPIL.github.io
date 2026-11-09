import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic, seed):
    """
    Run ACO. Returns the best total prize found.
    """
    n = len(prizes)
    rng = np.random.RandomState(seed)
    
    pheromone = np.ones((n, n), dtype=float)
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    
    dist_to_origin = dist[:, 0]
    
    best_overall = 0.0
    
    for it in range(30):
        best_iter = 0.0
        
        for ant in range(20):
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                feasible_mask = (~visited_mask) & (length + dist[current, :] + dist_to_origin <= max_len)
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                probs = pheromone[current, feasible_indices] * heur[current, feasible_indices]
                s = probs.sum()
                if s <= 0:
                    break
                
                probs = probs / s
                idx = int(rng.choice(len(feasible_indices), p=probs))
                j = int(feasible_indices[idx])
                
                length += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j
            
            if total_prize > best_iter:
                best_iter = total_prize
            if total_prize > best_overall:
                best_overall = total_prize
            
        pheromone = pheromone * 0.9 + best_iter / 10.0
        
    return best_overall

def _make_heuristic(prizes, dist, max_len, params):
    """
    Budget-aware expected utility:
    H[i,j] = (prize[j] + p_add)^p_exp / (d[i,j] + ret_weight * d[j,0] + eps)^d_exp
             * exp(-lambda * (d[i,j] + d[j,0]) / max_len)
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    ret = dist[:, 0][None, :].astype(float)
    
    a = params.get('d_exp', 1.0)
    b = params.get('ret_weight', 1.0)
    c = params.get('p_exp', 1.0)
    k = params.get('p_add', 0.0)
    lam = params.get('lambda', 1.0)
    
    total_dist = d + b * ret
    denom_base = total_dist + eps
    denom = denom_base ** a
    numer = (p + k) ** c
    
    # Budget penalty: penalize moves that consume a large fraction of max_len
    budget_frac = total_dist / max_len
    budget_penalty = np.exp(-lam * budget_frac)
    
    H = numer * budget_penalty / denom
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _seed_heuristic(prizes, dist):
    H = (prizes[None, :] + 1e-9) / (dist + 1e-9)
    return np.where(np.isfinite(H) & (H > 0), H, 1e-9)

def heuristics(prizes, distance_matrix, max_len):
    try:
        start = time.time()
        deadline = start + 295.0
        
        prizes = np.asarray(prizes, dtype=float)
        dist = np.asarray(distance_matrix, dtype=float)
        n = len(prizes)
        
        # Phase 1: Coarse search over budget-aware heuristic parameters
        # Focus on lambda (budget penalty strength), d_exp, ret_weight, p_exp
        coarse_params = [
            {'d_exp': 1.0, 'ret_weight': 1.0, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 1.0},
            {'d_exp': 1.0, 'ret_weight': 1.0, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 2.0},
            {'d_exp': 1.0, 'ret_weight': 1.0, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 3.0},
            {'d_exp': 1.5, 'ret_weight': 1.0, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 2.0},
            {'d_exp': 1.0, 'ret_weight': 1.5, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 2.0},
            {'d_exp': 1.0, 'ret_weight': 0.5, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 2.0},
            {'d_exp': 1.0, 'ret_weight': 1.0, 'p_exp': 1.5, 'p_add': 0.0, 'lambda': 2.0},
            {'d_exp': 1.5, 'ret_weight': 1.5, 'p_exp': 1.0, 'p_add': 0.0, 'lambda': 2.5},
        ]
        
        best_score = 0.0
        best_params = None
        best_H = None
        
        for params in coarse_params:
            if time.time() > deadline - 40.0:
                break
            
            H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            for s in range(3):
                if time.time() > deadline - 40.0:
                    break
                score = _run_aco(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_H = H
        
        if best_params is None:
            return _seed_heuristic(prizes, dist)
        
        # Phase 2: Refinement around best coarse params
        refine_params_list = []
        
        base_d = best_params['d_exp']
        base_r = best_params['ret_weight']
        base_p = best_params['p_exp']
        base_a = best_params['p_add']
        base_l = best_params['lambda']
        
        # Perturb lambda
        for dl in [-0.5, 0.5]:
            refine_params_list.append({
                'd_exp': base_d,
                'ret_weight': base_r,
                'p_exp': base_p,
                'p_add': base_a,
                'lambda': max(0.1, base_l + dl)
            })
        
        # Perturb d_exp
        for dd in [-0.3, 0.3]:
            refine_params_list.append({
                'd_exp': max(0.3, base_d + dd),
                'ret_weight': base_r,
                'p_exp': base_p,
                'p_add': base_a,
                'lambda': base_l
            })
        
        # Perturb ret_weight
        for dr in [-0.3, 0.3]:
            refine_params_list.append({
                'd_exp': base_d,
                'ret_weight': max(0.1, base_r + dr),
                'p_exp': base_p,
                'p_add': base_a,
                'lambda': base_l
            })
        
        # Perturb p_exp
        for dp in [-0.3, 0.3]:
            refine_params_list.append({
                'd_exp': base_d,
                'ret_weight': base_r,
                'p_exp': max(0.3, base_p + dp),
                'p_add': base_a,
                'lambda': base_l
            })
        
        # Combined small perturbations
        refine_params_list.append({
            'd_exp': base_d + 0.2,
            'ret_weight': base_r + 0.2,
            'p_exp': base_p,
            'p_add': base_a,
            'lambda': base_l + 0.3
        })
        refine_params_list.append({
            'd_exp': base_d - 0.2,
            'ret_weight': base_r - 0.2,
            'p_exp': base_p + 0.2,
            'p_add': base_a,
            'lambda': base_l - 0.3
        })
        
        for params in refine_params_list:
            if time.time() > deadline - 20.0:
                break
            
            H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            for s in range(3):
                if time.time() > deadline - 20.0:
                    break
                score = _run_aco(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_H = H
        
        if best_H is None:
            return _seed_heuristic(prizes, dist)
            
        H_final = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
        return H_final

    except Exception:
        try:
            prizes = np.asarray(prizes, dtype=float)
            dist = np.asarray(distance_matrix, dtype=float)
            return _seed_heuristic(prizes, dist)
        except Exception:
            try:
                n = len(prizes)
            except Exception:
                n = 50
            return np.ones((n, n))
