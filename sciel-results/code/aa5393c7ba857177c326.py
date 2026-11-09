import numpy as np
import time

def _run_aco_vectorized(prizes, dist, max_len, heuristic, seed):
    """
    Run ACO with vectorized feasibility and probability checks.
    Returns the best total prize found.
    """
    n = len(prizes)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix
    pheromone = np.ones((n, n), dtype=float)
    # Heuristic matrix, ensuring positive values
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pre-compute distances to origin for quick feasibility checks
    # dist_to_origin[j] = dist[j, 0]
    dist_to_origin = dist[:, 0]
    
    best_overall = 0.0
    
    for it in range(30):
        best_iter = 0.0
        
        for ant in range(20):
            # State
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                # Find feasible nodes
                # Condition: not visited AND length + dist[current, j] + dist[j, 0] <= max_len
                # dist[current, :] gives distance from current to all j
                # dist_to_origin gives distance from all j to 0
                # feasible_mask[j] = (not visited_mask[j]) AND (length + dist[current, j] + dist_to_origin[j] <= max_len)
                
                # Vectorized feasibility check
                # Note: dist[current, :] is a vector of size n
                # dist_to_origin is a vector of size n
                # We need to mask out visited nodes
                feasible_mask = (~visited_mask) & (length + dist[current, :] + dist_to_origin <= max_len)
                
                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                # Calculate probabilities
                # p[j] = pheromone[current, j] * heur[current, j]
                # We only care about feasible_indices
                probs = pheromone[current, feasible_indices] * heur[current, feasible_indices]
                
                s = probs.sum()
                if s <= 0:
                    break
                
                probs = probs / s
                
                # Select next node
                idx = int(rng.choice(len(feasible_indices), p=probs))
                j = int(feasible_indices[idx])
                
                # Update state
                length += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j
            
        if total_prize > best_iter:
            best_iter = total_prize
        if total_prize > best_overall:
            best_overall = total_prize
            
        # Pheromone update
        pheromone = pheromone * 0.9 + best_iter / 10.0
        
    return best_overall

def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix based on parameters.
    H[i, j] = (prize[j] + p_add)^p_exp / (dist[i, j] + ret_weight * dist[j, 0] + eps)^d_exp
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    ret = dist[:, 0][None, :].astype(float) # Distance from j to 0, broadcasted to rows
    
    a = params.get('d_exp', 1.0)
    b = params.get('ret_weight', 0.0)
    c = params.get('p_exp', 1.0)
    k = params.get('p_add', 0.0)
    
    # Denominator: (d + b * ret + eps)^a
    denom_base = d + b * ret + eps
    denom = denom_base ** a
    
    # Numerator: (p + k)^c
    numer = (p + k) ** c
    
    H = numer / denom
    
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _seed_heuristic(prizes, dist):
    H = (prizes[None, :] + 1e-9) / (dist + 1e-9)
    return np.where(np.isfinite(H) & (H > 0), H, 1e-9)

def heuristics(prizes, distance_matrix, max_len):
    try:
        start = time.time()
        deadline = start + 320.0 # 10s buffer
        
        prizes = np.asarray(prizes, dtype=float)
        dist = np.asarray(distance_matrix, dtype=float)
        n = len(prizes)
        
        # Structural change: Use a finer grid search over the parameter space
        # The previous approach used a coarse grid of 10 points followed by local refinement.
        # This new approach uses a more systematic grid search with finer resolution,
        # specifically for d_exp and p_exp, which were identified as key parameters.
        # We reduce the number of seeds per candidate to fit more candidates in the time budget.
        
        # Define the grid
        # d_exp: 8 values from 0.5 to 2.5
        d_exp_values = np.linspace(0.5, 2.5, 8)
        # p_exp: 8 values from 0.5 to 2.5
        p_exp_values = np.linspace(0.5, 2.5, 8)
        # ret_weight: 4 values from 0.0 to 1.0
        ret_weight_values = np.linspace(0.0, 1.0, 4)
        # p_add: 3 values from 0.0 to 0.2
        p_add_values = np.linspace(0.0, 0.2, 3)
        
        # Total candidates: 8 * 8 * 4 * 3 = 768
        # This is too many for a full grid with multiple seeds.
        # We will use a reduced grid or a smarter sampling strategy.
        
        # Let's try a reduced grid:
        # d_exp: 5 values
        # p_exp: 5 values
        # ret_weight: 3 values
        # p_add: 2 values
        # Total: 5 * 5 * 3 * 2 = 150 candidates.
        # With 2 seeds each, that's 300 ACO runs.
        # Each ACO run is ~30 * 20 * ~5 steps = 3000 steps.
        # 300 * 3000 = 900,000 steps.
        # Vectorized, this should be fast enough.
        
        d_exp_values = np.linspace(0.5, 2.5, 5)
        p_exp_values = np.linspace(0.5, 2.5, 5)
        ret_weight_values = np.linspace(0.0, 1.0, 3)
        p_add_values = np.linspace(0.0, 0.2, 2)
        
        best_score = 0.0
        best_params = None
        best_H = None
        
        # Phase 1: Grid Search
        for d_exp in d_exp_values:
            if time.time() > deadline - 30.0:
                break
            for p_exp in p_exp_values:
                if time.time() > deadline - 30.0:
                    break
                for ret_weight in ret_weight_values:
                    if time.time() > deadline - 30.0:
                        break
                    for p_add in p_add_values:
                        if time.time() > deadline - 30.0:
                            break
                        
                        params = {
                            'd_exp': d_exp,
                            'ret_weight': ret_weight,
                            'p_exp': p_exp,
                            'p_add': p_add
                        }
                        
                        H = _make_heuristic(prizes, dist, max_len, params)
                        scores = []
                        # Use 2 seeds for grid search
                        for s in range(2):
                            if time.time() > deadline - 30.0:
                                break
                            score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                            scores.append(score)
                        
                        if scores:
                            mean_score = np.mean(scores)
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params.copy()
                                best_H = H
        
        if best_params is None:
            # Fallback if no time was left or error
            return _seed_heuristic(prizes, dist)
        
        # Phase 2: Local Refinement
        # Perform a local search around best_params
        # Perturb each parameter slightly and test
        
        base_d = best_params['d_exp']
        base_r = best_params['ret_weight']
        base_p = best_params['p_exp']
        base_a = best_params['p_add']
        
        # Generate neighbors with smaller steps
        delta_d = 0.1
        delta_r = 0.05
        delta_p = 0.1
        delta_a = 0.02
        
        refine_params_list = []
        for dd in [-delta_d, 0, delta_d]:
            for dr in [-delta_r, 0, delta_r]:
                for dp in [-delta_p, 0, delta_p]:
                    for da in [-delta_a, 0, delta_a]:
                        refine_params_list.append({
                            'd_exp': max(0.1, base_d + dd),
                            'ret_weight': max(0.0, min(1.0, base_r + dr)),
                            'p_exp': max(0.1, base_p + dp),
                            'p_add': max(0.0, min(0.5, base_a + da))
                        })
        
        # Remove duplicates and sort by distance from best_params to prioritize closer points
        # But for simplicity, we'll just iterate.
        for params in refine_params_list:
            if time.time() > deadline - 20.0:
                break
            
            H = _make_heuristic(prizes, dist, max_len, params)
            scores = []
            # Use 2 seeds for refinement
            for s in range(2):
                if time.time() > deadline - 20.0:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_H = H
        
        # Phase 3: Final Polish
        # Ensure the final H is valid
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
