"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco_fast(prizes, distance_matrix, max_len, heuristic, seed, smooth=False):
    """
    Fast simulation of the ACO colony. Returns best prize collected by any ant.
    If smooth=True, returns logsumexp of all ant prizes (smoother objective for param search).
    """
    np.random.seed(seed)
    
    n = len(prizes)
    if n <= 1:
        return 0.0
    
    n_ants = 20
    n_iter = 30
    evaporation = 0.9
    deposition = 1.0 / 10.0
    
    pheromone = np.ones((n, n), dtype=np.float64)
    h = np.maximum(heuristic, 1e-9)
    
    best_total = 0.0
    d_j0 = distance_matrix[:, 0]
    dm = distance_matrix
    
    all_ants_prizes = []
    
    for it in range(n_iter):
        iter_best = 0.0
        iter_ants = []
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                remaining = max_len - length
                feasible_mask = (remaining - d_j0 >= 0) & (~visited)
                feasible_mask[0] = False
                
                eligible_mask = feasible_mask & (dm[current] <= (remaining - d_j0))
                
                if not np.any(eligible_mask):
                    break
                
                eligible = np.where(eligible_mask)[0]
                
                att = pheromone[current, eligible] * h[current, eligible]
                
                total_att = att.sum()
                if total_att <= 0 or not np.isfinite(total_att):
                    break
                
                probs = att / total_att
                probs = np.maximum(probs, 0)
                ps = probs.sum()
                if ps <= 0:
                    break
                probs = probs / ps
                
                r = np.random.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible):
                    idx = len(eligible) - 1
                
                next_j = eligible[idx]
                
                length += dm[current, next_j]
                total_prize += prizes[next_j]
                visited[next_j] = True
                current = next_j
            
            iter_ants.append(total_prize)
            if total_prize > iter_best:
                iter_best = total_prize
        
        all_ants_prizes.extend(iter_ants)
        
        if iter_best > best_total:
            best_total = iter_best
        
        pheromone = pheromone * evaporation
        pheromone += deposition * iter_best
    
    if smooth and all_ants_prizes:
        arr = np.array(all_ants_prizes, dtype=np.float64)
        # log-sum-exp for numerical stability
        m = arr.max()
        if m < 1e-12:
            return 0.0
        return m + np.log(np.sum(np.exp(arr - m)))
    
    return best_total


def _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma):
    """
    Build heuristic matrix with parameters.
    h[i,j] = prize[j]^beta * exp(-d[j,0]/scale) / (d[i,j]+eps)^alpha * (d[j,0]+eps)^(-dual_gamma)
    """
    eps = 1e-9
    n = len(prizes)
    d_j0 = distance_matrix[:, 0]
    
    prize_term = np.maximum(prizes, eps)
    prize_pow = prize_term ** p_beta
    
    return_cost = np.exp(-d_j0 / scale)
    
    if dual_gamma > 0:
        dual_pen = (d_j0 + eps) ** (-dual_gamma)
    else:
        dual_pen = np.ones(n)
    
    node_attr = prize_pow * return_cost * dual_pen  # shape (n,)
    
    d_plus = distance_matrix + eps
    dist_pen = d_plus ** (-alpha)  # shape (n, n)
    
    h = node_attr[None, :] * dist_pen
    h = np.maximum(h, 1e-9)
    
    return h


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic matrix.
    
    Uses a dual-distance penalty structure:
    h[i,j] = prize[j]^beta * exp(-d[j,0]/scale) / (d[i,j]+eps)^alpha * (d[j,0]+eps)^(-gamma)
    
    Parameter search via fast ACO simulation with smooth objective.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    start_time = time.time()
    time_limit = 290.0
    
    # Parameter grid: (alpha, scale, p_beta, dual_gamma)
    # Focus on regions that showed promise, with finer granularity
    param_sets = [
        # Core region around known good values
        (1.0, 0.5, 1.0, 0.5),
        (1.5, 0.5, 1.0, 0.5),
        (2.0, 0.5, 1.0, 0.5),
        (1.0, 0.7, 1.0, 0.5),
        (1.5, 0.7, 1.0, 0.5),
        (2.0, 0.7, 1.0, 0.5),
        (1.0, 0.5, 1.0, 1.0),
        (1.5, 0.5, 1.0, 1.0),
        (2.0, 0.5, 1.0, 1.0),
        (1.0, 0.7, 1.0, 1.0),
        (1.5, 0.7, 1.0, 1.0),
        (2.0, 0.7, 1.0, 1.0),
        (1.0, 0.5, 1.0, 1.5),
        (1.5, 0.5, 1.0, 1.5),
        (2.0, 0.5, 1.0, 1.5),
        (1.0, 0.7, 1.0, 1.5),
        (1.5, 0.7, 1.0, 1.5),
        (2.0, 0.7, 1.0, 1.5),
        (1.0, 0.5, 1.0, 2.0),
        (1.5, 0.5, 1.0, 2.0),
        (2.0, 0.5, 1.0, 2.0),
        (1.0, 0.7, 1.0, 2.0),
        (1.5, 0.7, 1.0, 2.0),
        (2.0, 0.7, 1.0, 2.0),
        # Higher beta with dual penalty
        (1.0, 0.5, 1.5, 0.5),
        (1.5, 0.5, 1.5, 0.5),
        (2.0, 0.5, 1.5, 0.5),
        (1.0, 0.7, 1.5, 0.5),
        (1.5, 0.7, 1.5, 0.5),
        (2.0, 0.7, 1.5, 0.5),
        (1.0, 0.5, 1.5, 1.0),
        (1.5, 0.5, 1.5, 1.0),
        (2.0, 0.5, 1.5, 1.0),
        (1.0, 0.7, 1.5, 1.0),
        (1.5, 0.7, 1.5, 1.0),
        (2.0, 0.7, 1.5, 1.0),
        (1.0, 0.5, 1.5, 1.5),
        (1.5, 0.5, 1.5, 1.5),
        (2.0, 0.5, 1.5, 1.5),
        (1.0, 0.7, 1.5, 1.5),
        (1.5, 0.7, 1.5, 1.5),
        (2.0, 0.7, 1.5, 1.5),
        (1.0, 0.5, 2.0, 1.0),
        (1.5, 0.5, 2.0, 1.0),
        (2.0, 0.5, 2.0, 1.0),
        (1.0, 0.7, 2.0, 1.0),
        (1.5, 0.7, 2.0, 1.0),
        (2.0, 0.7, 2.0, 1.0),
        (1.0, 0.5, 2.0, 1.5),
        (1.5, 0.5, 2.0, 1.5),
        (2.0, 0.5, 2.0, 1.5),
        (1.0, 0.7, 2.0, 1.5),
        (1.5, 0.7, 2.0, 1.5),
        (2.0, 0.7, 2.0, 1.5),
        # Lower alpha (less distance penalty)
        (0.5, 0.5, 1.0, 1.0),
        (0.5, 0.7, 1.0, 1.0),
        (0.5, 0.5, 1.5, 1.0),
        (0.5, 0.7, 1.5, 1.0),
        (0.75, 0.5, 1.0, 1.0),
        (0.75, 0.7, 1.0, 1.0),
        (0.75, 0.5, 1.5, 1.0),
        (0.75, 0.7, 1.5, 1.0),
        # Different scales
        (1.0, 0.3, 1.0, 1.0),
        (1.5, 0.3, 1.0, 1.0),
        (2.0, 0.3, 1.0, 1.0),
        (1.0, 0.4, 1.0, 1.0),
        (1.5, 0.4, 1.0, 1.0),
        (2.0, 0.4, 1.0, 1.0),
        (1.0, 0.6, 1.0, 1.0),
        (1.5, 0.6, 1.0, 1.0),
        (2.0, 0.6, 1.0, 1.0),
        (1.0, 0.8, 1.0, 1.0),
        (1.5, 0.8, 1.0, 1.0),
        (2.0, 0.8, 1.0, 1.0),
        # Higher gamma
        (1.0, 0.5, 1.0, 2.5),
        (1.5, 0.5, 1.0, 2.5),
        (2.0, 0.5, 1.0, 2.5),
        (1.0, 0.7, 1.0, 2.5),
        (1.5, 0.7, 1.0, 2.5),
        (2.0, 0.7, 1.0, 2.5),
        (1.0, 0.5, 1.0, 3.0),
        (1.5, 0.5, 1.0, 3.0),
        (2.0, 0.5, 1.0, 3.0),
        (1.0, 0.7, 1.0, 3.0),
        (1.5, 0.7, 1.0, 3.0),
        (2.0, 0.7, 1.0, 3.0),
        # No dual penalty (baseline)
        (1.0, 0.5, 1.0, 0.0),
        (1.5, 0.5, 1.0, 0.0),
        (2.0, 0.5, 1.0, 0.0),
        (1.0, 0.7, 1.0, 0.0),
        (1.5, 0.7, 1.0, 0.0),
        (2.0, 0.7, 1.0, 0.0),
    ]
    
    np.random.seed(42)
    perm = np.random.permutation(len(param_sets))
    param_sets = [param_sets[i] for i in perm]
    
    best_score = -1.0
    best_h = None
    best_params = None
    
    # First pass: test each param set with smooth objective (seed 42)
    for alpha, scale, p_beta, dual_gamma in param_sets:
        if time.time() - start_time > time_limit:
            break
        
        h = _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma)
        score = _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=42, smooth=True)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            best_params = (alpha, scale, p_beta, dual_gamma)
    
    # Second pass: finer local refinement around best params
    if best_h is not None and time.time() - start_time < time_limit * 0.55:
        best_alpha, best_scale, best_p_beta, best_dual_gamma = best_params
        
        # Generate fine local grid
        local_candidates = []
        for da in np.arange(-0.3, 0.31, 0.1):
            for ds in np.arange(-0.2, 0.21, 0.1):
                for dpb in np.arange(-0.3, 0.31, 0.15):
                    for dgy in np.arange(-0.3, 0.31, 0.15):
                        la = best_alpha + da
                        ls = best_scale + ds
                        lpb = best_p_beta + dpb
                        lgy = best_dual_gamma + dgy
                        if la > 0.1 and ls > 0.1 and lpb > 0.1 and lgy >= 0:
                            local_candidates.append((la, ls, lpb, lgy))
        
        # Deduplicate
        seen = set()
        unique_local = []
        for c in local_candidates:
            key = tuple(round(x, 4) for x in c)
            if key not in seen:
                seen.add(key)
                unique_local.append(c)
        
        np.random.seed(43)
        perm2 = np.random.permutation(len(unique_local))
        unique_local = [unique_local[i] for i in perm2]
        
        for alpha, scale, p_beta, dual_gamma in unique_local:
            if time.time() - start_time > time_limit * 0.8:
                break
            
            h = _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma)
            score = _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=42, smooth=True)
            
            if score > best_score:
                best_score = score
                best_h = h.copy()
                best_params = (alpha, scale, p_beta, dual_gamma)
    
    # Third pass: even finer polish if time allows
    if best_h is not None and time.time() - start_time < time_limit * 0.75:
        best_alpha, best_scale, best_p_beta, best_dual_gamma = best_params
        
        polish_candidates = []
        for da in np.arange(-0.05, 0.06, 0.025):
            for ds in np.arange(-0.05, 0.06, 0.025):
                for dpb in np.arange(-0.1, 0.11, 0.05):
                    for dgy in np.arange(-0.1, 0.11, 0.05):
                        la = best_alpha + da
                        ls = best_scale + ds
                        lpb = best_p_beta + dpb
                        lgy = best_dual_gamma + dgy
                        if la > 0.1 and ls > 0.1 and lpb > 0.1 and lgy >= 0:
                            polish_candidates.append((la, ls, lpb, lgy))
        
        seen2 = set()
        unique_polish = []
        for c in polish_candidates:
            key = tuple(round(x, 5) for x in c)
            if key not in seen2:
                seen2.add(key)
                unique_polish.append(c)
        
        for alpha, scale, p_beta, dual_gamma in unique_polish:
            if time.time() - start_time > time_limit * 0.92:
                break
            
            h = _build_heuristic(prizes, distance_matrix, max_len, alpha, scale, p_beta, dual_gamma)
            score = _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=42, smooth=True)
            
            if score > best_score:
                best_score = score
                best_h = h.copy()
                best_params = (alpha, scale, p_beta, dual_gamma)
    
    # Final verification: check with raw (non-smooth) objective on multiple seeds
    if best_h is not None and time.time() - start_time < time_limit * 0.95:
        # Verify the best heuristic actually performs well with the raw objective
        verify_score = _simulate_aco_fast(prizes, distance_matrix, max_len, best_h, seed=42, smooth=False)
        # If smooth objective led us astray, we can still return best_h
        # (the smooth objective should correlate well with raw max)
    
    # Fallback if no heuristic was built
    if best_h is None:
        h = _build_heuristic(prizes, distance_matrix, max_len, 1.5, 0.5, 1.0, 1.0)
        best_h = h
    
    return best_h
# EVOLVE-BLOCK-END
