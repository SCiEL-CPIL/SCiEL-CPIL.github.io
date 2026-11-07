"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco_fast(prizes, distance_matrix, max_len, heuristic, seed):
    """
    Fast simulation of the ACO colony. Returns best prize collected by any ant.
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
    
    for it in range(n_iter):
        iter_best = 0.0
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                remaining = max_len - length
                if remaining <= 0:
                    break
                
                # Need d[current, j] + d_j0[j] <= remaining
                eligible_mask = (distance_matrix[current] + d_j0 <= remaining) & (~visited)
                eligible_mask[0] = False  # node 0 is start
                
                if not np.any(eligible_mask):
                    break
                
                eligible = np.where(eligible_mask)[0]
                
                # Compute attractiveness
                att = pheromone[current, eligible] * h[current, eligible]
                
                total_att = att.sum()
                if total_att <= 0 or not np.isfinite(total_att):
                    break
                
                # Normalize and sample
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
                
                length += distance_matrix[current, next_j]
                total_prize += prizes[next_j]
                visited[next_j] = True
                current = next_j
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        if iter_best > best_total:
            best_total = iter_best
        
        # Update pheromone
        pheromone = pheromone * evaporation
        pheromone += deposition * iter_best
    
    return best_total


def _greedy_prize_path(prizes, distance_matrix, max_len, heuristic, seed=None):
    """
    Greedy path: at each step pick the eligible node maximizing heuristic[i,j].
    Returns total prize.
    """
    n = len(prizes)
    if n <= 1:
        return 0.0
    
    h = np.maximum(heuristic, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    length = 0.0
    total_prize = 0.0
    
    while True:
        remaining = max_len - length
        if remaining <= 0:
            break
        
        eligible_mask = (distance_matrix[current] + d_j0 <= remaining) & (~visited)
        eligible_mask[0] = False
        
        if not np.any(eligible_mask):
            break
        
        eligible = np.where(eligible_mask)[0]
        att = h[current, eligible]
        best_idx = np.argmax(att)
        next_j = eligible[best_idx]
        
        length += distance_matrix[current, next_j]
        total_prize += prizes[next_j]
        visited[next_j] = True
        current = next_j
    
    return total_prize


def _greedy_prize_path_multi(prizes, distance_matrix, max_len, heuristic, n_starts=5):
    """
    Try greedy from multiple starting perspectives (though all start at 0,
    we can vary the tie-breaking by adding small noise).
    """
    n = len(prizes)
    if n <= 1:
        return 0.0
    
    best = 0.0
    for i in range(n_starts):
        np.random.seed(1000 + i)
        noise = np.random.random((n, n)) * 1e-6
        h_noisy = heuristic + noise
        p = _greedy_prize_path(prizes, distance_matrix, max_len, h_noisy)
        if p > best:
            best = p
    return best


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic using a prize-to-cost ratio model
    with additional structural terms. We search over parameters using a
    combination of greedy evaluation (fast) and ACO simulation (accurate).
    
    h[i,j] = (prize[j])^alpha / (d[i,j] + d[j,0])^beta * (max_len / (d[i,j]+d[j,0]))^gamma
             * (1 + delta * prize[j] / d_j0[j])^eta
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    start_time = time.time()
    time_limit = 320.0  # Safety margin before 330s kill
    
    d_j0 = distance_matrix[:, 0]
    eps = 1e-9
    
    prize_term = np.maximum(prizes, eps)
    
    best_score = -1.0
    best_h = None
    best_params = None
    
    def build_heuristic(alpha, beta, gamma, eta=0.0, delta=0.0):
        """
        Build heuristic with additional structural term.
        """
        node_val = prize_term ** alpha * (max_len ** gamma)
        
        cost = distance_matrix + d_j0[None, :]
        cost = np.maximum(cost, eps)
        
        p = beta + gamma
        h = node_val[None, :] / (cost ** p)
        
        # Additional structural term: bonus for high prize relative to return cost
        if eta > 0 and delta > 0:
            # prize[j] / d_j0[j] as a "value density" at node j
            node_density = prize_term / np.maximum(d_j0, eps)
            # Normalize to [0,1]
            nd_max = node_density.max()
            if nd_max > 0:
                node_density = node_density / nd_max
            h = h * (1.0 + delta * node_density[None, :] ** eta)
        
        h[:, 0] = eps
        h = np.maximum(h, eps)
        return h
    
    def evaluate_params_fast(alpha, beta, gamma, eta=0.0, delta=0.0):
        """Fast evaluation using greedy path."""
        h = build_heuristic(alpha, beta, gamma, eta, delta)
        return _greedy_prize_path(prizes, distance_matrix, max_len, h)
    
    def evaluate_params_aco(alpha, beta, gamma, eta=0.0, delta=0.0, seed=42):
        """Accurate evaluation using ACO simulation."""
        h = build_heuristic(alpha, beta, gamma, eta, delta)
        return _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=seed)
    
    # Phase 1: Random search over extended parameter space (fast greedy eval)
    np.random.seed(12345)
    
    n_random = 200
    for _ in range(n_random):
        if time.time() - start_time > time_limit:
            break
        
        alpha = np.random.uniform(0.5, 3.5)
        beta = np.random.uniform(0.5, 4.5)
        gamma = np.random.uniform(0.0, 3.5)
        eta = np.random.uniform(0.0, 2.0)
        delta = np.random.uniform(0.0, 1.0)
        
        score = evaluate_params_fast(alpha, beta, gamma, eta, delta)
        
        if score > best_score:
            best_score = score
            best_params = (alpha, beta, gamma, eta, delta)
    
    # Phase 2: Local refinement around best params (fast greedy eval)
    if best_params is not None:
        base_alpha, base_beta, base_gamma, base_eta, base_delta = best_params
        
        # Coarse grid refinement
        for da in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            for db in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
                for dg in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
                    for de in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                        for dd in [-0.1, 0.0, 0.1]:
                            if time.time() - start_time > time_limit:
                                break
                            
                            a = max(0.1, base_alpha + da)
                            b = max(0.1, base_beta + db)
                            g = max(0.0, base_gamma + dg)
                            e = max(0.0, base_eta + de)
                            dl = max(0.0, base_delta + dd)
                            
                            score = evaluate_params_fast(a, b, g, e, dl)
                            
                            if score > best_score:
                                best_score = score
                                best_params = (a, b, g, e, dl)
        
        # Fine grid refinement
        base_alpha, base_beta, base_gamma, base_eta, base_delta = best_params
        for da in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            for db in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                for dg in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    for de in [-0.05, 0.0, 0.05]:
                        for dd in [-0.05, 0.0, 0.05]:
                            if time.time() - start_time > time_limit:
                                break
                            
                            a = max(0.1, base_alpha + da)
                            b = max(0.1, base_beta + db)
                            g = max(0.0, base_gamma + dg)
                            e = max(0.0, base_eta + de)
                            dl = max(0.0, base_delta + dd)
                            
                            score = evaluate_params_fast(a, b, g, e, dl)
                            
                            if score > best_score:
                                best_score = score
                                best_params = (a, b, g, e, dl)
    
    # Phase 3: Validate top candidates with ACO simulation
    if best_params is not None and time.time() - start_time < time_limit * 0.8:
        a, b, g, e, dl = best_params
        h = build_heuristic(a, b, g, e, dl)
        
        # Evaluate with multiple ACO seeds to get a robust score
        aco_scores = []
        for seed in [42, 7, 99, 123, 555]:
            if time.time() - start_time > time_limit:
                break
            s = _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=seed)
            aco_scores.append(s)
        
        aco_mean = np.mean(aco_scores) if aco_scores else 0.0
        aco_max = np.max(aco_scores) if aco_scores else 0.0
        
        # Try a few nearby parameter sets and pick the one with best ACO mean
        candidates = [(a, b, g, e, dl)]
        
        # Generate a few nearby candidates
        for da in [-0.1, 0.1]:
            for db in [-0.1, 0.1]:
                for dg in [-0.1, 0.1]:
                    ca = max(0.1, a + da)
                    cb = max(0.1, b + db)
                    cg = max(0.0, g + dg)
                    candidates.append((ca, cb, cg, e, dl))
        
        best_aco_mean = -1.0
        best_aco_params = None
        best_aco_h = None
        
        for (ca, cb, cg, ce, cd) in candidates[:8]:  # Limit to 8 to save time
            if time.time() - start_time > time_limit:
                break
            
            h_cand = build_heuristic(ca, cb, cg, ce, cd)
            
            # Quick ACO eval with 2 seeds
            s1 = _simulate_aco_fast(prizes, distance_matrix, max_len, h_cand, seed=42)
            s2 = _simulate_aco_fast(prizes, distance_matrix, max_len, h_cand, seed=7)
            mean_s = (s1 + s2) / 2.0
            
            if mean_s > best_aco_mean:
                best_aco_mean = mean_s
                best_aco_params = (ca, cb, cg, ce, cd)
                best_aco_h = h_cand
        
        if best_aco_h is not None:
            best_h = best_aco_h
            best_params = best_aco_params
    
    # Fallback
    if best_h is None:
        if best_params is not None:
            a, b, g, e, dl = best_params
            best_h = build_heuristic(a, b, g, e, dl)
        else:
            best_h = build_heuristic(1.0, 1.5, 0.5, 0.5, 0.3)
    
    # Final validation
    if time.time() - start_time < time_limit * 0.9:
        for test_seed in [7, 99, 123]:
            if time.time() - start_time > time_limit:
                break
            _ = _simulate_aco_fast(prizes, distance_matrix, max_len, best_h, seed=test_seed)
    
    return best_h
# EVOLVE-BLOCK-END
