"""Heuristic matrix for ant colony optimization of orienteering - lookahead + regret + potential approach."""
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # Safety margin before 330s kill

    N = len(prizes)
    d_j0 = distance_matrix[:, 0].copy()
    prizes_safe = np.maximum(prizes, 1e-9)
    
    # Precompute cost matrix: cost to go from i to j and then back to start
    # C[i,j] = d[i,j] + d[j,0]
    cost_mat = distance_matrix + d_j0[None, :]
    
    # Ensure no zero/negative distances cause issues
    cost_safe = np.maximum(cost_mat, 1e-9)

    def _simulate_aco(eta, seed):
        """Run the fixed ACO protocol and return best prize over all iterations."""
        np.random.seed(seed)
        tau = np.ones((N, N))
        best_prize_global = 0.0
        
        for it in range(30):
            best_prize_iter = 0.0
            
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    remaining = max_len - path_len
                    # Cost to go to j and return: d[current,j] + d[j,0]
                    # But we also need to ensure we can still reach 0 from j
                    # The constraint is: path_len + d[current,j] + d[j,0] <= max_len
                    # So the "effective cost" of moving to j is d[current,j] + d[j,0]
                    eff_cost = distance_matrix[current, :] + d_j0
                    eligible_mask = (~visited) & (eff_cost <= remaining)
                    
                    if not np.any(eligible_mask):
                        break
                    
                    eligible_idx = np.where(eligible_mask)[0]
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += distance_matrix[current, next_node]
                    prize_total += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    def _make_simple_heuristic(alpha=1.0, beta=1.0):
        """Prize / (dij + dj0)^alpha * dj0^-beta style."""
        p = prizes_safe[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        eta = np.power(p, alpha) / np.power(dij + dj0, beta)
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def _make_potential_heuristic(alpha=1.0):
        """Potential-based: prize_j / (dij + dj0)^alpha, emphasizing nodes that are close to start."""
        p = prizes_safe[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        # Nodes closer to start are "cheaper" to visit and return from
        eta = p / np.power(dij + dj0, alpha)
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def _make_lookahead_heuristic(alpha=1.0, gamma=1.0):
        """Lookahead: for each (i,j), estimate the best prize collectible after visiting j.
        This considers the remaining budget after visiting j and what other nodes could be reached."""
        p = prizes_safe[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        eta = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    eta[i, j] = 1e-9
                    continue
                
                cost_to_j = distance_matrix[i, j]
                cost_return = d_j0[j]
                total_cost = cost_to_j + cost_return
                
                # Prize of j
                base_prize = prizes[j]
                
                # Lookahead: what other nodes could we visit after j?
                # Remaining budget after going to j and returning: max_len - total_cost
                # But we don't return yet; we can continue to other nodes
                # Actually, the constraint is path_len + d[i,j] + d[j,0] <= max_len
                # This means we must be able to return to 0 from j
                # So after visiting j, we have max_len - cost_to_j remaining for the rest of the path
                # But we also need d[j,0] <= remaining to be able to finish
                # So effectively, remaining for other nodes = max_len - cost_to_j - d_j0[j]
                
                remaining = max_len - total_cost
                if remaining <= 0:
                    lookahead_prize = 0.0
                else:
                    # Estimate best additional prize from j within remaining budget
                    # Simple heuristic: sum of top prizes of nodes reachable from j within remaining
                    # A node k is reachable from j if d[j,k] + d[k,0] <= remaining
                    lookahead_prize = 0.0
                    for k in range(N):
                        if k == j or k == 0:
                            continue
                        if distance_matrix[j, k] + d_j0[k] <= remaining:
                            lookahead_prize += prizes[k]
                
                # Combine base prize and lookahead
                eta[i, j] = (base_prize + gamma * lookahead_prize) / np.power(total_cost, alpha)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def _make_regret_heuristic(alpha=1.0, beta=1.0):
        """Regret-based: how much prize do we lose by not visiting j?
        For node j, regret = prize_j - max prize of other nodes that are similarly positioned.
        This encourages visiting high-prize nodes that don't have many alternatives."""
        p = prizes_safe[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        eta = np.zeros((N, N))
        
        # For each node j, compute its "regret" relative to other nodes
        # Regret is higher if j has a high prize and few similar alternatives
        node_regret = np.zeros(N)
        for j in range(1, N):  # Skip node 0
            # Find nodes similar to j (similar distance to start)
            similar_mask = np.abs(d_j0 - d_j0[j]) < 0.1
            similar_mask[j] = False
            if np.any(similar_mask):
                max_similar_prize = np.max(prizes[similar_mask])
                regret = max(0, prizes[j] - max_similar_prize)
            else:
                regret = prizes[j]
            node_regret[j] = regret
        
        # Combine regret with prize and cost
        for i in range(N):
            for j in range(N):
                if i == j:
                    eta[i, j] = 1e-9
                    continue
                total_cost = distance_matrix[i, j] + d_j0[j]
                if total_cost <= 0:
                    total_cost = 1e-9
                eta[i, j] = (prizes[j] + node_regret[j]) / np.power(total_cost, alpha)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def _make_density_heuristic(alpha=1.0):
        """Density-based: prize density = prize / (dij + dj0), but also consider local density of prizes."""
        p = prizes_safe[None, :]
        dij = np.maximum(distance_matrix, 1e-9)
        dj0 = np.maximum(d_j0, 1e-9)[None, :]
        
        eta = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    eta[i, j] = 1e-9
                    continue
                total_cost = distance_matrix[i, j] + d_j0[j]
                if total_cost <= 0:
                    total_cost = 1e-9
                
                # Prize density
                density = prizes[j] / total_cost
                
                # Local density: sum of prizes of nodes near j
                local_prize = 0.0
                for k in range(1, N):
                    if k != j and distance_matrix[j, k] < 0.3:
                        local_prize += prizes[k]
                
                eta[i, j] = density * (1.0 + 0.5 * local_prize)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Generate candidate heuristics
    candidates = []
    
    # Simple heuristics
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for beta in [0.5, 0.75, 1.0, 1.25, 1.5]:
            candidates.append(('simple', alpha, beta))
    
    # Potential heuristics
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        candidates.append(('potential', alpha, 1.0))
    
    # Lookahead heuristics (expensive, so fewer)
    for alpha in [0.5, 1.0, 1.5]:
        for gamma in [0.0, 0.5, 1.0]:
            candidates.append(('lookahead', alpha, gamma))
    
    # Regret heuristics
    for alpha in [0.5, 1.0, 1.5]:
        candidates.append(('regret', alpha, 1.0))
    
    # Density heuristics
    for alpha in [0.5, 1.0, 1.5]:
        candidates.append(('density', alpha, 1.0))

    def _get_heuristic(name, alpha, beta):
        if name == 'simple':
            return _make_simple_heuristic(alpha, beta)
        elif name == 'potential':
            return _make_potential_heuristic(alpha)
        elif name == 'lookahead':
            return _make_lookahead_heuristic(alpha, beta)
        elif name == 'regret':
            return _make_regret_heuristic(alpha, beta)
        elif name == 'density':
            return _make_density_heuristic(alpha)
        else:
            return _make_simple_heuristic(1.0, 1.0)

    # Phase 1: Quick screen with 1 seed
    quick_results = []
    for name, alpha, beta in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        eta = _get_heuristic(name, alpha, beta)
        if not np.all(np.isfinite(eta)) or eta.shape != (N, N):
            continue
        s = _simulate_aco(eta, 0)
        quick_results.append((s, name, alpha, beta, eta))
    
    if not quick_results:
        # Fallback
        eta = _make_simple_heuristic(1.0, 1.0)
        return eta
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with more seeds
    top_k = min(8, len(quick_results))
    best_avg = -1
    best_eta = quick_results[0][4]
    
    for s_quick, name, alpha, beta, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 15:
                break
            s = _simulate_aco(eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Phase 3: Local refinement around best
    if time.time() - start_time < time_limit - 30:
        best_entry = quick_results[0]
        bname, balpha, bbeta = best_entry[1], best_entry[2], best_entry[3]
        
        # Generate nearby candidates
        nearby = []
        for da in [-0.25, -0.1, 0.1, 0.25]:
            for db in [-0.25, -0.1, 0.1, 0.25]:
                na = balpha + da
                nb = bbeta + db
                if na > 0.1 and nb > 0.1:
                    nearby.append((bname, na, nb))
        
        seen = set()
        for name, alpha, beta in nearby:
            key = (name, round(alpha, 3), round(beta, 3))
            if key in seen:
                continue
            seen.add(key)
            
            if time.time() - start_time > time_limit - 10:
                break
            
            eta = _get_heuristic(name, alpha, beta)
            if not np.all(np.isfinite(eta)) or eta.shape != (N, N):
                continue
            s = _simulate_aco(eta, 0)
            if s > best_avg * 0.95:
                scores = [s]
                for seed in [1, 2]:
                    if time.time() - start_time > time_limit - 8:
                        break
                    s2 = _simulate_aco(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_eta = eta
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = _make_simple_heuristic(1.0, 1.0)
    
    return best_eta
