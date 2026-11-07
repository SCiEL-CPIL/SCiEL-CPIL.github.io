import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, eta, seed):
    """Run the fixed ACO protocol and return best prize over all iterations."""
    np.random.seed(seed)
    N = len(prizes)
    tau = np.ones((N, N))
    best_prize_global = 0.0
    d_j0 = distance_matrix[:, 0]
    
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
                costs = distance_matrix[current, :] + d_j0
                eligible_mask = (~visited) & (costs <= remaining)
                
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

def _make_potential_heuristic(prizes, distance_matrix, max_len, prize_exp=1.0, slack_exp=1.0, cost_exp=0.5):
    """
    Dynamic Potential heuristic: entry (i, j) = prize[j]^prize_exp * slack_norm^slack_exp / (d[i,j] + eps)^cost_exp
    where slack_norm = max(0, max_len - d[i,j] - d[j,0]) / max_len
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    # Cost to go i -> j -> 0
    C = distance_matrix + d_j0[None, :]
    
    # Normalized slack: how much room is left after taking edge i->j and returning
    slack = np.maximum(0.0, max_len - C) / max_len
    slack = np.maximum(slack, 1e-9)
    
    # Distance from i to j (not including return)
    d_ij = np.maximum(distance_matrix, 1e-9)
    
    # Potential = prize^p * slack^s / d_ij^c
    eta = (np.power(prizes_safe[None, :], prize_exp) * 
           np.power(slack, slack_exp) / 
           np.power(d_ij, cost_exp))
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def _make_lookahead_heuristic(prizes, distance_matrix, max_len, depth=2, temp=1.0):
    """
    Lookahead heuristic: for each (i, j), estimate the best additional prize
    collectable after visiting j, given the remaining budget.
    Approximated by: prize[j] + max over k of prize[k] * P(k reachable after j)
    where P is based on slack after i->j->k->0.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    # For each pair (j, k), compute cost of j->k->0
    C_jk0 = distance_matrix + d_j0[None, :]  # C_jk0[j, k] = d[j,k] + d[k,0]
    
    eta = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                eta[i, j] = 1e-9
                continue
            
            cost_ij0 = distance_matrix[i, j] + d_j0[j]
            if cost_ij0 > max_len:
                eta[i, j] = 1e-9
                continue
            
            # Prize from j
            base = prizes_safe[j]
            
            # Lookahead: best additional prize from k after visiting j
            # Remaining budget after i->j->0 would be max_len - cost_ij0
            # But we don't have to return immediately; we can go j->k->0
            # Cost j->k->0 = C_jk0[j, k]
            # Feasible if cost_ij0 - d_j0[j] + C_jk0[j, k] <= max_len
            # i.e., d[i,j] + C_jk0[j, k] <= max_len
            
            remaining_for_k = max_len - distance_matrix[i, j]
            # C_jk0[j, k] = d[j,k] + d[k,0]
            # Feasible if d[j,k] + d[k,0] <= remaining_for_k
            
            feasible_k = C_jk0[j, :] <= remaining_for_k
            if np.any(feasible_k):
                # Weight by prize and slack
                slack_k = np.maximum(0.0, remaining_for_k - C_jk0[j, :]) / max_len
                att_k = prizes_safe * slack_k
                att_k[~feasible_k] = 0
                best_k = np.max(att_k) if np.any(feasible_k) else 0
            else:
                best_k = 0
            
            eta[i, j] = base + temp * best_k
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def _make_density_heuristic(prizes, distance_matrix, max_len, gamma=1.0):
    """
    Density-based heuristic: entry (i, j) = prize[j] * (1 + gamma * sum of prizes of nearby nodes k 
    that are also reachable after j) / (d[i,j] + d[j,0] + eps)
    This encourages visiting nodes that are in clusters of high-value nodes.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    C = distance_matrix + d_j0[None, :]  # C[i, j] = d[i,j] + d[j,0]
    C_safe = np.maximum(C, 1e-9)
    
    # For each j, compute the "cluster value": sum of prizes of nodes k within some radius
    # Use a soft kernel: sum_k prize[k] * exp(-d[j,k] / sigma)
    sigma = max_len / 4.0
    D = distance_matrix  # D[j, k] = d[j,k]
    kernel = np.exp(-D / sigma)
    cluster_val = prizes_safe @ kernel  # cluster_val[j] = sum_k prize[k] * kernel[j,k]
    
    # Heuristic: prize[j] * (1 + gamma * cluster_val[j]) / C[i,j]
    eta = prizes_safe[None, :] * (1.0 + gamma * cluster_val[None, :]) / C_safe
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def _make_regret_heuristic(prizes, distance_matrix, max_len, alpha=1.0, beta=1.0):
    """
    Regret-based heuristic: entry (i, j) approximates the counterfactual loss
    if the ant at i does NOT visit j next.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    
    att = prizes_safe[None, :] / C_safe
    
    att_scaled = alpha * att
    att_max = np.max(att_scaled, axis=1, keepdims=True)
    exp_att = np.exp(att_scaled - att_max)
    P_opt = exp_att / np.sum(exp_att, axis=1, keepdims=True)
    
    slack = np.maximum(0.0, max_len - C) / max_len
    slack = np.maximum(slack, 1e-9)
    
    eta = prizes_safe[None, :] * P_opt * (1.0 + beta * slack)
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def _make_simple_heuristic(prizes, distance_matrix, max_len, type_id, p1, p2, p3):
    """Generate various simple heuristic types."""
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    max_prize = np.max(prizes_safe)
    max_cost = np.max(C_safe)
    
    try:
        if type_id == 0:  # Power Law
            prize_norm = prizes_safe / (max_prize + 1e-9)
            cost_norm = C_safe / (max_cost + 1e-9)
            eta = np.power(prize_norm[None, :], p2) / np.power(cost_norm, p2)
        elif type_id == 1:  # Exponential
            prize_norm = prizes_safe / (max_prize + 1e-9)
            cost_norm = C_safe / (max_cost + 1e-9)
            eta = np.exp(p1 * prize_norm[None, :] - p2 * cost_norm)
        elif type_id == 2:  # Ratio with Slack
            slack = np.maximum(1.0 - C / max_len, 0.0)
            slack = np.maximum(slack, 1e-9)
            eta = (prizes_safe[None, :] / C_safe) * np.power(slack, p1)
        elif type_id == 3:  # Rank-based
            prize_ranks = np.argsort(np.argsort(prizes_safe))
            cost_ranks = np.argsort(np.argsort(C_safe))
            cost_ranks_inv = (N - 1) - cost_ranks
            eta = np.power(prize_ranks[None, :] + 1, p1) / np.power(cost_ranks_inv + 1, p2)
        elif type_id == 4:  # Hybrid
            eta = np.power(prizes_safe[None, :], p1) / (C_safe + p3)
        else:
            eta = np.ones((N, N))
    except Exception:
        eta = np.ones((N, N))
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0
    
    N = len(prizes)
    
    # Generate candidate heuristics from multiple families
    candidates = []
    
    # Family 1: Dynamic Potential
    for pe in [0.5, 1.0, 1.5, 2.0]:
        for se in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for ce in [0.0, 0.5, 1.0]:
                candidates.append(('potential', pe, se, ce))
    
    # Family 2: Lookahead
    for temp in [0.0, 0.5, 1.0, 2.0]:
        candidates.append(('lookahead', temp))
    
    # Family 3: Density
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        candidates.append(('density', gamma))
    
    # Family 4: Regret
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for beta in [0.0, 0.5, 1.0, 2.0]:
            candidates.append(('regret', alpha, beta))
    
    # Family 5: Simple heuristics
    for p2 in [0.5, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(('simple', 0, 1.0, p2, 0.0))
    for p1 in [1.0, 2.0, 3.0, 5.0]:
        for p2 in [1.0, 2.0, 3.0, 5.0]:
            candidates.append(('simple', 1, p1, p2, 0.0))
    for p1 in [0.5, 1.0, 2.0]:
        candidates.append(('simple', 2, p1, 0.0, 0.0))
    for p1 in [1.0, 2.0]:
        for p2 in [1.0, 2.0]:
            candidates.append(('simple', 3, p1, p2, 0.0))
    for p1 in [1.0, 2.0]:
        for p2 in [1.0, 2.0]:
            for p3 in [0.1, 0.5, 1.0]:
                candidates.append(('simple', 4, p1, p2, p3))
    
    def _make_eta(c):
        if c[0] == 'potential':
            _, pe, se, ce = c
            return _make_potential_heuristic(prizes, distance_matrix, max_len, pe, se, ce)
        elif c[0] == 'lookahead':
            _, temp = c
            return _make_lookahead_heuristic(prizes, distance_matrix, max_len, depth=2, temp=temp)
        elif c[0] == 'density':
            _, gamma = c
            return _make_density_heuristic(prizes, distance_matrix, max_len, gamma)
        elif c[0] == 'regret':
            _, alpha, beta = c
            return _make_regret_heuristic(prizes, distance_matrix, max_len, alpha, beta)
        else:
            _, type_id, p1, p2, p3 = c
            return _make_simple_heuristic(prizes, distance_matrix, max_len, type_id, p1, p2, p3)
    
    best_avg = -1
    best_eta = None
    
    # Pass 1: Quick screen with seed 0
    quick_results = []
    for c in candidates:
        if time.time() - start_time > time_limit - 90:
            break
        
        eta = _make_eta(c)
        s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
        quick_results.append((s, c, eta))
    
    if not quick_results:
        return _make_potential_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0, 0.5)
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Pass 2: Evaluate top candidates with seeds 1 and 2
    top_k = min(20, len(quick_results))
    for s_quick, c, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 30:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 25:
                break
            s = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Pass 3: Refine around best if time permits
    if best_eta is not None and time.time() - start_time < time_limit - 40:
        # Try perturbations of the best candidate
        best_c = None
        for s_quick, c, eta in quick_results[:top_k]:
            if np.array_equal(eta, best_eta):
                best_c = c
                break
        
        if best_c is not None:
            # Generate nearby candidates
            if best_c[0] == 'potential':
                base_pe, base_se, base_ce = best_c[1], best_c[2], best_c[3]
                for pe in [base_pe * 0.5, base_pe * 1.5]:
                    for se in [base_se * 0.5, base_se * 1.5]:
                        for ce in [base_ce * 0.5, base_ce * 1.5]:
                            if ce < 0:
                                continue
                            nc = ('potential', pe, se, ce)
                            if nc not in [c for _, c, _ in quick_results]:
                                eta = _make_eta(nc)
                                s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
                                if s > best_avg * 0.95:
                                    scores = [s]
                                    for seed in [1, 2]:
                                        if time.time() - start_time > time_limit - 15:
                                            break
                                        s2 = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
                                        scores.append(s2)
                                    avg = np.mean(scores)
                                    if avg > best_avg:
                                        best_avg = avg
                                        best_eta = eta
            elif best_c[0] == 'regret':
                base_alpha, base_beta = best_c[1], best_c[2]
                for alpha in [base_alpha * 0.5, base_alpha * 1.5]:
                    for beta in [base_beta * 0.5, base_beta * 1.5]:
                        nc = ('regret', alpha, beta)
                        if nc not in [c for _, c, _ in quick_results]:
                            eta = _make_eta(nc)
                            s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
                            if s > best_avg * 0.95:
                                scores = [s]
                                for seed in [1, 2]:
                                    if time.time() - start_time > time_limit - 15:
                                        break
                                    s2 = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
                                    scores.append(s2)
                                avg = np.mean(scores)
                                if avg > best_avg:
                                    best_avg = avg
                                    best_eta = eta
    
    # Final safety check
    if best_eta is None or not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = _make_potential_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0, 0.5)
    
    return best_eta
