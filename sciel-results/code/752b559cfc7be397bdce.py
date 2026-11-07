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

def _make_lookahead_heuristic(prizes, distance_matrix, max_len, depth=2, alpha=1.0):
    """
    Lookahead heuristic: for each edge (i,j), estimate the best prize obtainable
    in the next 'depth' steps after visiting j, given the remaining length budget.
    
    eta[i,j] = prize[j] + alpha * expected_future_prize(i->j)
    where expected_future_prize is computed by a greedy 2-step lookahead.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    
    eta = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                eta[i, j] = 1e-9
                continue
            
            cost_ij = distance_matrix[i, j]
            cost_j0 = distance_matrix[j, 0]
            total_cost = cost_ij + cost_j0
            
            if total_cost > max_len:
                eta[i, j] = 1e-9
                continue
            
            # Immediate prize
            immediate = prizes[j]
            
            # Lookahead: after going to j, what's the best next prize?
            remaining_after_j = max_len - cost_ij
            best_future = 0.0
            
            if depth >= 2:
                for k in range(N):
                    if k == i or k == j:
                        continue
                    cost_jk = distance_matrix[j, k]
                    cost_k0 = distance_matrix[k, 0]
                    if cost_jk + cost_k0 <= remaining_after_j:
                        # We can go j->k->0
                        # But we need to check if we can also go back to 0 from i eventually
                        # Actually the constraint is: path_len + d[i,j] + d[j,0] <= max_len
                        # After visiting j, the ant is at j. The next move is j->k.
                        # The constraint for j->k is: (path_len + d[i,j]) + d[j,k] + d[k,0] <= max_len
                        # path_len is 0 at start, so: d[i,j] + d[j,k] + d[k,0] <= max_len
                        if cost_ij + cost_jk + cost_k0 <= max_len:
                            future = prizes[k]
                            if future > best_future:
                                best_future = future
            
            eta[i, j] = immediate + alpha * best_future
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta

def _make_density_heuristic(prizes, distance_matrix, max_len, gamma=1.0):
    """
    Density heuristic: favor nodes that are close to high-prize clusters.
    eta[i,j] = prize[j] * (1 + gamma * density_j) / (d[i,j] + eps)
    where density_j = sum of prizes of nodes within radius r of j.
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    C = distance_matrix + d_j0[None, :]
    C_safe = np.maximum(C, 1e-9)
    
    # Compute density: for each node j, sum prizes of nodes within some distance
    # Use a radius based on max_len
    radius = max_len * 0.5
    density = np.zeros(N)
    for j in range(N):
        for k in range(N):
            if j != k and distance_matrix[j, k] <= radius:
                density[j] += prizes[k]
    
    density = np.maximum(density, 1e-9)
    max_density = np.max(density)
    if max_density > 0:
        density_norm = density / max_density
    else:
        density_norm = np.ones(N)
    
    eta = (prizes_safe[None, :] * (1.0 + gamma * density_norm[None, :])) / C_safe
    
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

def _make_potential_heuristic(prizes, distance_matrix, max_len, prize_exp=1.0, slack_exp=1.0, cost_exp=0.5):
    """
    Dynamic Potential heuristic: entry (i, j) = prize[j]^prize_exp * slack_norm^slack_exp / (d[i,j] + eps)^cost_exp
    """
    N = len(prizes)
    prizes_safe = np.maximum(prizes, 1e-9)
    d_j0 = distance_matrix[:, 0]
    
    C = distance_matrix + d_j0[None, :]
    
    slack = np.maximum(0.0, max_len - C) / max_len
    slack = np.maximum(slack, 1e-9)
    
    d_ij = np.maximum(distance_matrix, 1e-9)
    
    eta = (np.power(prizes_safe[None, :], prize_exp) * 
           np.power(slack, slack_exp) / 
           np.power(d_ij, cost_exp))
    
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 310.0  # Safety margin before 330s kill
    
    N = len(prizes)
    
    # Generate candidate heuristics
    candidates = []
    
    # Lookahead heuristics
    for alpha in [0.5, 1.0, 2.0]:
        candidates.append(('lookahead', alpha))
    
    # Density heuristics
    for gamma in [0.5, 1.0, 2.0]:
        candidates.append(('density', gamma))
    
    # Regret-based heuristics
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        for beta in [0.0, 0.5, 1.0, 2.0]:
            candidates.append(('regret', alpha, beta))
    
    # Potential heuristics
    for pe in [0.5, 1.0, 2.0]:
        for se in [0.5, 1.0, 2.0]:
            for ce in [0.5, 1.0]:
                candidates.append(('potential', pe, se, ce))
    
    # Simple heuristics
    for p2 in [0.5, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(('simple', 0, 1.0, p2, 0.0))
    
    for p1 in [1.0, 2.0, 3.0]:
        for p2 in [1.0, 2.0, 3.0]:
            candidates.append(('simple', 1, p1, p2, 0.0))
    
    for p1 in [0.5, 1.0, 2.0]:
        candidates.append(('simple', 2, p1, 0.0, 0.0))
    
    # Evaluate candidates with quick screen
    quick_results = []
    for c in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        
        try:
            if c[0] == 'lookahead':
                eta = _make_lookahead_heuristic(prizes, distance_matrix, max_len, depth=2, alpha=c[1])
            elif c[0] == 'density':
                eta = _make_density_heuristic(prizes, distance_matrix, max_len, gamma=c[1])
            elif c[0] == 'regret':
                eta = _make_regret_heuristic(prizes, distance_matrix, max_len, c[1], c[2])
            elif c[0] == 'potential':
                eta = _make_potential_heuristic(prizes, distance_matrix, max_len, c[1], c[2], c[3])
            elif c[0] == 'simple':
                eta = _make_simple_heuristic(prizes, distance_matrix, max_len, c[1], c[2], c[3], c[4])
            else:
                continue
            
            s = _simulate_aco(prizes, distance_matrix, max_len, eta, 0)
            quick_results.append((s, c, eta))
        except Exception:
            continue
    
    if not quick_results:
        return _make_regret_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0)
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Second pass: Evaluate top candidates with more seeds
    best_avg = -1
    best_eta = None
    top_k = min(10, len(quick_results))
    for s_quick, c, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 15:
                break
            s = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Final safety check
    if best_eta is None or not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = _make_regret_heuristic(prizes, distance_matrix, max_len, 1.0, 1.0)
    
    return best_eta
