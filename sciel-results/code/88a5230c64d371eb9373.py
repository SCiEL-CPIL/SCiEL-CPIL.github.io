import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # Safety margin
    
    N = len(prizes)
    d_j0 = distance_matrix[:, 0].copy()
    prizes_safe = np.maximum(prizes, 1e-9)
    
    # Precompute cost matrix C[i,j] = d[i,j] + d[j,0]
    # This is the cost to go from i to j and then back to start (conceptually for budget check)
    # Note: The actual path constraint is cumulative, but this is the marginal cost of adding j to path ending at i.
    C = distance_matrix + d_j0[None, :]
    
    # Slack matrix: 1 - C[i,j] / max_len
    # If C[i,j] >= max_len, slack <= 0. We clamp to 0 for heuristic calculation, 
    # but ACO will mask these anyway.
    slack = 1.0 - C / max_len
    slack = np.maximum(slack, 0.0)
    
    def run_aco_fast(eta, seed):
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
                    # Cost to move to j and ensure we can return to 0
                    # Constraint: path_len + d[current, j] + d[j, 0] <= max_len
                    # So d[current, j] + d[j, 0] <= remaining
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

    def make_heuristic(alpha, beta, gamma):
        """
        Heuristic: (prize_j^gamma) * (slack_ij)^beta / (C_ij)^alpha
        Where C_ij = d_ij + d_j0
        slack_ij = max(0, 1 - C_ij / max_len)
        
        alpha: penalty for distance
        beta: emphasis on slack (opportunity cost)
        gamma: emphasis on prize
        """
        # Avoid division by zero
        C_safe = np.maximum(C, 1e-9)
        
        # Base prize term
        prize_term = np.power(prizes_safe[None, :], gamma)
        
        # Distance penalty
        dist_term = np.power(C_safe, -alpha)
        
        # Slack term (Opportunity Cost)
        # If slack is 0, this term is 0 (unless beta=0).
        # We add a small epsilon to avoid 0^0 issues or log issues, but power is fine.
        # However, if beta > 0 and slack is 0, eta is 0. This is good for infeasible nodes.
        slack_term = np.power(slack, beta)
        
        eta = prize_term * dist_term * slack_term
        
        # Ensure finite and positive
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta

    # Define candidate space
    # alpha: 0.5 to 2.0
    # beta: 0.0 to 2.0
    # gamma: 0.5 to 2.0
    
    candidates = []
    
    # Coarse grid
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for gamma in [0.5, 1.0, 1.5, 2.0]:
                candidates.append((alpha, beta, gamma))
    
    # Quick screen
    quick_results = []
    for alpha, beta, gamma in candidates:
        if time.time() - start_time > time_limit - 60:
            break
        eta = make_heuristic(alpha, beta, gamma)
        s = run_aco_fast(eta, 0)
        quick_results.append((s, alpha, beta, gamma, eta))
    
    if not quick_results:
        # Fallback
        return make_heuristic(1.0, 1.0, 1.0)
    
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with more seeds
    top_k = min(10, len(quick_results))
    best_avg = -1
    best_eta = quick_results[0][4]
    
    for s_quick, alpha, beta, gamma, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]
        for seed in [1, 2]:
            if time.time() - start_time > time_limit - 15:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        avg = np.mean(scores)
        if avg > best_avg:
            best_avg = avg
            best_eta = eta
    
    # Phase 3: Local refinement
    if time.time() - start_time < time_limit - 30:
        best_entry = quick_results[0]
        balpha, bbeta, bgamma = best_entry[1], best_entry[2], best_entry[3]
        
        nearby = []
        # Fine grid around best
        for da in [-0.25, -0.1, 0.1, 0.25]:
            for db in [-0.25, -0.1, 0.1, 0.25]:
                for dg in [-0.25, -0.1, 0.1, 0.25]:
                    na = balpha + da
                    nb = bbeta + db
                    ng = bgamma + dg
                    if na > 0 and nb >= 0 and ng > 0:
                        nearby.append((na, nb, ng))
        
        seen = set()
        for alpha, beta, gamma in nearby:
            key = (round(alpha, 3), round(beta, 3), round(gamma, 3))
            if key in seen:
                continue
            seen.add(key)
            
            if time.time() - start_time > time_limit - 10:
                break
            
            eta = make_heuristic(alpha, beta, gamma)
            s = run_aco_fast(eta, 0)
            if s > best_avg * 0.95:
                scores = [s]
                for seed in [1, 2]:
                    if time.time() - start_time > time_limit - 8:
                        break
                    s2 = run_aco_fast(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_eta = eta
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = make_heuristic(1.0, 1.0, 1.0)
    
    return best_eta
