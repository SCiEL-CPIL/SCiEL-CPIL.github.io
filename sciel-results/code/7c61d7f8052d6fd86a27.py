import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a regret-based heuristic matrix for the given instance.
    Entry (i, j) represents the counterfactual value of moving from i to j,
    approximated by prize[j] / (d[i,j] + d[j,0])^gamma, which captures
    the prize density per unit of length cost. Nodes with high prize relative
    to their round-trip cost are strongly favored.
    """
    prizes = np.array(prizes, dtype=np.float64)
    dist = np.array(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Start time for deadline guard
    t_start = time.time()
    deadline = 300.0  # Must finish within 300s
    
    # The cost of visiting node j from node i and returning to 0 is d[i,j] + d[j,0]
    # This is the "length cost" that determines when j becomes unreachable
    # We want to favor edges where prize[j] is high relative to this cost
    
    # Compute the round-trip cost matrix: cost[i,j] = d[i,j] + d[j,0]
    # d[j,0] is the distance from j back to start
    d_j0 = dist[:, 0]  # shape (n,)
    cost = dist + d_j0[None, :]  # cost[i,j] = d[i,j] + d[j,0], shape (n,n)
    
    # Prize vector, broadcast to rows
    p_j = prizes[None, :]  # shape (1, n)
    
    # Base heuristic: prize[j] / cost[i,j]^gamma
    # We'll try a few values of gamma and pick the best via a quick evaluation
    # But first, let's define a fast simulation
    
    def simulate_aco_fast(prizes, dist, max_len, eta, seed):
        """Fast ACO simulation returning best prize."""
        n = len(prizes)
        np.random.seed(seed)
        tau = np.ones((n, n))
        best_overall = 0.0
        
        for _ in range(30):
            best_iter = 0.0
            for _ in range(20):
                current = 0
                visited_set = {0}
                path_len = 0.0
                total_prize = 0.0
                
                while True:
                    valid = []
                    probs = []
                    for j in range(n):
                        if j in visited_set:
                            continue
                        if path_len + dist[current, j] + dist[j, 0] <= max_len:
                            valid.append(j)
                            val = tau[current, j] * eta[current, j]
                            probs.append(max(val, 1e-9))
                    
                    if not valid:
                        break
                    
                    p = np.array(probs)
                    p = p / p.sum()
                    idx = np.random.choice(len(valid), p=p)
                    next_node = valid[idx]
                    
                    path_len += dist[current, next_node]
                    total_prize += prizes[next_node]
                    visited_set.add(next_node)
                    current = next_node
                
                if total_prize > best_iter:
                    best_iter = total_prize
            
            tau = tau * 0.9 + best_iter / 10.0
            if best_iter > best_overall:
                best_overall = best_iter
        
        return best_overall
    
    # Try several gamma values and pick the best
    # gamma controls how strongly we penalize high-cost edges
    # Small gamma: prize-dominated, large gamma: cost-sensitive
    
    best_eta = None
    best_score = -1.0
    best_gamma = 1.0
    
    # Check time
    if time.time() - t_start > deadline - 60:
        # Not enough time, use a reasonable default
        gamma = 1.0
        eps = 1e-9
        eta = p_j / (cost + eps) ** gamma
        eta[eta <= 0] = 1e-9
        return eta
    
    # Evaluate a range of gamma values
    gamma_candidates = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for gamma in gamma_candidates:
        if time.time() - t_start > deadline - 30:
            break
        
        eps = 1e-9
        eta = p_j / (cost + eps) ** gamma
        eta[eta <= 0] = 1e-9
        
        # Evaluate on a couple of seeds for robustness
        scores = []
        for s in [0, 1]:
            score = simulate_aco_fast(prizes, dist, max_len, eta, s)
            scores.append(score)
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_gamma = gamma
            best_eta = eta.copy()
    
    # If we have time, refine around the best gamma
    if time.time() - t_start < deadline - 15 and best_eta is not None:
        # Local search around best_gamma
        for dg in [-0.2, -0.1, 0.1, 0.2]:
            if time.time() - t_start > deadline - 10:
                break
            gamma = best_gamma + dg
            if gamma <= 0:
                gamma = 0.1
            eps = 1e-9
            eta = p_j / (cost + eps) ** gamma
            eta[eta <= 0] = 1e-9
            
            scores = []
            for s in [0, 1]:
                score = simulate_aco_fast(prizes, dist, max_len, eta, s)
                scores.append(score)
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_gamma = gamma
                best_eta = eta.copy()
    
    if best_eta is None:
        # Fallback
        eps = 1e-9
        best_eta = p_j / (cost + eps)
        best_eta[best_eta <= 0] = 1e-9
    
    return best_eta
