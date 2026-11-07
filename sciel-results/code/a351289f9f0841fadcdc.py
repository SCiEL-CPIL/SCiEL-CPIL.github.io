import numpy as np
from scipy.optimize import minimize

def _simulate_aco(prizes, dist, max_len, eta, seed):
    """
    Simulates the ACO process for a single instance given a heuristic matrix.
    Returns the best prize collected by any ant in the run.
    """
    n = len(prizes)
    np.random.seed(seed)
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_prize_overall = 0.0
    
    for _ in range(30):
        best_prize_iter = 0.0
        
        for _ in range(20):
            current = 0
            visited = [0]
            visited_set = {0}
            path_len = 0.0
            total_prize = 0.0
            
            while True:
                # Find valid next nodes
                valid_nodes = []
                valid_probs = []
                
                for j in range(n):
                    if j in visited_set:
                        continue
                    # Check constraint: length so far + d[i,j] + d[j,0] <= max_len
                    if path_len + dist[current, j] + dist[j, 0] <= max_len:
                        valid_nodes.append(j)
                        # Pheromone * Heuristic
                        # Ensure positive, otherwise 1e-9 (though our eta is positive)
                        val = tau[current, j] * eta[current, j]
                        if val > 0:
                            valid_probs.append(val)
                        else:
                            valid_probs.append(1e-9)
                
                if not valid_nodes:
                    break
                
                # Normalize probabilities
                probs = np.array(valid_probs)
                probs = probs / np.sum(probs)
                
                # Select next node
                idx = np.random.choice(len(valid_nodes), p=probs)
                next_node = valid_nodes[idx]
                
                # Update state
                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited.append(next_node)
                visited_set.add(next_node)
                current = next_node
        
        if total_prize > best_prize_iter:
            best_prize_iter = total_prize
            
        # Update pheromone
        tau = tau * 0.9 + (best_prize_iter / 10.0)
        
        if best_prize_iter > best_prize_overall:
            best_prize_overall = best_prize_iter

    return best_prize_overall

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an optimized heuristic matrix for the given instance.
    Uses local search on the parameters of a parametric heuristic family.
    """
    prizes = np.array(prizes)
    dist = np.array(distance_matrix)
    n = len(prizes)
    
    # Parametric heuristic: eta[i, j] = (prize[j] + eps)^alpha / (dist[i, j] + eps)^beta
    # We optimize alpha and beta.
    
    def objective(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e6 # Penalty
        
        eps = 1e-9
        # Vectorized computation of eta
        # prizes[None, :] -> (1, n), dist -> (n, n)
        # We want eta[i, j] to depend on prize[j] and dist[i, j]
        p_j = prizes[None, :] + eps
        d_ij = dist + eps
        
        eta = (p_j ** alpha) / (d_ij ** beta)
        
        # Ensure non-negative/positive handling as per spec
        eta[eta <= 0] = 1e-9
        
        # Evaluate on a small set of seeds to find robust parameters
        # Using seeds 0, 1, 2 to average out stochastic noise
        scores = []
        for s in [0, 1, 2]:
            score = _simulate_aco(prizes, dist, max_len, eta, s)
            scores.append(score)
            
        return -np.mean(scores)
    
    # Initial guess
    x0 = [1.0, 1.0]
    
    # Bounds
    bounds = [(0.1, 5.0), (0.1, 5.0)]
    
    # Optimize
    try:
        # Nelder-Mead is robust for small dimensions
        res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                       options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-4})
        alpha, beta = res.x
    except Exception:
        alpha, beta = 1.0, 1.0
        
    # Clamp to bounds
    alpha = np.clip(alpha, 0.1, 5.0)
    beta = np.clip(beta, 0.1, 5.0)
    
    eps = 1e-9
    p_j = prizes[None, :] + eps
    d_ij = dist + eps
    
    final_eta = (p_j ** alpha) / (d_ij ** beta)
    final_eta[final_eta <= 0] = 1e-9
    
    return final_eta
