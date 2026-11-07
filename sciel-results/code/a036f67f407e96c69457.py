import numpy as np
import time

def _run_aco_fast(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """
    Fast vectorized ACO simulation.
    Returns the best total prize collected by any ant across all iterations.
    """
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    d = distance_matrix
    d_j0 = d[0, :]
    cost = d + d_j0[None, :]  # cost[i,j] = d[i,j] + d[j,0]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
        
        # Pheromone update
        tau = 0.9 * tau + (best_iter / 10.0)
    
    return best_global


def _heuristic_from_params(prizes, distance_matrix, max_len, params):
    """
    Build a heuristic matrix from a set of parameters.
    params = [alpha, beta, gamma, delta, epsilon_scale]
    heuristic[i,j] = (prizes[j]^alpha) / (d[i,j] + d[j,0])^beta * exp(-gamma * d[i,j]) * (1 + delta * prizes[j])
    with epsilon_scale controlling the floor.
    """
    n = len(prizes)
    d = distance_matrix
    p = prizes
    d_j0 = d[0, :]
    
    alpha, beta, gamma, delta, eps_scale = params
    
    # Ensure positive
    p_safe = np.maximum(p, 1e-9)
    
    p_pow = p_safe ** alpha
    denom = (d + d_j0[None, :] + 1e-9) ** beta
    exp_term = np.exp(-gamma * d)
    prize_boost = 1.0 + delta * p_safe
    
    h = (p_pow[None, :] / denom) * exp_term * prize_boost[None, :]
    
    # Floor
    h = np.maximum(h, eps_scale)
    
    return h


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    
    # Initial guess for Nelder-Mead
    # Start with reasonable values
    x0 = np.array([1.0, 1.0, 0.5, 0.0, 1e-9])
    
    # Define the objective: maximize ACO score, so minimize negative
    def objective(params):
        if time.time() > deadline - 5.0:
            return 0.0
        try:
            h = _heuristic_from_params(prizes, distance_matrix, max_len, params)
            if h.shape != (n, n) or not np.all(np.isfinite(h)):
                return 0.0
            # Quick ACO run for optimization
            score = _run_aco_fast(prizes, distance_matrix, max_len, h, 
                                   num_ants=10, num_iters=10, seed=0)
            return -score
        except:
            return 0.0
    
    # Run Nelder-Mead
    best_score = 0.0
    best_params = x0.copy()
    
    try:
        # Manual Nelder-Mead implementation for control over time
        from scipy.optimize import minimize
        
        # Try scipy minimize with time check
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-4,
                               'adaptive': True})
        
        if time.time() < deadline - 10.0:
            # Evaluate the result
            h_candidate = _heuristic_from_params(prizes, distance_matrix, max_len, res.x)
            if h_candidate.shape == (n, n) and np.all(np.isfinite(h_candidate)):
                score_candidate = _run_aco_fast(prizes, distance_matrix, max_len, 
                                                 h_candidate, num_ants=20, num_iters=30, seed=0)
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_params = res.x.copy()
    except Exception as e:
        pass
    
    # If we have time, try a few more local refinements
    if time.time() < deadline - 30.0:
        # Try perturbing the best params
        perturbations = [
            best_params + np.array([0.2, 0, 0, 0, 0]),
            best_params + np.array([-0.2, 0, 0, 0, 0]),
            best_params + np.array([0, 0.2, 0, 0, 0]),
            best_params + np.array([0, -0.2, 0, 0, 0]),
            best_params + np.array([0, 0, 0.2, 0, 0]),
            best_params + np.array([0, 0, -0.2, 0, 0]),
            best_params + np.array([0, 0, 0, 0.2, 0]),
            best_params + np.array([0, 0, 0, -0.2, 0]),
        ]
        
        for pert in perturbations:
            if time.time() > deadline - 15.0:
                break
            if pert[0] < 0.1 or pert[1] < 0.1 or pert[2] < 0 or pert[3] < 0 or pert[4] < 1e-9:
                continue
            try:
                h_pert = _heuristic_from_params(prizes, distance_matrix, max_len, pert)
                if h_pert.shape != (n, n) or not np.all(np.isfinite(h_pert)):
                    continue
                score_pert = _run_aco_fast(prizes, distance_matrix, max_len, 
                                            h_pert, num_ants=20, num_iters=30, seed=0)
                if score_pert > best_score:
                    best_score = score_pert
                    best_params = pert.copy()
            except:
                pass
    
    # Final: build the heuristic from best params
    try:
        final_h = _heuristic_from_params(prizes, distance_matrix, max_len, best_params)
        if final_h.shape != (n, n) or not np.all(np.isfinite(final_h)):
            raise ValueError("Invalid heuristic")
    except:
        # Fallback: simple prize/distance heuristic
        d_j0 = d[0, :]
        p_safe = np.maximum(p, 1e-9)
        final_h = p_safe[None, :] / (d + d_j0[None, :] + 1e-9)
    
    final_h = final_h.copy()
    final_h[final_h <= 0] = 1e-9
    
    return final_h
