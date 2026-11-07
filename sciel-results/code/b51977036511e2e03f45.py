import numpy as np
import time

def _run_aco(prizes, dist, max_len, eta, n_ants=20, n_iter=30, seed=42):
    """
    Run ACO and return the best total prize collected by any ant.
    Vectorized where possible.
    """
    N = len(prizes)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix starts at 1
    tau = np.ones((N, N))
    
    best_ever = 0.0
    return_cost = dist[:, 0]
    
    for iteration in range(n_iter):
        best_this_iter = 0.0
        
        for ant in range(n_ants):
            visited = np.zeros(N, dtype=bool)
            visited[0] = True
            current = 0
            length_so_far = 0.0
            prize_collected = 0.0
            
            while True:
                # Find unvisited nodes j such that length_so_far + d[current, j] + d[j, 0] <= max_len
                candidates = []
                for j in range(N):
                    if visited[j]:
                        continue
                    if length_so_far + dist[current, j] + return_cost[j] <= max_len + 1e-9:
                        candidates.append(j)
                
                if not candidates:
                    break
                
                candidates = np.array(candidates)
                
                # Compute probabilities proportional to tau[current, j] * eta[current, j]
                tau_vals = np.maximum(tau[current, candidates], 1e-9)
                eta_vals = np.maximum(eta[current, candidates], 1e-9)
                weights = tau_vals * eta_vals
                
                total = weights.sum()
                if total <= 0:
                    break
                probs = weights / total
                
                j = rng.choice(candidates, p=probs)
                
                length_so_far += dist[current, j]
                prize_collected += prizes[j]
                visited[j] = True
                current = j
            
            if prize_collected > best_this_iter:
                best_this_iter = prize_collected
        
        if best_this_iter > best_ever:
            best_ever = best_this_iter
        
        tau *= 0.9
        tau += best_this_iter / 10.0
    
    return best_ever


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin (evaluator kills at 330s)
    
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    N = len(prizes)
    
    # Precompute return costs
    return_cost = dist[:, 0]
    
    # Structural change: opportunity-cost heuristic
    # eta[i,j] = prize[j] * (1 - (d[i,j] + d[j,0]) / max_len)^beta
    # This explicitly penalizes nodes that consume too much of the length budget.
    # Nodes with high prize that still leave slack get high attractiveness.
    
    def make_eta_opportunity(beta, gamma):
        """
        beta: exponent on the slack fraction
        gamma: additional penalty for return cost
        """
        eps = 1e-9
        # total cost of visiting j from i and returning to 0
        total_cost = dist + return_cost[None, :]
        # slack fraction: how much of the budget remains after this move
        slack = np.maximum(1.0 - total_cost / max_len, 0.0)
        # opportunity cost: prize * slack^beta * (1 / return_cost)^gamma
        prize_factor = prizes[None, :] + eps
        return_factor = (return_cost[None, :] + eps)**gamma
        eta = prize_factor * (slack + eps)**beta / return_factor
        return np.maximum(eta, 1e-9)
    
    def make_eta_alpha(alpha):
        eps = 1e-9
        total_cost = dist + return_cost[None, :]
        return np.maximum((prizes[None, :] + eps) / (total_cost + eps)**alpha, 1e-9)
    
    def make_eta_exp(beta):
        eps = 1e-9
        total_cost = dist + return_cost[None, :]
        return np.maximum((prizes[None, :] + eps) * np.exp(-beta * total_cost), 1e-9)
    
    def make_eta_combined(alpha, gamma):
        eps = 1e-9
        prize_factor = prizes[None, :] + eps
        total_cost = dist + return_cost[None, :]
        cost_factor = (total_cost + eps)**alpha
        return_factor = (return_cost[None, :] + eps)**gamma
        return np.maximum(prize_factor / (cost_factor * return_factor), 1e-9)
    
    # Evaluate a heuristic by running ACO
    def evaluate_eta(eta):
        N_local = len(prizes)
        rng = np.random.RandomState(42)
        tau = np.ones((N_local, N_local))
        best_ever = 0.0
        return_cost_local = dist[:, 0]
        
        for iteration in range(30):
            best_this_iter = 0.0
            
            for ant in range(20):
                visited = np.zeros(N_local, dtype=bool)
                visited[0] = True
                current = 0
                length_so_far = 0.0
                prize_collected = 0.0
                
                while True:
                    candidates = []
                    for j in range(N_local):
                        if visited[j]:
                            continue
                        if length_so_far + dist[current, j] + return_cost_local[j] <= max_len + 1e-9:
                            candidates.append(j)
                    
                    if not candidates:
                        break
                    
                    candidates = np.array(candidates)
                    tau_vals = np.maximum(tau[current, candidates], 1e-9)
                    eta_vals = np.maximum(eta[current, candidates], 1e-9)
                    weights = tau_vals * eta_vals
                    total = weights.sum()
                    if total <= 0:
                        break
                    probs = weights / total
                    j = rng.choice(candidates, p=probs)
                    
                    length_so_far += dist[current, j]
                    prize_collected += prizes[j]
                    visited[j] = True
                    current = j
                
                if prize_collected > best_this_iter:
                    best_this_iter = prize_collected
            
            if best_this_iter > best_ever:
                best_ever = best_this_iter
            
            tau *= 0.9
            tau += best_this_iter / 10.0
        
        return best_ever
    
    # Search over parameters
    best_score = 0.0
    best_eta = None
    
    # Candidate heuristics
    candidates = []
    
    # Opportunity-cost based (new structural approach)
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        for gamma in [0.0, 0.5, 1.0, 1.5, 2.0]:
            eta = make_eta_opportunity(beta, gamma)
            candidates.append((f"opp_b={beta}_g={gamma}", eta))
    
    # Alpha-based (existing)
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        eta = make_eta_alpha(a)
        candidates.append((f"alpha={a}", eta))
    
    # Exp-based (existing)
    for b in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
        eta = make_eta_exp(b)
        candidates.append((f"exp_beta={b}", eta))
    
    # Combined (existing)
    for a in [0.5, 1.0, 2.0, 3.0]:
        for g in [0.0, 0.5, 1.0, 2.0, 3.0]:
            eta = make_eta_combined(a, g)
            candidates.append((f"combined_a={a}_g={g}", eta))
    
    # Evaluate each candidate
    for name, eta in candidates:
        if time.time() - start_time > time_limit - 30:
            break
        score = evaluate_eta(eta)
        if score > best_score:
            best_score = score
            best_eta = eta
    
    # If we have time, refine around the best opportunity-cost parameters
    if best_eta is not None and time.time() - start_time < time_limit - 60:
        # Refine opportunity-cost parameters
        best_opp_beta = 1.0
        best_opp_gamma = 0.5
        for beta in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]:
            for gamma in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
                if time.time() - start_time > time_limit - 30:
                    break
                eta = make_eta_opportunity(beta, gamma)
                score = evaluate_eta(eta)
                if score > best_score:
                    best_score = score
                    best_eta = eta
                    best_opp_beta = beta
                    best_opp_gamma = gamma
    
    # If no eta was found, use a default
    if best_eta is None:
        best_eta = make_eta_opportunity(1.0, 0.5)
    
    return best_eta
