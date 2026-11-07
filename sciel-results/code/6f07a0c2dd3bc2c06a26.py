import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280.0  # 50s margin from 330s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    if n == 0:
        return np.array([])
    
    # Greedy evaluator: given heuristic, find best path greedily
    def greedy_prize(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        total_len = 0.0
        total_prize = 0.0
        
        for _ in range(n):
            best_val = -1.0
            best_j = -1
            for j in range(1, n):
                if visited[j]:
                    continue
                new_len = total_len + distance_matrix[curr, j] + distance_matrix[j, 0]
                if new_len <= max_len:
                    val = eta[curr, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
            
            if best_j == -1:
                break
            
            total_len += distance_matrix[curr, best_j]
            total_prize += prizes[best_j]
            visited[best_j] = True
            curr = best_j
        
        return total_prize
    
    # Compute heuristic for given parameters
    def compute_heuristic(mode, alpha=1.0, beta=0.0):
        eps = 1e-9
        dist_to_origin = distance_matrix[:, 0]
        
        if mode == 0:  # Independent of i: prize[j] / (d[j,0] + eps)^alpha
            prize_term = (prizes + eps)
            dist_j0_term = (dist_to_origin + eps) ** alpha
            eta = prize_term[None, :] / dist_j0_term[None, :]
        elif mode == 1:  # prize[j] / ((d[i,j] + eps)^beta * (d[j,0] + eps)^alpha)
            prize_term = (prizes + eps)
            dist_ij_term = (distance_matrix + eps) ** beta
            dist_j0_term = (dist_to_origin + eps) ** alpha
            eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
        elif mode == 2:  # prize[j] / (d[i,j] + d[j,0] + eps)
            total_cost = distance_matrix + dist_to_origin[None, :]
            eta = (prizes + eps)[None, :] / (total_cost + eps)
        elif mode == 3:  # prize[j] / (d[j,0] + eps) * (1 / (d[i,j] + eps))^beta
            prize_term = (prizes + eps)
            dist_j0_term = (dist_to_origin + eps)
            dist_ij_term = (distance_matrix + eps) ** beta
            eta = (prize_term[None, :] / dist_j0_term[None, :]) / dist_ij_term
        elif mode == 4:  # prize[j] / (d[j,0] + eps)^alpha * (1 - d[i,j]/max_d)^beta
            max_d = np.max(distance_matrix)
            if max_d == 0:
                max_d = 1.0
            prize_term = (prizes + eps)
            dist_j0_term = (dist_to_origin + eps) ** alpha
            dist_norm = distance_matrix / max_d
            dist_factor = (1 - dist_norm + eps) ** beta
            eta = (prize_term[None, :] / dist_j0_term[None, :]) * dist_factor
        else:
            eta = np.ones((n, n))
        
        # Ensure non-negative and finite
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta
    
    # Search over parameters
    best_prize = -1.0
    best_config = (0, 1.0, 0.0)
    
    modes = [0, 1, 2, 3, 4]
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    betas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    
    for mode in modes:
        for alpha in alphas:
            for beta in betas:
                # Skip invalid combinations
                if mode == 0 and beta != 0.0:
                    continue
                if mode == 2 and (alpha != 1.0 or beta != 0.0):
                    continue
                if time.time() > deadline - 10:
                    break
                
                eta = compute_heuristic(mode, alpha, beta)
                p = greedy_prize(eta)
                
                if p > best_prize:
                    best_prize = p
                    best_config = (mode, alpha, beta)
            
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    mode, alpha, beta = best_config
    heuristic = compute_heuristic(mode, alpha, beta)
    
    return heuristic
