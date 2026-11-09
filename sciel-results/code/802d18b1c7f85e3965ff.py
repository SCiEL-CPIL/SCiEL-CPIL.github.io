import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Analytic base heuristic
    # eta_analytic[i, j] = prize[j] / (d[i, j] + d[j, 0])
    # This favors nodes with high prize that are close to current and depot.
    
    # Ensure no division by zero
    denom = distance_matrix + distance_matrix[:, 0][None, :] 
    # Wait, d[j, 0] is the same for all i.
    # denom[i, j] = d[i, j] + d[j, 0]
    
    # Calculate analytic part
    # prizes shape (n,), dist shape (n, n)
    # We want eta[i, j] to depend on prize[j]
    prize_col = prizes[None, :] # (1, n)
    d_to_depot = distance_matrix[:, 0] # (n,) -> (1, n)
    
    denom = distance_matrix + d_to_depot[None, :]
    # Avoid zero division
    denom = np.where(denom == 0, 1e-9, denom)
    eta_analytic = prize_col / denom
    
    # Set node 0 prize to 0 effectively, but we are calculating transitions TO j.
    # If j=0, prize is 0, so eta is 0. That's fine, we don't return to 0 in the middle.
    
    # Empirical part: Run greedy tours
    # Time check
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for heuristic generation is plenty
    
    K = 500 # Number of greedy tours
    counts = np.zeros((n, n), dtype=np.int32)
    
    for _ in range(K):
        if time.time() > deadline:
            break
            
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        current_len = 0.0
        
        while True:
            # Find feasible next nodes
            # Condition: current_len + d[curr, j] + d[j, 0] <= max_len
            # and not visited[j]
            
            # Vectorized feasibility check
            # d_curr = distance_matrix[curr, :]
            # d_depot = distance_matrix[:, 0]
            # feasible = ~visited & (current_len + d_curr + d_depot <= max_len)
            
            d_curr = distance_matrix[curr, :]
            d_depot = distance_matrix[:, 0]
            
            feasible_mask = (~visited) & (current_len + d_curr + d_depot <= max_len)
            
            if not np.any(feasible_mask):
                break
            
            # Calculate scores for feasible nodes
            # score = prize[j] / (d[curr, j] + d[j, 0])
            # Add small noise to diversify
            scores = np.zeros(n)
            # Only compute for feasible to save time? Or just mask.
            # Using analytic formula for score
            scores[feasible_mask] = eta_analytic[curr, feasible_mask]
            
            # Add noise
            # Multiplicative noise
            noise = np.random.uniform(0.8, 1.2, size=n)
            scores = scores * noise
            
            # Pick max score
            # Set infeasible to -1
            scores[~feasible_mask] = -1
            next_node = np.argmax(scores)
            
            # Record edge
            counts[curr, next_node] += 1
            
            # Update state
            current_len += d_curr[next_node]
            visited[next_node] = True
            curr = next_node
            
    # Combine analytic and empirical
    # Normalize counts
    # counts / K gives probability
    # We want to boost edges that are frequently used in good greedy tours.
    
    # If an edge is never used, counts=0.
    # Let's use: eta_final = eta_analytic * (1 + alpha * (counts / K))
    # This keeps the analytic structure but boosts "good" edges.
    
    alpha = 1.0
    empirical_factor = 1.0 + alpha * (counts / K)
    
    eta_final = eta_analytic * empirical_factor
    
    # Ensure positive values
    eta_final = np.where(eta_final > 0, eta_final, 1e-9)
    
    return eta_final
