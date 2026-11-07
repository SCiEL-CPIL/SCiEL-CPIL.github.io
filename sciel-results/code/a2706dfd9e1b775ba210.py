import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    # Pre-compute distance to start for all nodes
    d_to_start = distance_matrix[:, 0]
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0.0
            total_prize = prizes[0]
            
            while True:
                remaining_budget = max_len - path_len
                # Condition: d[i, j] + d[j, 0] <= remaining_budget
                d_from_current = distance_matrix[current_node, :]
                valid_indices = np.where(
                    (~visited_mask) & 
                    (d_from_current + d_to_start <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                # Get heuristic weights for valid nodes
                h_vals = H[current_node, valid_indices]
                
                # Normalize to probabilities
                total_h = np.sum(h_vals)
                if total_h == 0:
                    break
                    
                probs = h_vals / total_h
                
                # Sample next node
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
                # Update state
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    n = len(prizes)
    eps = 1e-9
    
    # Pre-compute static components
    d_to_start = distance_matrix[:, 0]
    
    # Strategy: 
    # The heuristic H[i, j] should reflect the "value density" of moving i->j.
    # Value: prize[j]
    # Cost: The "opportunity cost" is how much of the budget is consumed.
    # Specifically, moving i->j consumes d[i,j] from the path length, and we must reserve d[j,0] for return.
    # Total constraint: path_len + d[i,j] + d[j,0] <= max_len.
    # At the start of the trip, the "slack" is max_len.
    # A node j is attractive if prize[j] is high and (d[i,j] + d[j,0]) is low relative to the budget.
    
    # We construct H[i, j] = prize[j] / (d[i,j] + d[j,0] + beta)
    # But we want it to be instance-specific and robust.
    # Let's try a few variations of this "budget-aware" heuristic and pick the best one via a quick simulation.
    
    # Variation 1: Inverse distance to start weighted by prize
    # H1[i, j] = prize[j] / (d[i,j] + d[j,0])
    
    # Variation 2: Prize weighted by "slack" preservation
    # If we go to j, we use d[i,j]. We must keep d[j,0].
    # Effective cost = d[i,j] + d[j,0].
    # H2[i, j] = prize[j] / (d[i,j] + d[j,0])^gamma
    
    # Variation 3: Normalize by max possible distance to ensure scale invariance
    # max_d = max(d[i,j] + d[j,0])
    # H3[i, j] = prize[j] / (max_d * (d[i,j] + d[j,0]) / max_d + beta) -> essentially same as V2 with scaling
    
    # Let's define a generic function to build heuristic from parameters
    def build_heuristic(gamma, beta, alpha_prize):
        # alpha_prize: power of prize? or offset? Let's stick to linear prize for now, maybe power.
        # H[i,j] = (prize[j] + alpha) / (d[i,j] + d[j,0] + beta)^gamma
        # We need H to be a matrix of same shape.
        # d[i,j] is distance_matrix[i,j]. d[j,0] is d_to_start[j].
        
        # Vectorized construction:
        # cost[i,j] = distance_matrix[i,j] + d_to_start[j]
        cost = distance_matrix + d_to_start[None, :]
        cost = np.maximum(cost, eps)
        
        # prize component
        p = prizes[None, :] ** alpha_prize
        p = np.maximum(p, eps)
        
        H = p / (cost ** gamma)
        
        # Add small beta to denominator to avoid division by zero if cost is 0 (only i=j=0, but 0 is start, usually excluded or handled)
        # Actually, if i=0, j=0, cost=0. But we don't move to visited nodes.
        # Let's just ensure H is positive.
        H = np.where(H > 0, H, eps)
        
        return H

    # Candidate parameters to test
    # gamma: distance sensitivity (higher = more local)
    # beta: not used in build_heuristic directly if we use cost+eps, but let's add it.
    # alpha_prize: prize sensitivity (higher = favors high prize nodes more aggressively)
    
    candidates = [
        (1.0, 0.0, 1.0),  # Standard 1/d
        (1.5, 0.0, 1.0),  # More local
        (2.0, 0.0, 1.0),  # Very local
        (1.0, 0.0, 1.5),  # Favors high prizes
        (1.0, 0.0, 2.0),  # Strongly favors high prizes
        (0.5, 0.0, 1.0),  # Less local, more global
        (1.0, 0.0, 0.5),  # Less prize sensitive
    ]
    
    best_score = -1
    best_H = None
    
    # Evaluate each candidate
    for gamma, beta, alpha_p in candidates:
        if time.time() > deadline:
            break
            
        H = build_heuristic(gamma, beta, alpha_p)
        
        # Quick simulation to evaluate
        score = _simulate_aco(prizes, distance_matrix, max_len, H)
        
        if score > best_score:
            best_score = score
            best_H = H

    # If we have time, try a few more random variations around the best found
    if best_H is None:
        # Fallback
        best_H = build_heuristic(1.0, 0.0, 1.0)
        return best_H

    # Let's try to refine the best parameters found
    # We don't know which candidate was best, so let's just do a small random search around typical good values
    rng = np.random.default_rng(99)
    
    # We'll test a few random points in the space of gamma and alpha_p
    for _ in range(10):
        if time.time() > deadline:
            break
            
        gamma = rng.uniform(0.5, 3.0)
        alpha_p = rng.uniform(0.5, 2.5)
        
        H = build_heuristic(gamma, 0.0, alpha_p)
        score = _simulate_aco(prizes, distance_matrix, max_len, H)
        
        if score > best_score:
            best_score = score
            best_H = H

    return best_H
