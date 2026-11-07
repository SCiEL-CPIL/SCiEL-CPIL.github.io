import numpy as np
import time
import random

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Simulates the ACO process to return the best prize collected by any ant.
    This is the fitness function for our meta-optimization.
    """
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0
            total_prize = prizes[0]
            
            while True:
                # Find valid next nodes
                remaining_budget = max_len - path_len
                # Condition: d[current_node, j] + d[j, 0] <= remaining_budget
                valid_indices = np.where(
                    (~visited_mask) & 
                    (distance_matrix[current_node, :] + distance_matrix[:, 0] <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                # Calculate probabilities
                probs = H[current_node, valid_indices]
                total_prob = np.sum(probs)
                if total_prob == 0:
                    break
                
                probs = probs / total_prob
                
                # Choose next node
                next_idx_in_valid = rng.choice(len(valid_indices), p=probs)
                next_node = valid_indices[next_idx_in_valid]
                
                # Update state
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        
        # Pheromone update logic from contract:
        # "pheromone is multiplied by 0.9 and every entry gains (best prize of that iteration) / 10"
        # Since we are simulating the *effect* of the heuristic on the path, and the pheromone
        # starts uniform and updates uniformly (additive constant), the *relative* probabilities
        # remain proportional to the heuristic H in all iterations.
        # Proof: P_ij = (tau_ij * H_ij) / sum_k (tau_ik * H_ik).
        # If tau_ij = C (constant for all i,j), then P_ij = H_ij / sum_k H_ik.
        # The update rule tau_new = 0.9 * tau_old + delta is affine.
        # If tau is uniform initially, it remains uniform after affine updates with a constant delta.
        # Thus, the choice probabilities are determined solely by H.
        # We do not need to track tau explicitly for the path selection logic,
        # as it does not change the relative weights.
        
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    n = len(prizes)
    eps = 1e-9
    
    # Parameterization of the heuristic:
    # H[i,j] = (prize[j] + offset_prize) / (d[i,j] + offset_dist) ** power
    # We will optimize offset_prize, offset_dist, and power.
    
    best_h = None
    best_score = -1
    
    # Initial guess: standard prize/distance
    def get_heuristic(offset_prize, offset_dist, power):
        d = distance_matrix + offset_dist
        p = prizes[None, :] + offset_prize
        # Ensure no division by zero or negative bases
        d = np.maximum(d, eps)
        p = np.maximum(p, eps)
        h = p / (d ** power)
        return h
    
    # Candidate parameters to test
    # We'll do a coarse grid search followed by local refinement
    # Parameters: offset_prize in [0, 0.5], offset_dist in [0.01, 0.5], power in [0.5, 2.0]
    
    # 1. Coarse Grid Search
    op_grid = [0.0, 0.1, 0.2, 0.5]
    od_grid = [0.01, 0.05, 0.1, 0.2, 0.5]
    pw_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # We can't test all 4*5*5=100 combinations with full ACO if it's slow, 
    # but 50 nodes, 20 ants, 30 iters is fast (~1ms per sim). 100 sims is fine.
    
    for op in op_grid:
        for od in od_grid:
            for pw in pw_grid:
                if time.time() > deadline:
                    break
                
                h = get_heuristic(op, od, pw)
                h = np.where(h > 0, h, 1e-9)
                
                # Use a subset of ants/iters for quick screening? 
                # No, let's use full spec for accuracy, it's fast enough.
                score = _simulate_aco(prizes, distance_matrix, max_len, h)
                
                if score > best_score:
                    best_score = score
                    best_h = h
                    best_params = (op, od, pw)
        
        if time.time() > deadline:
            break
    
    if best_h is None:
        # Fallback
        best_h = get_heuristic(0.0, 0.01, 1.0)
        best_h = np.where(best_h > 0, best_h, 1e-9)
        return best_h
    
    # 2. Local Refinement (Hill Climbing) around best_params
    op, od, pw = best_params
    
    # Define a step size
    def try_step(current_params, step_op, step_od, step_pw):
        new_op = max(0.0, current_params[0] + step_op)
        new_od = max(0.001, current_params[1] + step_od)
        new_pw = max(0.1, current_params[2] + step_pw)
        
        h = get_heuristic(new_op, new_od, new_pw)
        h = np.where(h > 0, h, 1e-9)
        score = _simulate_aco(prizes, distance_matrix, max_len, h)
        
        return score, (new_op, new_od, new_pw)
    
    # Perform a few steps of hill climbing
    for _ in range(50): # Max 50 steps
        if time.time() > deadline:
            break
        
        improved = False
        # Try small perturbations
        for step in [0.01, -0.01, 0.05, -0.05]:
            # Perturb op
            s, new_p = try_step(best_params, step, 0, 0)
            if s > best_score:
                best_score = s
                best_params = new_p
                h = get_heuristic(*best_params)
                best_h = np.where(h > 0, h, 1e-9)
                improved = True
                break
                
            # Perturb od
            s, new_p = try_step(best_params, 0, step, 0)
            if s > best_score:
                best_score = s
                best_params = new_p
                h = get_heuristic(*best_params)
                best_h = np.where(h > 0, h, 1e-9)
                improved = True
                break
                
            # Perturb pw
            s, new_p = try_step(best_params, 0, 0, step)
            if s > best_score:
                best_score = s
                best_params = new_p
                h = get_heuristic(*best_params)
                best_h = np.where(h > 0, h, 1e-9)
                improved = True
                break
        
        if not improved:
            break
            
    return best_h
