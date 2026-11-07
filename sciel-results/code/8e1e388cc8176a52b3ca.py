import numpy as np
import time
import random

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
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
                d_from_current = distance_matrix[current_node, :]
                valid_indices = np.where(
                    (~visited_mask) & 
                    (d_from_current + d_to_start <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                h_vals = H[current_node, valid_indices]
                
                total_h = np.sum(h_vals)
                if total_h <= 0:
                    break
                    
                probs = h_vals / total_h
                
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
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
    
    # 1. Initialize heuristic with a strong parametric prior
    # This provides a good starting point for the non-parametric optimizer
    d_to_start = distance_matrix[:, 0]
    # Prior: h[i, j] = (prize[j] + 0.1) / (d[i, j] + 0.05)
    # We want to encourage visiting high prize nodes that are close to current node
    # and also close to start (to allow return)
    # Actually, the constraint is d[i,j] + d[j,0] <= remaining.
    # A good heuristic should reflect "efficiency" of prize per "cost" (distance).
    
    # Let's start with a simple prize/distance heuristic
    # h[i,j] = prizes[j] / (distance_matrix[i,j] + 0.01)
    # But we must ensure it's a valid matrix.
    
    # Let's use a slightly more informed initialization
    # h[i,j] = prizes[j] / (distance_matrix[i,j] + d_to_start[j] + 0.01)
    # This penalizes nodes that are far from current AND far from start.
    
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 1e-9
            else:
                cost = distance_matrix[i, j] + d_to_start[j]
                H[i, j] = (prizes[j] + 0.01) / (cost + 0.01)
    
    # Ensure positive
    H = np.maximum(H, eps)
    
    best_score = _simulate_aco(prizes, distance_matrix, max_len, H)
    
    # 2. Coordinate Ascent Optimization
    # We will iteratively pick a random entry (i, j) and try to adjust it to improve the score.
    # This is a form of stochastic coordinate ascent.
    
    rng_opt = np.random.default_rng(123)
    
    # Pre-compute valid pairs (i, j) where j != 0 (since we start at 0, and 0 is not visited again)
    # Actually, the heuristic is used for transitions i -> j.
    # i can be 0. j cannot be 0 (since 0 is visited at start).
    # Also, the ACO only uses H[i, j] if j is valid.
    # We can optimize all i, j where j != 0.
    
    valid_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and j != 0:
                valid_pairs.append((i, j))
    
    # We will perform a fixed number of steps or until time limit
    # Each step: pick a random pair, try several perturbations, keep best.
    
    # To speed up, we can batch the perturbation evaluation?
    # No, the score depends on the whole matrix.
    # However, changing one entry only affects the probabilities of ants that are at node i and considering j.
    # But the RNG sequence is fixed. Changing H[i, j] changes the probability vector, which changes the random number used?
    # No, rng.choice uses the random state. If the probabilities change, the *interpretation* of the random number changes,
    # but the random number generated is the same.
    # Wait, numpy rng.choice with p=probs generates a uniform random number and bins it.
    # If p changes, the binning changes, so the next node chosen might change.
    # This means a change in H[i, j] can cascade through the entire path of an ant if that ant visits i and considers j.
    # So we cannot easily localize the evaluation. We must re-run the simulation.
    
    # To make this efficient, we need the simulation to be fast.
    # Current simulation: 20 ants * 30 iters * ~5 steps * numpy ops.
    # It should be reasonably fast.
    
    num_steps = 5000  # Target number of coordinate updates
    
    for step in range(num_steps):
        if time.time() > deadline - 5:
            break
        
        # Pick a random pair
        idx = rng_opt.integers(0, len(valid_pairs))
        i, j = valid_pairs[idx]
        
        # Current value
        val = H[i, j]
        
        # Try a few perturbations
        # Multiplicative perturbation is good for positive values
        # Try: val * 0.5, val * 0.8, val * 1.2, val * 2.0, val * 5.0, val * 0.1
        # And maybe some absolute shifts? No, multiplicative is safer for scale invariance.
        
        multipliers = [0.5, 0.8, 1.25, 2.0, 0.2, 5.0, 0.05, 10.0]
        
        best_mult = 1.0
        best_val = val
        best_local_score = best_score
        
        # To save time, we can just try 3-4 random multipliers per step
        # Or a fixed set of small perturbations
        # Let's try a random multiplier from a log-uniform distribution
        # log_mult = rng_opt.uniform(-2, 2)
        # mult = np.exp(log_mult)
        
        # Let's try a fixed set of diverse multipliers to explore the space
        # But we only have limited time.
        # Let's try 3 random multipliers per step.
        
        for _ in range(3):
            log_mult = rng_opt.uniform(-1.5, 1.5)
            mult = np.exp(log_mult)
            new_val = val * mult
            
            # Clamp to reasonable range
            if new_val < 1e-6:
                new_val = 1e-6
            elif new_val > 1e6:
                new_val = 1e6
                
            H[i, j] = new_val
            
            score = _simulate_aco(prizes, distance_matrix, max_len, H)
            
            if score > best_local_score:
                best_local_score = score
                best_mult = mult
                best_val = new_val
            else:
                # Revert if not better?
                # We will revert at the end of the loop for this pair
                pass
        
        # If we found a better value, keep it. Otherwise revert to original.
        if best_mult != 1.0:
            H[i, j] = best_val
            best_score = best_local_score
        else:
            H[i, j] = val
            
    # Final check
    # Ensure no NaNs or Infs
    if not np.all(np.isfinite(H)):
        # Fallback to initial heuristic
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, j] = 1e-9
                else:
                    cost = distance_matrix[i, j] + d_to_start[j]
                    H[i, j] = (prizes[j] + 0.01) / (cost + 0.01)
        H = np.maximum(H, eps)
        
    return H

