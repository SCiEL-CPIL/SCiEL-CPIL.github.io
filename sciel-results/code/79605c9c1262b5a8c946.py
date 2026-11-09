import numpy as np
import time

def _run_aco_vectorized(prizes, dist, max_len, heuristic, seed):
    """
    Run ACO with vectorized feasibility and probability checks.
    Returns the best total prize found.
    """
    n = len(prizes)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix
    pheromone = np.ones((n, n), dtype=float)
    # Heuristic matrix, ensuring positive values
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pre-compute distances to origin for quick feasibility checks
    # dist_to_origin[j] = dist[j, 0]
    dist_to_origin = dist[:, 0]
    
    best_overall = 0.0
    
    for it in range(30):
        best_iter = 0.0
        
        for ant in range(20):
            # State
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            current = 0
            length = 0.0
            total_prize = 0.0
            
            while True:
                # Vectorized feasibility check
                # Condition: not visited AND length + dist[current, j] + dist[j, 0] <= max_len
                feasible_mask = (~visited_mask) & (length + dist[current, :] + dist_to_origin <= max_len)
                
                # Get indices of feasible nodes
                feasible_indices = np.where(feasible_mask)[0]
                
                if len(feasible_indices) == 0:
                    break
                
                # Calculate probabilities
                # p[j] = pheromone[current, j] * heur[current, j]
                # We only care about feasible_indices
                probs = pheromone[current, feasible_indices] * heur[current, feasible_indices]
                s = probs.sum()
                if s <= 0:
                    break
                probs = probs / s
                
                # Select next node
                j = int(rng.choice(feasible_indices, p=probs))
                
                # Update state
                length += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j
            
            if total_prize > best_iter:
                best_iter = total_prize
            if total_prize > best_overall:
                best_overall = total_prize
        
        # Pheromone update
        # pheromone = pheromone * 0.9 + best_iter / 10.0
        pheromone *= 0.9
        pheromone += best_iter / 10.0
        
    return best_overall

def _make_heuristic(prizes, dist, max_len, alpha, beta, gamma):
    """
    Generate heuristic matrix based on parameters.
    H_ij = (Prize_j + eps)^gamma / (dist_ij + beta * dist_j0 + eps)^alpha
    """
    eps = 1e-9
    p = prizes[None, :].astype(float)
    d = dist.astype(float)
    d_ret = dist[:, 0][None, :].astype(float) # shape (1, N), broadcast to (N, N)
    
    numer = (p + eps) ** gamma
    denom = (d + beta * d_ret + eps) ** alpha
    
    H = numer / denom
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def _seed_heuristic(prizes, dist):
    """
    Fallback heuristic: Prize / Dist
    """
    H = (prizes[None, :] + 1e-9) / (dist + 1e-9)
    return np.where(np.isfinite(H) & (H > 0), H, 1e-9)

def heuristics(prizes, distance_matrix, max_len):
    try:
        start_time = time.time()
        deadline = start_time + 290.0 # 10s buffer
        
        prizes = np.asarray(prizes, dtype=float)
        dist = np.asarray(distance_matrix, dtype=float)
        n = len(prizes)
        
        # Define candidate parameters
        # alpha: distance exponent
        # beta: return distance weight
        # gamma: prize exponent
        candidates = [
            (1.0, 0.0, 1.0), # Baseline-ish
            (1.0, 0.5, 1.0), # Moderate return weight
            (1.0, 1.0, 1.0), # Full return weight
            (1.5, 0.5, 1.0), # Stronger distance penalty
            (2.0, 1.0, 1.0), # Strong distance penalty, full return
            (1.0, 0.0, 1.5), # Prize bias
            (1.5, 0.5, 1.5), # Combined
            (2.0, 0.5, 1.0), # 
            (1.5, 1.0, 1.0), #
        ]
        
        num_seeds = 3
        best_score = -1.0
        best_heuristic = None
        best_params = None
        
        # Evaluate candidates
        for alpha, beta, gamma in candidates:
            if time.time() > deadline:
                break
                
            H = _make_heuristic(prizes, dist, max_len, alpha, beta, gamma)
            scores = []
            
            # Run ACO with a few seeds to estimate quality
            for s in range(num_seeds):
                if time.time() > deadline:
                    break
                score = _run_aco_vectorized(prizes, dist, max_len, H, s)
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_heuristic = H
                    best_params = (alpha, beta, gamma)
        
        # If we found a good heuristic, return it
        if best_heuristic is not None:
            # Final safety check for validity
            best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
            return best_heuristic
        
        # Fallback to seed heuristic if no time or no candidates evaluated
        return _seed_heuristic(prizes, dist)
        
    except Exception:
        # Robust fallback
        try:
            prizes = np.asarray(prizes, dtype=float)
            dist = np.asarray(distance_matrix, dtype=float)
            return _seed_heuristic(prizes, dist)
        except Exception:
            try:
                n = len(prizes)
            except Exception:
                n = 50
            return np.ones((n, n))

# EVOLVE-BLOCK-START
# The code above is the complete program. The EVOLVE-BLOCK markers are typically used by the framework.
# Since the prompt asks for the COMPLETE program in a single code block, and the current program 
# has the function inside the block, I will structure the output to match the expected format 
# where the main logic is inside the block or the whole file is the block. 
# Given the instruction "Keep the function names and signatures of the seed program", 
# and the seed program had the function inside EVOLVE-BLOCK-START/END, I will place 
# the entire implementation inside these markers to be safe, or just provide the full code.
# The prompt says "Output the COMPLETE program in a single
