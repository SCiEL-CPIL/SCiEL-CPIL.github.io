import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an adaptive heuristic matrix for ACO by evaluating 
    a family of heuristics using a fast internal simulation.
    """
    start_time = time.time()
    DEADLINE = 300.0  # Seconds
    
    prizes = np.array(prizes, dtype=float)
    dm = np.array(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Precompute return distances from any node to 0
    ret_dists = dm[:, 0]
    
    # Candidate parameters
    # Alpha: prize exponent
    # Beta: distance exponent
    # Lambda: weight of return distance in the cost term
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5]
    betas = [0.5, 1.0, 1.5, 2.0, 2.5]
    lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    best_score = -1
    best_alpha = 1.0
    best_beta = 1.0
    best_lambda = 0.5
    best_eta = None
    
    # Fast internal ACO evaluation
    def mini_aco_eval(alpha, beta, lam, rng_seed=42):
        # Compute heuristic matrix
        # eta_ij = prize_j^alpha / (d_ij + lam * d_j0)^beta
        # Note: d_j0 is the same for all i, so we can precompute the denominator part for j
        # denom_ij = (dm[i, j] + lam * ret_dists[j])^beta
        
        # Vectorized calculation for the whole matrix
        # ret_dists[None, :] is shape (1, n), dm is (n, n)
        # Broadcasting: dm + lam * ret_dists[None, :] works
        cost_term = dm + lam * ret_dists[None, :]
        # Ensure no negative costs (should be non-negative by geometry)
        cost_term = np.maximum(cost_term, 1e-9)
        
        denom = cost_term ** beta
        # Avoid division by zero
        denom[denom == 0] = 1e-9
        
        # Prize term: prizes[None, :] ** alpha
        # If prize is 0, 0**alpha is 0. 
        # If alpha < 1, 0**alpha is 0.
        # We want eta to be positive. The problem says "Heuristic entries that are not positive become 1e-9".
        # So we compute, then clamp.
        
        prize_term = prizes[None, :] ** alpha
        
        eta = prize_term / denom
        
        # Enforce positivity as per problem statement
        eta[eta <= 0] = 1e-9
        # Ensure finite
        eta[~np.isfinite(eta)] = 1e-9
        
        # Run Mini ACO
        num_ants = 5
        num_iters = 5
        
        # Pheromone matrix, initialized to 1
        tau = np.ones((n, n))
        
        best_prize_found = 0
        
        # Use a local RNG to ensure reproducibility for the same parameters
        rng = np.random.RandomState(rng_seed)
        
        for _ in range(num_iters):
            iter_best_prize = 0
            
            for _ in range(num_ants):
                # Ant state
                current_node = 0
                current_len = 0
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                total_prize = 0.0
                
                while True:
                    # Find eligible neighbors
                    # Vectorized check for eligibility
                    # Condition: not visited AND current_len + d[i, j] + d[j, 0] <= max_len
                    
                    # d[i, :] is dm[current_node, :]
                    # d[j, 0] is ret_dists
                    # required_len = current_len + dm[current_node, :] + ret_dists
                    required_len = current_len + dm[current_node, :] + ret_dists
                    
                    eligible_mask = (~visited) & (required_len <= max_len)
                    
                    if not np.any(eligible_mask):
                        break
                    
                    eligible_indices = np.where(eligible_mask)[0]
                    
                    # Calculate probabilities
                    # prob_ij proportional to tau[i, j] * eta[i, j]
                    # We only care about eligible nodes
                    probs = tau[current_node, eligible_indices] * eta[current_node, eligible_indices]
                    
                    total_prob = np.sum(probs)
                    if total_prob == 0:
                        # Fallback to random
                        j = eligible_indices[rng.randint(len(eligible_indices))]
                    else:
                        probs = probs / total_prob
                        # Roulette wheel selection
                        r = rng.random()
                        cumsum = 0
                        j = eligible_indices[-1]
                        for k, p in enumerate(probs):
                            cumsum += p
                            if r <= cumsum:
                                j = eligible_indices[k]
                                break
                    
                    # Move to j
                    current_len += dm[current_node, j]
                    total_prize += prizes[j]
                    visited[j] = True
                    current_node = j
                    
                if total_prize > iter_best_prize:
                    iter_best_prize = total_prize
                
            # Update pheromone
            tau *= 0.9
            tau += iter_best_prize / 10.0
            
            if iter_best_prize > best_prize_found:
                best_prize_found = iter_best_prize
                
        return best_prize_found, eta

    # Grid search
    # We might not need to store best_eta for every step, just the best params
    # But calculating eta is cheap.
    
    total_combos = len(alphas) * len(betas) * len(lambdas)
    current_combo = 0
    
    for a in alphas:
        for b in betas:
            for l in lambdas:
                current_combo += 1
                # Check deadline
                if time.time() > DEADLINE - 10:
                    break
                
                score, eta_candidate = mini_aco_eval(a, b, l)
                if score > best_score:
                    best_score = score
                    best_alpha = a
                    best_beta = b
                    best_lambda = l
                    best_eta = eta_candidate
            if time.time() > DEADLINE - 10:
                break
        if time.time() > DEADLINE - 10:
            break
            
    # If the grid search didn't find a valid eta (e.g. due to timeout or error),
    # construct a default one based on the best params found so far, or a sensible default.
    if best_eta is None:
        # Default heuristic: Prize / Distance
        cost_term = dm + best_lambda * ret_dists[None, :]
        cost_term = np.maximum(cost_term, 1e-9)
        denom = cost_term ** best_beta
        denom[denom == 0] = 1e-9
        prize_term = prizes[None, :] ** best_alpha
        best_eta = prize_term / denom
        best_eta[best_eta <= 0] = 1e-9
        best_eta[~np.isfinite(best_eta)] = 1e-9
        
    # Final sanity check
    if not np.all(np.isfinite(best_eta)):
        best_eta = np.where(np.isfinite(best_eta), best_eta, 1e-9)
    if not np.all(best_eta > 0):
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)
        
    return best_eta
