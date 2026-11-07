import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an adaptive heuristic matrix for ACO by evaluating 
    a family of heuristics using a fast internal simulation.
    """
    n = len(prizes)
    # Ensure prizes[0] is 0 as per problem description, but safe to handle generally
    prizes = np.array(prizes, dtype=float)
    dm = np.array(distance_matrix, dtype=float)
    
    # Precompute return distances from any node to 0
    ret_dists = dm[:, 0]
    
    # Candidate parameters: alpha (prize power), beta (distance power), lambda (return dist weight)
    # We want to explore a range of trade-offs
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.5, 1.0, 1.5, 2.0]
    lambdas = [0.0, 0.5, 1.0, 1.5]
    
    best_score = -1
    best_alpha = 1.0
    best_beta = 1.0
    best_lambda = 0.5
    
    # Fast internal ACO evaluation
    def mini_aco_eval(alpha, beta, lam):
        # Compute heuristic matrix
        # eta_ij = prize_j^alpha / (d_ij + lam * d_j0)^beta
        # Handle division by zero and negative values (though d>=0)
        denom = (dm + lam * ret_dists[None, :]) ** beta
        # Add small epsilon to avoid div by zero
        denom[denom == 0] = 1e-9
        eta = (prizes[None, :] ** alpha) / denom
        
        # Ensure positive
        eta[eta <= 0] = 1e-9
        
        # Run Mini ACO
        num_ants = 5
        num_iters = 5
        
        # Pheromone matrix, initialized to 1
        tau = np.ones((n, n))
        
        best_prize_found = 0
        
        for _ in range(num_iters):
            iter_best_prize = 0
            
            for _ in range(num_ants):
                # Ant state
                current_node = 0
                current_len = 0
                visited = [False] * n
                visited[0] = True
                total_prize = 0
                
                while True:
                    # Find eligible neighbors
                    eligible = []
                    for j in range(n):
                        if not visited[j]:
                            # Check length constraint: current_len + d[i,j] + d[j,0] <= max_len
                            if current_len + dm[current_node, j] + ret_dists[j] <= max_len:
                                eligible.append(j)
                    
                    if not eligible:
                        break
                    
                    # Calculate probabilities
                    # prob_ij proportional to tau[i,j] * eta[i,j]
                    probs = np.zeros(len(eligible))
                    for k, j in enumerate(eligible):
                        probs[k] = tau[current_node, j] * eta[current_node, j]
                    
                    total_prob = np.sum(probs)
                    if total_prob == 0:
                        # Fallback to random if something went wrong
                        j = eligible[np.random.choice(len(eligible))]
                    else:
                        probs = probs / total_prob
                        # Roulette wheel selection
                        r = np.random.random()
                        cumsum = 0
                        j = eligible[-1]
                        for k, p in enumerate(probs):
                            cumsum += p
                            if r <= cumsum:
                                j = eligible[k]
                                break
                    
                    # Move to j
                    current_len += dm[current_node, j]
                    total_prize += prizes[j]
                    visited[j] = True
                    current_node = j
                    
                if total_prize > iter_best_prize:
                    iter_best_prize = total_prize
                
                # Update pheromone for this ant? 
                # The problem says: "After each iteration, pheromone is multiplied by 0.9 and every entry gains (best prize of that iteration) / 10."
                # This implies pheromone update happens per iteration, not per ant step.
                # So we accumulate the best prize of the iteration, and update tau once after all ants finish.
                
            # Update pheromone
            tau *= 0.9
            tau += iter_best_prize / 10.0
            
            if iter_best_prize > best_prize_found:
                best_prize_found = iter_best_prize
                
        return best_prize_found

    # Grid search
    for a in alphas:
        for b in betas:
            for l in lambdas:
                score = mini_aco_eval(a, b, l)
                if score > best_score:
                    best_score = score
                    best_alpha = a
                    best_beta = b
                    best_lambda = l
    
    # Construct final heuristic matrix with best parameters
    denom = (dm + best_lambda * ret_dists[None, :]) ** best_beta
    denom[denom == 0] = 1e-9
    eta = (prizes[None, :] ** best_alpha) / denom
    eta[eta <= 0] = 1e-9
    
    return eta
