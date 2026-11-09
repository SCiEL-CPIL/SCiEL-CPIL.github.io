import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO using a Value-to-Cost Ratio approach.
    This approach calculates the attractiveness of moving to node j from i
    as the prize of j divided by a power of the incremental path cost.
    This contrasts with the Gaussian kernel approach by using a divisive
    penalty for distance rather than a multiplicative exponential decay.
    """
    # prizes shape: (N,)
    # distance_matrix shape: (N, N)
    
    # Extract distances from every node j to node 0
    d_to_start = distance_matrix[:, 0] # Shape (N,)
    
    # Effective cost for moving i -> j
    # cost[i, j] = d[i, j] + d[j, 0]
    effective_cost = distance_matrix + d_to_start[None, :]
    
    # Avoid division by zero (shouldn't happen for valid moves i!=j, 
    # but d[i,i]=0, d[i,0] might be 0 if i=0).
    # We set cost to infinity where it is 0 to make the ratio 0.
    safe_cost = np.where(effective_cost > 0, effective_cost, np.inf)
    
    # Exponent for distance penalty
    # Alpha=1.0 is the standard greedy ratio.
    alpha = 1.0
    
    # Calculate the ratio: prize_j / (cost_ij)^alpha
    heuristic = prizes[None, :] / (safe_cost ** alpha)
    
    # Ensure non-negative and finite.
    # If prize is 0, heuristic is 0.
    # If cost is inf, heuristic is 0.
    # The task says "Heuristic entries that are not positive become 1e-9".
    
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
