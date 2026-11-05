import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Primary: Lowest Degree.
    2. Secondary: Prefer vertices whose neighbors have high degree, but with a 
       quadratic weighting to emphasize hubs more strongly.
       
    Rationale:
    Selecting a vertex u blocks its neighbors N(u). The "value" of blocking a neighbor v
    is proportional to the number of vertices v would have blocked (i.e., deg(v)).
    However, blocking a "super-hub" is disproportionately valuable because it saves
    a large fraction of the graph from being blocked. 
    Using the sum of squared degrees of neighbors (adj @ (deg**2)) emphasizes 
    high-degree neighbors more than the linear sum, potentially improving the 
    greedy choice in graphs with degree variance.
    
    Score = -(deg(u)) + beta * (sum of deg(v)^2 for v in N(u))
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate sum of squared degrees of neighbors
    # neighbor_deg_sq_sum[i] = sum_{j in N(i)} deg[j]^2
    deg_sq = deg ** 2
    neighbor_deg_sq_sum = adj @ deg_sq
    
    # 3. Combine scores
    # We want to minimize degree (cost of blocking neighbors)
    # and maximize the "potential saved" by blocking neighbors (benefit)
    # The benefit is weighted by the square of the neighbor's degree.
    
    # We need to scale the benefit term to be comparable to the degree term.
    # In ER graphs, deg ~ Poisson(mu).
    # E[deg] = mu
    # E[deg^2] = mu^2 + mu
    # E[neighbor_deg_sq_sum] = deg * (mu^2 + mu) ~ mu^3
    # So the benefit term is O(mu^3) while cost is O(mu).
    # We need a small beta to balance them.
    # Let's estimate mu = n * p. We don't know p, but we can estimate it from deg.
    # mu_est = mean(deg)
    # If mu is 10, cost ~ 10, benefit ~ 1000.
    # We want benefit to be roughly equal to cost.
    # So beta ~ 1/mu^2.
    
    mu_est = np.mean(deg)
    if mu_est > 0:
        beta = 1.0 / (mu_est ** 2)
    else:
        beta = 1.0
        
    # Add a small linear term for neighbor degree sum as a tie-breaker?
    # neighbor_deg_sum = adj @ deg
    # score = -deg + beta * neighbor_deg_sq_sum + gamma * neighbor_deg_sum
    
    # Let's stick to just the quadratic term for distinctness, but add a small linear component
    # to avoid zero-division issues or flat spots.
    neighbor_deg_sum = adj @ deg
    
    score = -deg + beta * neighbor_deg_sq_sum + 0.5 * beta * mu_est * neighbor_deg_sum
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
