"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO.
    Uses a Prize Density Field approach combined with return cost penalty.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, 10s margin
    
    n = len(prizes)
    d = distance_matrix
    
    # 1. Compute Prize Density Field V[j]
    # V[j] represents the sum of prizes weighted by inverse distance from j.
    # This identifies "clusters" of high-value nodes.
    # We use a soft kernel 1 / (1 + d[j,k]) to smooth the field.
    
    # Vectorized computation of V
    # d is (n, n). prizes is (n,).
    # We want V[j] = sum_k prize[k] / (1 + d[j,k])
    # Note: d[j,k] is distance from j to k.
    
    kernel = 1.0 / (1.0 + d)
    # V[j] is the row sum of kernel * prizes_col
    prizes_col = prizes[:, None] # (n, 1)
    V = kernel @ prizes_col # (n, 1)
    V = V.flatten() # (n,)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple heuristic
        return (prizes[None, :] + 1e-9) / (d + 1e-9)

    # 2. Combine Local Prize and Density
    # Attractiveness of node j is boosted if it has high local prize AND high density.
    # Use prize[j] + V[j]
    attractiveness = prizes + V
    
    # 3. Return Cost Penalty
    # Nodes far from start (node 0) are risky to visit because they consume
    # budget that could be used for other nodes.
    # Penalty factor: 1 / (1 + d[j, 0])
    d_to_0 = d[:, 0]
    safety_factor = 1.0 / (1.0 + d_to_0)
    
    # Adjusted attractiveness
    adj_attractiveness = attractiveness * safety_factor[None, :] # (1, n) broadcast
    
    # 4. Construct Heuristic Matrix
    # H[i, j] = adj_attractiveness[j] / (d[i, j] + eps)
    eps = 1e-9
    d_safe = d + eps
    
    # We want H[i, j] to be proportional to adj_attractiveness[j] / d[i, j]
    # This is a rank-1 matrix structure: col_vec * row_vec? No.
    # H[i, j] = (adj_attractiveness[j]) / d[i, j]
    # This can be computed as:
    # H = np.divide(adj_attractiveness[None, :], d_safe)
    
    H = adj_attractiveness[None, :] / d_safe
    
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Optional: Normalize rows to prevent magnitude issues?
    # ACO probability is proportional to pheromone * heuristic.
    # Pheromone starts at 1.
    # If heuristics are very large, pheromone update might be affected?
    # Pheromone update: tau = tau * 0.9 + delta.
    # The probability selection is local, so scale doesn't matter much,
    # but relative scale within a row matters.
    
    # Check time again
    if time.time() > deadline:
        return H

    return H
# EVOLVE-BLOCK-END
