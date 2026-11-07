"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    
    The heuristic uses a combination of:
    1. Prize density: prize[j] / (d[i,j] + d[j,0])
    2. Neighborhood potential: sum of prizes of feasible neighbors of j
       weighted by proximity, to reward edges leading to high-value clusters
    3. Soft feasibility: a smooth penalty based on how close the path is to max_len
    
    eta[i, j] = (prize[j] + eps) / (d[i,j] + d[j,0] + eps) * (1 + alpha * neighbor_potential[j])
    
    where neighbor_potential[j] is a measure of how many high-prize nodes are
    reachable from j within the remaining budget, using a soft (exponential)
    kernel instead of a hard cutoff.
    """
    n = len(prizes)
    eps = 1e-9
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s limit
    
    # Basic prize/cost ratio
    prize_col = prizes[None, :] + eps  # (1, n)
    dist_to_origin = distance_matrix[:, 0][None, :] + eps  # (1, n)
    cost_matrix = distance_matrix + dist_to_origin  # (n, n)
    
    # Base heuristic: prize / cost
    base_eta = prize_col / cost_matrix  # (n, n)
    
    # Compute neighborhood potential for each node j
    # For node j, look at all nodes k (k != j) that are "close" to j
    # and have high prizes. Use a soft exponential kernel:
    #   potential[j] = sum_k prize[k] * exp(-d[j,k] / sigma)
    # where sigma is a scale parameter. This rewards nodes surrounded by
    # high-prize neighbors.
    
    # Choose sigma based on the problem scale. With 50 points in unit square,
    # typical distances are ~0.1-0.3. Use sigma ~ 0.15.
    sigma = 0.15
    
    # distance_matrix has shape (n, n). Compute exp(-d[j,k]/sigma) for all j,k
    # but exclude self (k=j) by zeroing the diagonal
    kernel = np.exp(-distance_matrix / sigma)  # (n, n)
    np.fill_diagonal(kernel, 0.0)
    
    # neighbor_potential[j] = sum_k kernel[j,k] * prize[k]
    neighbor_potential = kernel @ prizes  # (n,)
    
    # Normalize neighbor_potential to [0, 1] range for stable multiplication
    np_max = neighbor_potential.max()
    if np_max > eps:
        neighbor_potential_norm = neighbor_potential / np_max
    else:
        neighbor_potential_norm = np.zeros(n)
    
    # Blend: use alpha to control how much neighborhood info matters
    # alpha in [0, 1]. Larger alpha = more weight on neighborhood.
    alpha = 0.5
    
    # The multiplier for each column j
    col_multiplier = 1.0 + alpha * neighbor_potential_norm  # (n,)
    
    # Apply multiplier: eta[i,j] *= col_multiplier[j]
    eta = base_eta * col_multiplier[None, :]  # (n, n)
    
    # Soft feasibility penalty: penalize edges where d[i,j] + d[j,0] is close to max_len
    # remaining_budget[i,j] = max_len - (d[i,j] + d[j,0])
    # If remaining_budget <= 0, the edge is infeasible (but ACO handles that)
    # Use a smooth penalty: penalty = exp(-beta * remaining_budget) when budget is tight
    # Actually, we want to FAVOR edges with more remaining budget.
    # Let's use: feasibility_factor = 1 / (1 + exp(-gamma * (max_len - cost_matrix)))
    # This is a sigmoid that is ~1 when cost << max_len and ~0.5 when cost = max_len
    # But since ACO already filters infeasible edges, we just need to slightly
    # prefer edges that leave more budget.
    
    # Simpler: multiply by a factor that increases with remaining budget
    # remaining = max_len - cost_matrix
    # factor = 1 + delta * remaining (clamped to be positive)
    remaining = max_len - cost_matrix  # (n, n)
    # Clip to [0, max_len] for stability
    remaining_clipped = np.clip(remaining, 0, max_len)
    
    # Scale remaining to [0, 1]
    if max_len > eps:
        remaining_norm = remaining_clipped / max_len
    else:
        remaining_norm = np.zeros_like(remaining_clipped)
    
    # Apply a gentle boost for edges with more remaining budget
    beta = 0.3
    feasibility_boost = 1.0 + beta * remaining_norm  # (n, n)
    
    eta = eta * feasibility_boost
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Verify shape
    if eta.shape != (n, n):
        eta = np.ones((n, n)) * 1e-9
    
    return eta
# EVOLVE-BLOCK-END
