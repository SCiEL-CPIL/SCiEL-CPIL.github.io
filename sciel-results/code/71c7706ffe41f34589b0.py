"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP using a
    cooling-schedule blend of distance-focused and demand-aware heuristics,
    plus a local-density (cluster) term that smooths the demand penalty.
    
    Args:
        distance_matrix: (n, n) symmetric distance matrix
        coordinates: (n, 2) coordinates
        demands: (n,) demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
        
    Returns:
        (n, n) heuristic matrix where larger = more attractive
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s limit

    n = distance_matrix.shape[0]
    d = distance_matrix.copy().astype(np.float64)
    np.fill_diagonal(d, np.inf)
    
    # Ensure finite distances
    d = np.where(np.isfinite(d), d, 1e10)

    # Precompute useful vectors
    demand = np.asarray(demands, dtype=float).copy()
    demand[0] = 0.0
    demand_ratio = demand / capacity

    # Local density: for each node, sum of 1/dist to its k nearest neighbours
    k = min(8, n - 1)
    d_work = d.copy()
    np.fill_diagonal(d_work, np.inf)
    k_small = np.argpartition(d_work, k, axis=1)[:, :k]
    density = np.zeros(n)
    for i in range(n):
        idx = k_small[i]
        vals = d_work[i, idx]
        density[i] = np.sum(1.0 / vals)
    dmax = density.max() if density.max() > 0 else 1.0
    density_norm = density / dmax

    # Three-phase cooling schedule with vectorized demand penalty
    phases = [
        {'alpha': 3.0, 'demand_weight': 0.0, 'weight': 0.35},  # Early: steep distance
        {'alpha': 2.0, 'demand_weight': 0.3, 'weight': 0.40},  # Mid: balanced
        {'alpha': 1.5, 'demand_weight': 0.6, 'weight': 0.25},  # Late: demand-aware
    ]

    H = np.zeros((n, n))
    
    for phase in phases:
        if time.time() > deadline - 10.0:
            break
            
        alpha = phase['alpha']
        dw = phase['demand_weight']
        w = phase['weight']
        
        # Steep inverse distance: higher alpha = stronger preference for nearby nodes
        with np.errstate(divide='ignore', invalid='ignore'):
            base = 1.0 / (d ** alpha)
        base = np.where(np.isfinite(base), base, 0.0)
        
        # Vectorized demand penalty: penalize edges to high-demand nodes
        # From depot: mild penalty (fresh capacity)
        # From customer: stronger penalty (uncertain remaining capacity)
        depot_penalty = 1.0 - 0.1 * dw * demand_ratio
        customer_penalty = 1.0 - 0.4 * dw * demand_ratio
        
        # Build penalty matrix: row 0 gets depot penalty, others get customer penalty
        penalty_matrix = np.outer(
            np.ones(n) * 0.0,  # placeholder
            np.ones(n)
        )
        # Row 0 (depot): use depot_penalty for all columns
        penalty_matrix[0, :] = depot_penalty
        # Rows 1..n-1 (customers): use customer_penalty for all columns
        penalty_matrix[1:, :] = customer_penalty
        # Clamp to [0.1, 1.0]
        penalty_matrix = np.clip(penalty_matrix, 0.1, 1.0)
        
        # Cluster term: boost edges between nodes with similar high density
        # Only active in early/mid phases to form clusters
        cluster_w = 0.15 * (1.0 - dw)  # stronger early
        if cluster_w > 0.01:
            dens_diff = np.abs(density_norm[:, None] - density_norm[None, :])
            cluster_kernel = np.exp(-dens_diff / 0.5)
            cluster_min = np.minimum(density_norm[:, None], density_norm[None, :])
            cluster_term = cluster_kernel * cluster_min
            # Combine: base * penalty * (1 + cluster_w * cluster_term)
            H_phase = base * penalty_matrix * (1.0 + cluster_w * cluster_term)
        else:
            H_phase = base * penalty_matrix
        
        # Zero out diagonal
        np.fill_diagonal(H_phase, 0.0)
        H_phase = np.where(np.isfinite(H_phase), H_phase, 0.0)
        
        H += w * H_phase
    
    # Normalize to avoid numerical issues
    H_max = np.max(H)
    if H_max > 0:
        H = H / H_max
    
    # Ensure all off-diagonal entries are positive and finite
    np.fill_diagonal(H, 1e-9)
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
