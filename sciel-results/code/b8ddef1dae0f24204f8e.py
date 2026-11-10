"""Heuristic matrix for ACO CVRP using depot-relative savings structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy: Depot-relative savings heuristic.
    - For edges from depot (row 0): attractiveness = 1/d(0,j) normalized.
      Shorter distance to depot means lower cost to insert as first stop.
    - For edges between customers (i,j where i,j > 0): attractiveness reflects
      the savings of linking i->j versus serving them independently.
      savings(i,j) = d(0,i) + d(0,j) - d(i,j)
      Higher savings means the edge is more valuable in a route.
      We use 1/(d(i,j) + eps) * (1 + savings_bonus) where savings_bonus
      amplifies edges that create good savings.
    - Normalization: scale so mean off-diagonal is ~1.0 to balance with
      initial pheromone of 1.0.
    """
    deadline = time.time() + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)

    eps = 1e-9

    # Base: inverse distance with exponent 1.0 (standard)
    eta = 1.0 / (d + eps)

    # Depot-relative savings adjustment for customer-to-customer edges
    # d0[i] = distance from depot (node 0) to customer i
    d0 = d[0, :].copy()  # shape (n,)
    d0[0] = 0.0  # depot to itself

    # For each pair (i, j) where both are customers (i>0, j>0, i!=j):
    # savings = d0[i] + d0[j] - d[i,j]
    # If savings > 0, linking i->j is beneficial vs serving separately.
    # We amplify attractiveness by a factor based on normalized savings.
    if n > 1:
        # Compute savings matrix for customer pairs
        d0_col = d0[:, np.newaxis]  # (n, 1)
        d0_row = d0[np.newaxis, :]  # (1, n)
        savings = d0_col + d0_row - d  # (n, n)
        
        # Only apply to customer-customer edges (exclude depot row/col)
        # Create mask: i > 0 and j > 0 and i != j
        mask_cc = np.zeros((n, n), dtype=bool)
        if n > 1:
            mask_cc[1:, 1:] = True
            np.fill_diagonal(mask_cc, False)
        
        # Normalize savings to [0, 1] range for the customer-customer block
        savings_cc = savings[mask_cc]
        if len(savings_cc) > 0:
            s_min = np.min(savings_cc)
            s_max = np.max(savings_cc)
            if s_max > s_min:
                savings_norm = (savings_cc - s_min) / (s_max - s_min)
            else:
                savings_norm = np.zeros_like(savings_cc)
            
            # Reconstruct full normalized savings matrix
            savings_norm_full = np.zeros((n, n), dtype=np.float64)
            savings_norm_full[mask_cc] = savings_norm
            
            # Amplification factor: 1 + alpha * savings_norm
            # alpha controls how strongly savings influence the heuristic
            alpha = 0.5
            amplification = 1.0 + alpha * savings_norm_full
            amplification[~mask_cc] = 1.0  # no change for depot edges
            
            eta = eta * amplification

    # Normalize: mean off-diagonal should be ~1.0
    mask_off = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask_off])

    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        eta = 1.0 / (d + eps)
        mean_eta = np.mean(eta[mask_off])
        if mean_eta > 0 and np.isfinite(mean_eta):
            eta = eta / mean_eta

    # Sanitize
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9

    return eta
# EVOLVE-BLOCK-END
