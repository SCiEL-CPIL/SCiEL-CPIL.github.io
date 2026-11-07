"""Heuristic for ACO-CVRP: Multi-Objective Structural Heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    This heuristic combines multiple structural signals:
    1. Savings-based: edges that reduce total route length by merging routes
    2. Proximity-based: short edges are preferred
    3. Demand-aware: edges between low-demand nodes are slightly preferred
    4. Depot-distance: edges far from depot are penalized for customer-customer links
    
    The heuristic is designed to guide the ACO towards compact, efficient routes.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h
    
    # If time is running out, return a safe default
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        h = 1.0 / (d + 1e-9)
        np.fill_diagonal(h, 0.0)
        return h
    
    eps = 1e-9
    
    # === Component 1: Savings Heuristic ===
    # savings[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # High savings means visiting i and j consecutively saves distance
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - dm
    savings = np.maximum(savings, 0.0)
    
    # === Component 2: Inverse Distance ===
    # Short edges are preferred
    d_inv = 1.0 / (dm + eps)
    
    # === Component 3: Demand-Based Attractiveness ===
    # Nodes with lower demand are more "flexible" - edges between them are slightly preferred
    # Normalize demands to [0, 1]
    D_max = np.max(D)
    if D_max > 0:
        D_norm = D / D_max
    else:
        D_norm = np.zeros_like(D)
    
    # Edge demand score: 1 - (demand_i + demand_j) / (2 * cap)
    # This gives higher scores to edges between low-demand nodes
    demand_score = 1.0 - (D_norm[:, np.newaxis] + D_norm[np.newaxis, :]) / 2.0
    demand_score = np.maximum(demand_score, 0.1)  # Ensure positive
    
    # === Component 4: Depot Distance Penalty for Customer-Customer Edges ===
    # Edges far from the depot create long detours; penalize them
    # For customer-customer edges, use average distance to depot
    depot_dist = dm[:, 0]  # distance of each node to depot
    # For edge (i,j), penalize if both i and j are far from depot
    depot_penalty = 1.0 / (1.0 + (depot_dist[:, np.newaxis] + depot_dist[np.newaxis, :]) / (2.0 * np.max(depot_dist) + eps))
    
    # === Assemble the Heuristic ===
    h = np.ones((n, n))
    
    # For customer-customer edges (i > 0, j > 0)
    d_cc = dm[1:, 1:].copy()
    np.fill_diagonal(d_cc, 1e9)
    savings_cc = savings[1:, 1:]
    d_inv_cc = d_inv[1:, 1:]
    demand_score_cc = demand_score[1:, 1:]
    depot_penalty_cc = depot_penalty[1:, 1:]
    
    # Combine components with weighted product
    # h_cc = savings_cc^a * d_inv_cc^b * demand_score_cc^c * depot_penalty_cc^d
    # Use moderate exponents to balance the components
    a = 0.5   # savings weight
    b = 0.8   # proximity weight
    c = 0.3   # demand weight
    d_exp = 0.4  # depot penalty weight
    
    h_cc = (savings_cc + eps) ** a * d_inv_cc ** b * demand_score_cc ** c * depot_penalty_cc ** d_exp
    
    # For depot-customer edges (i = 0 or j = 0)
    d_depot = dm[0, 1:].copy()
    d_inv_depot = 1.0 / (d_depot + eps)
    demand_score_depot = 1.0 - D_norm[1:]  # demand score for depot edges
    
    # Depot edges: primarily proximity, with slight demand preference
    h_depot = d_inv_depot ** 1.0 * demand_score_depot ** 0.5
    
    # Assemble the full heuristic matrix
    h[0, 1:] = h_depot
    h[1:, 0] = h_depot
    h[1:, 1:] = h_cc
    
    # Ensure all values are positive
    h = np.where(h > 0, h, eps)
    
    # Normalize to keep values in a reasonable range
    max_h = np.max(h)
    if max_h > eps:
        h = h / max_h
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(h, 0.0)
    
    # Final check: ensure all off-diagonal entries are positive
    h = np.where(h > 0, h, eps)
    np.fill_diagonal(h, 0.0)
    
    return h
# EVOLVE-BLOCK-END
