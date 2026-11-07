"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Build an edge-attractiveness matrix that anticipates ACO pheromone convergence
    using a savings-based route-construction heuristic.
    
    Strategy: Combine three signals that encode route quality:
    1. Inverse distance squared: strong bias toward short edges
    2. Clarke-Wright savings: edges that merge two depot-trips save distance
    3. Demand-aware adjustment: penalize edges to customers with high demand
       relative to remaining capacity (approximated statically)
    
    The savings term is the key structural element: it directly measures how
    much distance is saved by visiting i->j consecutively vs. separately from
    the depot. This is the same signal that a successful ACO colony will
    reinforce through pheromone, so encoding it in the static heuristic
    guides early ants toward edges that will be reinforced.
    """
    n = distance_matrix.shape[0]
    d = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    capacity = float(capacity)
    
    # Ensure finite distances
    d = np.where(np.isfinite(d), d, 1e6)
    d = np.maximum(d, 1e-9)
    
    # 1. Base distance heuristic: 1 / (d^2 + eps)
    # Squared distance creates stronger discrimination between close and far edges
    eps = 1e-4
    dist_heur = 1.0 / (d**2 + eps)
    
    # 2. Clarke-Wright savings: S[i,j] = d[i,0] + d[0,j] - d[i,j]
    # This measures distance saved by visiting i and j consecutively
    # instead of making two separate trips from the depot
    d_i0 = d[:, 0].reshape(-1, 1)  # (n, 1)
    d_0j = d[0, :].reshape(1, -1)  # (1, n)
    
    S = d_i0 + d_0j - d
    
    # Clip negative savings (shouldn't happen with Euclidean, but safe)
    S = np.maximum(S, 0.0)
    
    # Normalize savings to [0, 1]
    S_max = np.max(S)
    if S_max > 0:
        S_norm = S / S_max
    else:
        S_norm = np.zeros_like(S)
    
    # 3. Demand-aware adjustment:
    # Edges to customers with high demand are less attractive because they
    # consume more of the vehicle capacity, potentially forcing earlier returns.
    # We use a soft penalty: multiplier decreases with demand_j / capacity
    # This is static (doesn't depend on current route state) but approximates
    # the expected remaining capacity when arriving at customer j.
    
    # Average remaining capacity when visiting a customer is roughly capacity/2
    # (since we start full and end empty). So the "risk" of a high-demand customer
    # is higher when average remaining is low.
    # Use: demand_penalty = 1 / (1 + gamma * demand_j / capacity)
    # Higher demand_j -> lower multiplier
    gamma = 0.5
    demand_penalty = 1.0 / (1.0 + gamma * (demands / capacity))
    demand_penalty[0] = 1.0  # depot has no demand penalty
    
    # Apply demand penalty as a row-independent multiplier (depends only on j)
    # heuristic[i,j] *= demand_penalty[j]
    demand_mult = demand_penalty[np.newaxis, :]  # (1, n)
    
    # 4. Combine all components
    # eta[i,j] = dist_heur[i,j] * (1 + lambda_s * S_norm[i,j]) * demand_mult[j]
    
    lambda_s = 1.5  # Strength of savings boost
    
    savings_mult = 1.0 + lambda_s * S_norm
    
    heuristic = dist_heur * savings_mult * demand_mult
    
    # 5. Handle diagonal and invalid values
    # Diagonal: self-loops are invalid, set to 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are finite and positive (or 1e-9 as per contract)
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
