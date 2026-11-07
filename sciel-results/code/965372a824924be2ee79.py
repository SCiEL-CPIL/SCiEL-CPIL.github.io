import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    Combines inverse distance with a savings-based multiplier and demand-aware
    adjustment to guide ants towards edges that complete routes efficiently.
    """
    # Ensure inputs are numpy arrays of type float
    d = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = d.shape[0]
    
    # 1. Base Distance Heuristic: 1 / (d^2 + eps)
    # We use a softened denominator (1e-4) to prevent extreme values for very close points,
    # which can cause premature convergence in ACO.
    eps = 1e-4
    dist_heur = 1.0 / (d**2 + eps)
    
    # 2. Savings Heuristic (Clarke-Wright style)
    # S[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # This measures how much distance is saved by visiting i and j consecutively 
    # instead of separately from the depot.
    d_i0 = d[:, 0].reshape(-1, 1)  # (n, 1)
    d_0j = d[0, :].reshape(1, -1)  # (1, n)
    
    S = d_i0 + d_0j - d
    
    # Clip negative savings to 0 (shouldn't happen with Euclidean, but safe)
    S = np.maximum(S, 0)
    
    # Normalize savings to [0, 1] to act as a multiplier
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
    
    # 4. Combine Heuristics
    # eta[i, j] = dist_heur[i, j] * (1 + lambda * S_norm[i, j]) * demand_mult[j]
    # The savings term boosts attractiveness for edges that are "good" for route construction.
    # Lambda controls the strength of this boost.
    lambda_s = 1.5
    
    savings_mult = 1.0 + lambda_s * S_norm
    
    heuristic = dist_heur * savings_mult * demand_mult
    
    # 5. Handle Diagonal and Invalid Values
    # The diagonal represents self-loops, which are invalid moves.
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # We explicitly set diagonal to 0 first, then apply the mask.
    np.fill_diagonal(heuristic, 0.0)
    
    # Mask for non-finite or non-positive values
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
