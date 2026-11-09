import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    d = np.array(distance_matrix, dtype=float)
    demands = np.array(demands, dtype=float)
    n = d.shape[0]
    
    # Base distance heuristic with power 5 (sharper than 4, which scored better than 3.5)
    heuristic = np.zeros_like(d)
    mask = d > 0
    heuristic[mask] = 1.0 / (d[mask] ** 5)
    
    # Gentle demand-aware boost: favor destinations with higher demand
    # This helps ants pack trucks more efficiently by visiting high-demand nodes
    # The boost is subtle to keep distance heuristic dominant
    demand_boost = 1.0 + 0.1 * (demands / capacity)
    # demand_boost shape: (n,), broadcast along rows (each column j gets boosted by demands[j])
    heuristic *= demand_boost[np.newaxis, :]
    
    # Ensure all values are positive (non-positive becomes 1e-9 per contract)
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
