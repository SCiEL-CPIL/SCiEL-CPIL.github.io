"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix combining inverse-squared distance,
    savings-based consolidation reward, demand-aware weighting, and angular
    continuity to encourage non-crossing, spatially compact routes.
    
    Key design:
    - Base: 1/d^2 (steep gradient, strong nearest-neighbor preference)
    - Savings component: reward edges that save depot trips
    - Demand weighting: prioritize consolidating high-demand customers
    - Angular continuity: reward edges that maintain smooth route direction
    - Depot edges: tuned to encourage good route starts/ends
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    
    # Base heuristic: inverse squared distance (steeper gradient than 1/d)
    # This creates strong preference for nearest neighbors, forming tight clusters
    base_heuristic = 1.0 / (dm + 1e-9) ** 2
    
    # Compute savings matrix: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    # Positive savings mean connecting i and j saves distance vs. separate trips
    depot = 0
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s = dm[depot, i] + dm[depot, j] - dm[i, j]
                savings[i, j] = s
                savings[j, i] = s
    
    # Normalize savings to [0, 1] range
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = np.clip(savings, 0, max_savings) / max_savings
    else:
        savings_norm = np.zeros_like(savings)
    
    # Demand-aware weighting: edges between customers with higher combined
    # demand are more valuable for consolidation (fewer depot returns needed)
    demand_weight = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                demand_weight[i, j] = (demands[i] + demands[j]) / (2.0 * capacity)
    
    # Angular continuity score: reward edges that maintain smooth route direction
    # For edge (i,j), compute the angle between vectors (depot->i) and (i->j)
    # Small angles indicate smooth route continuation
    depot_coords = coords[0]
    angular_score = np.zeros((n, n))
    
    for i in range(1, n):
        if time.time() > deadline - 30.0:
            break
        vec_depot_i = coords[i] - depot_coords
        norm_di = np.linalg.norm(vec_depot_i)
        if norm_di < 1e-10:
            continue
        vec_depot_i_unit = vec_depot_i / norm_di
        
        for j in range(1, n):
            if i == j:
                continue
            vec_i_j = coords[j] - coords[i]
            norm_ij = np.linalg.norm(vec_i_j)
            if norm_ij < 1e-10:
                continue
            vec_i_j_unit = vec_i_j / norm_ij
            
            # Cosine of angle between depot->i and i->j
            cos_angle = np.dot(vec_depot_i_unit, vec_i_j_unit)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Reward small angles (smooth continuation), penalize large angles
            # Map cos_angle from [-1, 1] to [0.5, 1.5] so it's always positive
            # and peaks at 1.0 when angles match perfectly
            angular_score[i, j] = 0.5 + 0.5 * (cos_angle + 1.0) / 2.0
            angular_score[j, i] = 0.5 + 0.5 * (cos_angle + 1.0) / 2.0
    
    # Combine: base heuristic modulated by savings, demand, and angular structure
    # The multiplier increases attractiveness for edges that:
    # 1. Have high savings (good for consolidation)
    # 2. Connect customers with higher combined demand
    # 3. Maintain smooth route direction (non-crossing)
    savings_boost = 1.0 + 1.5 * savings_norm * demand_weight
    angular_boost = 1.0 + 0.8 * (angular_score - 1.0)  # centered around 1.0
    
    # Final heuristic for customer-customer edges
    heuristic = base_heuristic * savings_boost * angular_boost
    
    # Handle depot edges specially
    # Depot-to-customer edge attractiveness should reflect:
    # 1. Distance to depot (shorter = better)
    # 2. How well the customer can be consolidated with others
    # 3. Angular position relative to depot
    for i in range(1, n):
        # Average savings potential of customer i with other customers
        if n > 2:
            avg_savings_i = np.mean(savings_norm[i, 1:])
        else:
            avg_savings_i = 0.0
        
        # Depot edge boost: customers with high consolidation potential
        # get slightly boosted depot edges to encourage route inclusion
        depot_boost = 1.0 + 0.8 * avg_savings_i
        
        heuristic[0, i] = base_heuristic[0, i] * depot_boost
        heuristic[i, 0] = base_heuristic[i, 0] * depot_boost
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Validate shape
    if heuristic.shape != distance_matrix.shape:
        raise ValueError("Wrong shape")
    if not np.all(np.isfinite(heuristic)):
        raise ValueError("Non-finite values")
    
    return heuristic
# EVOLVE-BLOCK-END
