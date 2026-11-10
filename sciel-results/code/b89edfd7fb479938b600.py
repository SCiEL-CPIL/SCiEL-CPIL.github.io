import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ACO orienteering.
    
    The heuristic balances:
    1. Prize value of the destination node
    2. Effective distance cost (d_ij + d_j0) - accounts for return to origin
    3. Feasibility margin - how much budget remains after visiting j
    
    This creates a strong preference for high-prize nodes that are close
    and leave ample budget for future visits.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Extract distance from each node to the origin (node 0)
    d_to_origin = distance_matrix[:, 0]
    
    # Effective distance: cost of going i->j and then j->0
    d_eff = distance_matrix + d_to_origin[None, :]
    
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    d_eff_safe = d_eff + eps
    d_ij_safe = distance_matrix + eps
    
    # Feasibility margin: how much budget remains if we go i->j->0
    # This is max_len - (d_ij + d_j0)
    # Higher margin means more room for future nodes
    margin = max_len - d_eff
    # Clip to avoid negative values affecting the heuristic
    margin_safe = np.maximum(margin, 0.0)
    
    # Heuristic component 1: Prize / effective distance (reward per unit cost)
    # This favors high-prize nodes that are cheap to reach
    eta1 = prizes[None, :] / d_eff_safe
    
    # Heuristic component 2: Prize / effective distance squared (stronger penalty for distance)
    eta2 = prizes[None, :] / (d_eff_safe ** 2)
    
    # Heuristic component 3: Prize * feasibility margin
    # This favors moves that leave lots of budget for future exploration
    eta3 = prizes[None, :] * margin_safe
    
    # Heuristic component 4: Prize / direct distance (immediate reward density)
    eta4 = prizes[None, :] / d_ij_safe
    
    # Normalize each component to [0, 1] range
    def normalize(eta):
        """Normalize eta to [0, 1] range, handling zeros and non-finite values."""
        eta = eta.copy()
        eta[~np.isfinite(eta)] = 0
        eta[eta < 0] = 0
        max_val = np.max(eta)
        if max_val > 0:
            eta = eta / max_val
        return eta
    
    eta1 = normalize(eta1)
    eta2 = normalize(eta2)
    eta3 = normalize(eta3)
    eta4 = normalize(eta4)
    
    # Combine heuristics with weights
    # Weight towards reward-per-cost and feasibility margin
    # eta1: moderate weight for balanced reward/cost
    # eta2: lower weight for strong distance penalty
    # eta3: high weight for budget preservation
    # eta4: moderate weight for immediate reward density
    w1, w2, w3, w4 = 0.30, 0.15, 0.35, 0.20
    eta = w1 * eta1 + w2 * eta2 + w3 * eta3 + w4 * eta4
    
    # Ensure non-negative and handle any potential numerical issues
    # The task says: "Heuristic entries that are not positive become 1e-9"
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
