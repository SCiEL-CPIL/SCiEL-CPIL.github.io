"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)

    n = distance_matrix.shape[0]

    # Base heuristic: inverse distance, sharpened with a power
    # Use a small epsilon to avoid division by zero
    base = 1.0 / (distance_matrix + 1e-9)

    # Clarke-Wright savings: saving(i,j) = d[0,i] + d[0,j] - d[i,j]
    # High savings indicate edges that are structurally beneficial
    d0 = distance_matrix[0, :]  # distance from depot to each node
    # savings[i,j] = d0[i] + d0[j] - d[i,j] for i != j
    savings = d0[:, None] + d0[None, :] - distance_matrix

    # Zero out diagonal and depot row/col (depot edges handled separately)
    savings[np.diag_indices(n)] = 0.0
    savings[0, :] = 0.0
    savings[:, 0] = 0.0

    # Normalize savings to [0, 1] range for scaling
    max_saving = np.max(savings)
    if max_saving > 0:
        savings_norm = savings / max_saving
    else:
        savings_norm = np.zeros_like(savings)

    # Boost edges with high savings: multiply base heuristic by a factor
    # that increases with savings. Use an exponential boost for strong signal.
    # boost = 1 + alpha * savings_norm^beta
    alpha = 5.0   # strength of savings boost
    beta = 1.0    # exponent on savings normalization
    savings_boost = 1.0 + alpha * (savings_norm ** beta)

    # Demand-aware component: prefer edges to customers with moderate demand
    # (not too high, not zero). This is a soft signal.
    # Normalize demands to [0,1]
    max_demand = np.max(demands) if np.max(demands) > 0 else 1.0
    demands_norm = demands / max_demand

    # Edges to customers with lower demand are slightly more attractive
    # (they leave more capacity for other customers)
    demand_factor = 1.0 + 0.3 * (1.0 - demands_norm[None, :])
    demand_factor[0, :] = 1.0  # depot has no demand effect
    demand_factor[:, 0] = 1.0

    # Combine: base * savings_boost * demand_factor
    heuristic = base * savings_boost * demand_factor

    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9

    # Verify shape matches input
    assert heuristic.shape == distance_matrix.shape, "Shape mismatch"

    return heuristic
# EVOLVE-BLOCK-END
