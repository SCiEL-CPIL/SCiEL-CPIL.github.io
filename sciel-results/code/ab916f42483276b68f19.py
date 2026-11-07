"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    Strategy: Use a slack-weighted prize density heuristic. For each edge (i, j),
    estimate the marginal gain of visiting j after i by weighting prize[j] by the
    remaining path budget capacity (max_len - d[i,j] - d[j,0]). This dynamically
    prioritizes nodes that maximize prize density per unit of remaining path budget.
    
    We also incorporate a global prize ranking and distance-based scaling to create
    a more nuanced heuristic than simple prize/distance ratios.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure prizes[0] = 0 (as per task spec)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # Compute distance from each node to origin (node 0)
    d_to_origin = distance_matrix[:, 0].copy()
    
    # For each potential move i -> j, compute:
    # 1. The cost of going i -> j -> 0: d[i,j] + d[j,0]
    # 2. The slack: max_len - (d[i,j] + d[j,0])
    # 3. Weight prize[j] by this slack (only if slack > 0)
    
    # Cost matrix: cost[i,j] = d[i,j] + d[j,0]
    cost_to_return = distance_matrix + d_to_origin[None, :]
    
    # Slack: how much budget remains after making this move and returning
    slack = max_len - cost_to_return
    
    # Only positive slack is useful
    slack = np.maximum(slack, 0.0)
    
    # Prize per unit of remaining budget: prize[j] / (1 + slack) 
    # Actually, we want to prioritize high prize with adequate slack.
    # A node with high prize but very little slack is risky.
    # A node with moderate prize and lots of slack allows for more future visits.
    
    # Let's use: attractiveness[i,j] = prize[j] * slack[i,j] / (1 + distance_matrix[i,j])
    # This rewards high prize, good slack, and short distances.
    
    # But we also want to encourage exploration of high-prize nodes early.
    # Let's blend multiple signals:
    
    # Signal 1: Prize density with slack weighting
    # Higher prize and more remaining budget = more attractive
    signal1 = prizes[None, :] * slack / (1.0 + distance_matrix)
    
    # Signal 2: Pure prize ranking (higher prize = more attractive)
    # Normalize prizes to [0, 1]
    max_prize = np.max(prizes[1:]) if n > 1 else 1.0
    if max_prize > 0:
        normalized_prizes = prizes / max_prize
    else:
        normalized_prizes = prizes.copy()
    signal2 = normalized_prizes[None, :]
    
    # Signal 3: Distance-based (closer = more attractive, but not too dominant)
    # Use inverse distance with a floor to avoid division by zero
    min_dist = np.min(distance_matrix[distance_matrix > 0]) if np.any(distance_matrix > 0) else 1.0
    inv_dist = 1.0 / (distance_matrix + min_dist * 0.1)
    max_inv_dist = np.max(inv_dist)
    if max_inv_dist > 0:
        signal3 = inv_dist / max_inv_dist
    else:
        signal3 = np.ones_like(distance_matrix)
    
    # Blend the signals with weights that emphasize prize and slack
    # Weight: 0.5 * signal1 + 0.3 * signal2 + 0.2 * signal3
    heuristic = 0.5 * signal1 + 0.3 * signal2 + 0.2 * signal3
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Diagonal should be very low (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Edge from node 0: should be based on prize and distance to return
    # The heuristic for edges from 0 is already computed above
    
    return heuristic
# EVOLVE-BLOCK-END
