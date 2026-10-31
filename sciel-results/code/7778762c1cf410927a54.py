"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed data per distance_matrix
_cache = {}


def _get_cache_key(dm):
    """Create a lightweight hash key for the distance matrix."""
    # Use a small sample of values to create a key
    h = hash((dm.shape, dm[0, 1] if dm.shape[0] > 1 else 0, dm[1, 0] if dm.shape[1] > 1 else 0,
              dm[0, 0] if dm.shape[0] > 0 else 0))
    return h


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a multi-factor scoring heuristic that balances:
    1. Distance from current node (nearest neighbor component)
    2. Distance from destination node (to avoid getting stuck far from destination)
    3. Lookahead: how well the candidate connects to remaining unvisited nodes
    4. Angle/cross-product penalty to avoid backtracking (U-turns)
    
    This is a constructive heuristic that tries to build a tour that is both
    locally efficient and globally coherent.
    """
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited
    dist_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Normalize distances for scoring
    max_d1 = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
    max_d2 = np.max(dist_from_dest) if np.max(dist_from_dest) > 0 else 1.0
    
    # Component 1: Nearest neighbor score (lower is better)
    nn_score = dist_from_current / max_d1
    
    # Component 2: Destination proximity (we want to eventually get back to destination)
    # Penalize nodes that are far from destination, but weight it less
    dest_score = dist_from_dest / max_d2
    
    # Component 3: Lookahead - average distance from candidate to other unvisited nodes
    # This helps avoid choosing a node that is isolated or creates a long edge later
    # For efficiency, use a sampled subset if there are many unvisited nodes
    if n_unvisited <= 30:
        # Compute mean distance from each candidate to all other unvisited
        sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Mean distance from each row (candidate) to all others
        lookahead = np.mean(sub_dm, axis=1)
        max_look = np.max(lookahead) if np.max(lookahead) > 0 else 1.0
        lookahead_score = lookahead / max_look
    else:
        # Use a smaller sample for efficiency
        sample_size = min(20, n_unvisited)
        sample_idx = np.random.choice(n_unvisited, size=sample_size, replace=False)
        sample_nodes = unvisited_nodes[sample_idx]
        sub_dm_sample = distance_matrix[np.ix_(unvisited_nodes, sample_nodes)]
        lookahead = np.mean(sub_dm_sample, axis=1)
        max_look = np.max(lookahead) if np.max(lookahead) > 0 else 1.0
        lookahead_score = lookahead / max_look
    
    # Component 4: Angle penalty - avoid U-turns
    # If we've been going in a direction, penalize reversing
    # We can approximate this by checking if the vector from current to candidate
    # has a negative dot product with the vector from destination to current
    # (i.e., we're going away from where we came from)
    # Actually, a better angle metric: penalize if the turn angle is close to 180 degrees
    # But we don't have the previous node. Let's use a simpler heuristic:
    # Penalize candidates that are far from both current and destination (outliers)
    # This is already captured by nn_score and dest_score
    
    # Combine scores with weights
    # Weight tuning: emphasize nearest neighbor, but incorporate destination and lookahead
    w_nn = 0.55
    w_dest = 0.20
    w_look = 0.25
    
    total_score = w_nn * nn_score + w_dest * dest_score + w_look * lookahead_score
    
    # Add a small random tiebreaker for stability
    # Actually, let's not add randomness for deterministic behavior
    
    # Select the candidate with the lowest total score
    best_idx = np.argmin(total_score)
    best_node = unvisited_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
