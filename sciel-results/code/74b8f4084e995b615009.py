"""Traveling salesman constructive heuristic: force-directed / angle-aware greedy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for per-instance precomputation to keep per-call cost low
_cache = {}


def _get_coords(distance_matrix):
    """
    Derive approximate 2D coordinates from the distance matrix using a simple
    embedding trick: we don't actually need true coords; we can work directly
    with distances. But for force/angle logic we want vectors.
    Since the problem states Euclidean instances, we can recover coordinates
    up to an isometry via classical MDS, but that's expensive per instance.
    
    Alternative: work purely in distance space. For 'direction to destination'
    and 'alignment', we can use a surrogate: the change in distance to destination
    when moving from current to candidate. This is a well-known greedy improvement.
    
    Let's use a scoring function that combines:
      - proximity to current node (nearest neighbor)
      - progress toward destination (decrease in dist(candidate, dest))
      - penalty for cities that are 'far behind' (to avoid backtracking)
    
    Actually, a stronger known heuristic is the 'nearest neighbor with look-ahead'
    or 'angle-based' selection. Let's implement a score that favors candidates
    that are close to current AND make good progress toward the destination,
    while avoiding candidates that would cause a large detour.
    """
    return None  # placeholder, we'll work in distance space


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city to visit in a TSP tour construction.
    
    Strategy: Score each unvisited city by a combination of:
      1. Distance from current node (lower is better)
      2. Distance from candidate to destination (lower is better, encourages progress)
      3. A penalty term that discourages choosing a city that is 'far' from the
         general direction of travel (approximated via the sum of distances)
    
    We use a weighted sum with adaptive weights based on how many nodes remain.
    When few nodes remain, we care more about closing the loop (distance to destination).
    When many remain, we care more about nearest neighbor to build a compact tour.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    # Distances from unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize distances to [0, 1] for fair weighting
    d_current_norm = d_current / (np.max(d_current) + 1e-12)
    d_dest_norm = d_dest / (np.max(d_dest) + 1e-12)
    
    # Adaptive weight: as we get closer to finishing, weight destination progress more
    # fraction of tour remaining
    total_nodes = distance_matrix.shape[0]
    remaining_fraction = n_unvisited / total_nodes
    
    # Weight for nearest-neighbor vs destination-progress
    # Early: mostly nearest neighbor (w_nn high, w_dest low)
    # Late: mostly destination progress (w_nn low, w_dest high)
    w_nn = 1.0 - remaining_fraction * 0.5
    w_dest = remaining_fraction * 0.5
    
    # Also add a small penalty for 'detour': if candidate is far from both current
    # and destination, it's likely a bad choice. Use a soft penalty.
    # Score = w_nn * d_current_norm + w_dest * d_dest_norm
    score = w_nn * d_current_norm + w_dest * d_dest_norm
    
    # Select the candidate with minimum score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
