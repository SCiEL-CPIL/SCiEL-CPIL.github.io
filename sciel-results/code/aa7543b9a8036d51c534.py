# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric heuristic that balances:
    1. Distance to candidate (primary)
    2. Alignment with the direction from current to destination (secondary)
    3. Avoidance of crossing the current-to-destination line (tertiary)
    
    This mimics optimal TSP tour curvature by preferring paths that
    smoothly progress toward the destination while avoiding backtracking.
    """
    if len(unvisited_nodes) == 0:
        return -1  # Should not happen if called correctly
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Get coordinates implicitly through distance matrix
    # We don't have explicit coordinates, so we work with distances only.
    
    # Primary: distance from current to each unvisited
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    
    # Secondary: distance from each candidate to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score: we want candidates that are close to current AND
    # make progress toward destination.
    # A simple weighted sum: alpha * dist_to_candidate + (1-alpha) * dist_to_dest
    # But we can do better: use the concept of "angle" via triangle inequality.
    
    # The straight-line distance from current to destination is a reference.
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # For each candidate, compute how "detour" it is:
    # detour = dist_to_candidate + dist_to_dest - dist_current_to_dest
    # Small detour means the candidate lies roughly on the path from current to dest.
    detours = dist_to_candidates + dist_to_dest - dist_current_to_dest
    
    # We want to minimize a combination of:
    # - dist_to_candidates (local progress)
    # - detours (global alignment)
    
    # Weighted score
    alpha = 0.7  # weight on local distance
    beta = 0.3   # weight on detour penalty
    
    # Normalize detours to be comparable to distances
    max_detour = np.max(detours)
    if max_detour > 1e-10:
        normalized_detours = detours / max_detour
    else:
        normalized_detours = detours
    
    scores = alpha * dist_to_candidates + beta * normalized_detours
    
    # Select the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
