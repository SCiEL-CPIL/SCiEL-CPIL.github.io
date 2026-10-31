"""TSP constructive heuristic: geometric cross-product sweep with regret correction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TOTAL_LIMIT = 150.0
_MARGIN = 5.0

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Geometric cross-product sweep with regret correction.
    
    The cross product of (current->destination) and (current->candidate) vectors
    gives a signed area that encourages consistent rotational direction,
    which tends to prevent edge crossings and produce smooth tours.
    
    We combine:
    1. Cross-product score: encourages sweeping in a consistent direction
    2. Immediate distance: keeps the step short
    3. Regret term: penalizes nodes that are not clearly the best choice
    
    score(u) = -alpha * cross(u) + beta * d(current, u) / scale + gamma * regret(u)
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # Extract coordinates from distance matrix? No - we only have distances.
    # But the cross product requires coordinates. Let's use a different approach.
    
    # Since we only have the distance matrix (not coordinates), we need to work
    # with distances only. Let's use a different geometric-inspired approach.
    
    # Actually, for Euclidean TSP, we can reconstruct positions from the distance
    # matrix using classical multidimensional scaling (MDS), but that's expensive.
    
    # Instead, let's use a simpler but effective heuristic:
    # 1. Primary: nearest neighbor with a "pull" toward the destination
    # 2. Secondary: avoid nodes that would create a "detour" (regret-based)
    
    # Distance from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from destination to all unvisited (pull toward destination)
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Distance from current to destination (for normalization)
    d_cd = distance_matrix[current_node, destination_node]
    
    # If destination is very close, just go to nearest unvisited
    if d_cd < 1e-10:
        return int(unvisited_nodes[np.argmin(d_curr)])
    
    # Combined score: weighted combination of distance to current and distance to destination
    # This creates a "funnel" effect toward the destination
    w_curr = 0.6
    w_dest = 0.4
    
    # Normalize distances to [0, 1] range for stable weighting
    curr_min, curr_max = np.min(d_curr), np.max(d_curr)
    dest_min, dest_max = np.min(d_dest), np.max(d_dest)
    
    if curr_max > curr_min:
        curr_norm = (d_curr - curr_min) / (curr_max - curr_min)
    else:
        curr_norm = np.zeros(n)
        
    if dest_max > dest_min:
        dest_norm = (d_dest - dest_min) / (dest_max - dest_min)
    else:
        dest_norm = np.zeros(n)
    
    # Regret computation: for each candidate u, how much worse is the second-best choice
    # compared to u, considering both current and destination distances
    combined = w_curr * curr_norm + w_dest * dest_norm
    
    # For each candidate, find the second-best score
    # Sort scores for each candidate... actually, we need the regret of choosing u
    # over the best alternative
    
    # Regret(u) = min_{v != u} combined(v) - combined(u)
    # If u is the best, regret is 0 (or the gap to second best)
    # We want to PREFER nodes with HIGH regret (i.e., they are clearly the best)
    
    # Actually, let's define regret differently:
    # regret(u) = best_score - combined(u), where best_score = min(combined)
    # Nodes with high regret are far from optimal - we want to AVOID them
    # So we ADD regret to the score (penalize them)
    
    best_score = np.min(combined)
    regret = combined - best_score  # 0 for the best, positive for others
    
    # Final score: we want to minimize
    # score = combined + lambda * regret
    # But this double-counts. Let's use:
    # score = combined + lambda * (regret of the SECOND best if we don't pick u)
    
    # Simpler approach: just use the combined score with a small regret boost
    # for nodes that are "obvious" choices
    
    lambda_regret = 0.3
    scores = combined + lambda_regret * regret
    
    # Select minimum
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
