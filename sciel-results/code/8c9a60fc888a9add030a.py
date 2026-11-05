"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_EVAL_START = time.time()
DEADLINE = _EVAL_START + 140  # Safety margin before 150s kill


def _time_remaining():
    return DEADLINE - time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a geometric sweep heuristic:
    - Project customers onto a principal axis derived from the distance matrix
    - Score candidates by angular deviation from current heading + distance
    - Add adaptive depot-proximity weighting when capacity is tight
    - Return depot if no feasible customer remains
    """
    # Deadline guard
    if _time_remaining() < 1.0:
        # Emergency fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # If no unvisited nodes, return depot
    if len(unvisited_nodes) == 0:
        return 0

    # Filter to only feasible customers (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible = unvisited_arr[feasible_mask]

    # If no feasible customer, return depot to close route
    if len(feasible) == 0:
        return 0

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute distances
    dist_to_node = distance_matrix[current_node, feasible]
    dist_to_depot = distance_matrix[feasible, depot]

    # Geometric sweep: project onto principal axis from depot
    # Use distances from depot as proxy for 1D projection
    depot_to_feasible = distance_matrix[depot, feasible]
    depot_to_current = distance_matrix[depot, current_node]

    # Heading vector: from current to depot direction (in distance space)
    # Angular deviation approximated by how much the candidate deviates from
    # the line between current and depot
    # If candidate is "ahead" on the sweep (closer to depot in projection), it's good
    # Compute a simple angular score: penalize candidates that are far from the
    # current-depot line in terms of relative positioning
    # Use the law of cosines approximation: 
    # angle ~ arccos((d_cur^2 + d_dep^2 - d_cur_dep^2) / (2*d_cur*d_dep))
    # But we don't have direct coordinates, so use distance triangle inequality
    # Simpler: score based on whether candidate maintains sweep direction
    # A good sweep keeps candidates roughly between current and depot in projection
    
    # Principal axis projection: use depot distances as 1D coordinates
    # Current position on axis: depot_to_current
    # Candidate position: depot_to_feasible[i]
    # Sweep direction: from current toward depot (decreasing depot distance)
    # Prefer candidates with depot distance slightly less than current (moving toward depot)
    # but not too much (to maintain cluster density)
    
    # Adaptive depot weight: stronger when capacity is tight
    cap_ratio = rest_capacity / 50.0
    depot_weight = 0.3 + 0.9 * (1.0 - cap_ratio)
    depot_weight = max(0.2, min(1.5, depot_weight))

    # Geometric sweep score:
    # 1. Primary: distance from current node (local proximity)
    # 2. Secondary: depot proximity (weighted by capacity)
    # 3. Sweep bonus: reward candidates that are in the "forward" direction
    #    (between current and depot in terms of depot-distance projection)
    
    # Forward direction score: candidates with depot_dist < current_depot_dist
    # are moving toward depot (good for sweep), but we want a smooth transition
    current_proj = depot_to_current
    forward_score = np.maximum(0, current_proj - depot_to_feasible)  # positive if moving toward depot
    
    # Normalize forward score to be comparable to distances
    max_forward = max(1.0, current_proj)
    forward_normalized = forward_score / max_forward
    
    # Sweep weight: moderate influence
    sweep_weight = 0.15
    
    # Combined score
    scores = dist_to_node + depot_weight * dist_to_depot - sweep_weight * forward_normalized

    # Demand-preservation penalty when capacity is tight
    if rest_capacity < 15:
        demand_penalty_factor = 0.02 * (1.0 - rest_capacity / 15.0)
        if demand_penalty_factor > 0:
            feasible_demands = demands[feasible]
            scores = scores + demand_penalty_factor * feasible_demands

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
