"""Rewritten program using angular sweep heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using an angular sweep heuristic around the depot.
    
    The idea: compute the angle of each candidate node relative to the depot,
    and the angle of the current node relative to the depot. Prefer nodes that
    have a small angular deviation from the current node's direction, which
    encourages spatial clustering and reduces backtracking.
    
    Also respect capacity constraints: only consider nodes whose demand fits
    in rest_capacity. If no feasible node exists, return depot (0) to close route.
    """
    depot = int(depot)
    
    # Filter unvisited nodes by capacity feasibility
    feasible = []
    for node in unvisited_nodes:
        node = int(node)
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if not feasible:
        return 0  # Close route, return to depot
    
    # Compute angles relative to depot
    depot_coords = None
    # We need coordinates. Since we only have distance_matrix, we'll use a proxy:
    # Use the distance from depot and from current_node to create a pseudo-angular score.
    # 
    # Alternative: use the distance matrix to estimate relative positions.
    # A simpler geometric proxy: score each candidate by a combination of:
    #   - distance from current_node (local proximity)
    #   - distance from depot (to encourage staying in a "sector")
    # 
    # But the directive says to use angular sweep. Without explicit coordinates,
    # we can approximate angles using the law of cosines if we know three distances:
    #   d(depot, current), d(depot, candidate), d(current, candidate)
    # Then angle at depot between current and candidate can be computed.
    #
    # Let's do that:
    d_depot_current = distance_matrix[depot, current_node]
    
    best_node = feasible[0]
    best_score = float('inf')
    
    for node in feasible:
        node = int(node)
        d_depot_node = distance_matrix[depot, node]
        d_current_node = distance_matrix[current_node, node]
        
        # Compute angle at depot using law of cosines
        # cos(theta) = (a^2 + b^2 - c^2) / (2*a*b)
        # where a = d_depot_current, b = d_depot_node, c = d_current_node
        a = d_depot_current
        b = d_depot_node
        c = d_current_node
        
        if a < 1e-12 or b < 1e-12:
            # Degenerate case: current node or candidate is at depot
            angle = 0.0
        else:
            cos_theta = (a*a + b*b - c*c) / (2.0 * a * b)
            # Clamp to [-1, 1] for numerical stability
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = np.arccos(cos_theta)
        
        # Score: prefer small angular deviation. Also add a small distance penalty
        # to break ties and avoid pure angular clustering that might pick far nodes.
        # Weight: primarily angular, secondarily distance from current.
        score = angle + 0.1 * (d_current_node / max(a, b, 1e-12))
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return int(best_node)
# EVOLVE-BLOCK-END
