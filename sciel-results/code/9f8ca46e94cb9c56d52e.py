import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def _compute_angle_score(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Geometric convex-hull-peeling heuristic.
    Score = distance + angle penalty.
    Angle penalty rewards nodes that continue the current direction toward destination.
    """
    n = len(unvisited_nodes)
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get coordinates if possible? No, we only have distance matrix.
    # We must work with distances only.
    # "Angle" can be approximated by how well the triangle inequality holds or by relative distances.
    # A node j is "good" if d(c, j) + d(j, dest) is close to d(c, dest).
    # This implies j is roughly on the path from c to dest.
    # However, for TSP, we want to trace the hull.
    # A better geometric proxy with only distances:
    # Compute d(j, dest) for all j.
    # Compute d(c, dest).
    # The "directional" score can be: d(c, j) - alpha * (d(c, dest) - d(j, dest)) ?
    # If j is between c and dest, d(c, j) + d(j, dest) = d(c, dest).
    # If j is far off, d(c, j) + d(j, dest) > d(c, dest).
    # So, penalty = d(c, j) + d(j, dest) - d(c, dest) is a measure of "off-path-ness".
    # But we also want to minimize d(c, j).
    
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    d_c_dest = distance_matrix[current_node, destination_node]
    
    # "Off-path" cost: how much longer is the path c->j->dest compared to c->dest
    off_path = d_current + d_dest - d_c_dest
    
    # We want to minimize d_current, but penalize being far off the direct path?
    # Actually, for convex hull peeling, we want to go to the "extreme" point in the direction of travel.
    # If we are moving from c to dest, the "forward" direction is defined by the vector c->dest.
    # Points with small off_path are "in front". Points with large off_path are "behind" or "side".
    # To trace the hull, we might want to pick the point that is most "extreme" in the forward direction.
    
    # Let's try a score: d_current * (1 + off_path / d_c_dest)
    # Or simply: d_current + beta * off_path
    
    # Let's look at the "2-step" lookahead again but modified.
    # The previous program used: d_current[j] + min_{k != j} (d_jk + d_k_dest)
    # This is a 2-step lookahead.
    # The geometric intuition suggests that we should prefer nodes that are "on the hull".
    # With only distances, identifying hull nodes is hard.
    
    # Alternative structural change:
    # Use a "potential" function.
    # Score = d_current[j] + d_dest[j] - d_c_dest
    # This is exactly the off_path cost.
    # Minimizing this picks the node that is most "aligned" with the destination.
    # This is a "greedy toward destination" heuristic, which is bad for TSP (it makes a line, not a loop).
    # BUT, if we combine it with the 2-step lookahead?
    
    # Let's try: Score = d_current[j] + 0.5 * min_{k != j} (d_jk + d_k_dest) + 0.5 * (d_current[j] + d_dest[j] - d_c_dest)
    
    # Actually, let's look at the inspiration: "select the next node that minimizes the angle deviation from the current direction vector relative to the destination".
    # Since we don't have coordinates, we can't compute angles directly.
    # But we can use the Law of Cosines if we assume Euclidean space?
    # d(c, j)^2 + d(j, dest)^2 - 2*d(c, j)*d(j, dest)*cos(theta) = d(c, dest)^2
    # cos(theta) = (d(c, j)^2 + d(j, dest)^2 - d(c, dest)^2) / (2 * d(c, j) * d(j, dest))
    # Small theta means cos(theta) is close to 1.
    # We want to MINIMIZE angle deviation, i.e., MINIMIZE theta, i.e., MAXIMIZE cos(theta).
    # So we want to MAXIMIZE the cosine value.
    # Equivalently, MINIMIZE (1 - cos(theta)).
    
    # Let's compute cos_theta for each candidate j.
    # Handle division by zero (if d(c, j) or d(j, dest) is 0).
    
    d_cj = d_current
    d_jd = d_dest
    d_cd = d_c_dest
    
    # Numerator of cosine
    num = d_cj**2 + d_jd**2 - d_cd**2
    # Denominator of cosine
    den = 2 * d_cj * d_jd
    
    # Avoid division by zero
    den_safe = np.where(den > 1e-9, den, 1e-9)
    cos_theta = num / den_safe
    
    # Clip to [-1, 1] due to numerical errors
    cos_theta = np.clip(cos_theta, -1, 1)
    
    # Angle deviation score: we want to minimize theta, so minimize (1 - cos_theta)
    # Or simply maximize cos_theta.
    # Let's use score = d_current - alpha * cos_theta
    # If alpha is large, we prefer aligned nodes.
    # If alpha is small, we prefer close nodes.
    
    # Let's try a balance.
    # Scale cos_theta to be comparable to distances?
    # Distances are in same units as d_current.
    # cos_theta is dimensionless.
    # We can scale cos_theta by d_c_dest or some average distance.
    
    # Let's try: score = d_current - 0.5 * d_c_dest * cos_theta
    # This rewards nodes that are close AND aligned.
    
    scores = d_current - 0.5 * d_c_dest * cos_theta
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor if we are running out of time
        d_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_current)
        return int(unvisited_nodes[best_idx])
    
    if n == 2:
        j0 = unvisited_nodes[0]
        j1 = unvisited_nodes[1]
        cost0 = distance_matrix[current_node, j0] + distance_matrix[j0, j1] + distance_matrix[j1, destination_node]
        cost1 = distance_matrix[current_node, j1] + distance_matrix[j1, j0] + distance_matrix[j0, destination_node]
        
        if cost0 < cost1:
            return int(j0)
        else:
            return int(j1)
            
    # Use geometric angle heuristic
    return _compute_angle_score(current_node, destination_node, unvisited_nodes, distance_matrix)
