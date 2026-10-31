"""Constructive TSP heuristic using a geometrically-informed scoring function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds, with safety margin
_TIME_MARGIN = 5.0  # seconds before hard limit


def _check_time():
    """Return True if we are within the time budget."""
    return (time.time() - _START_TIME) < (_DEADLINE - _TIME_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour construction.
    
    Uses a geometrically-informed scoring that balances:
    1. Distance from current node (local connectivity)
    2. Distance to destination (global closure tendency)
    3. Circumradius of triangle (current, destination, candidate) - prefers tight turns
    4. Angle-based penalty for sharp turns (avoid inefficient zigzags)
    
    The score is a weighted combination that prioritizes geometrically "tight"
    paths that maintain local convexity while progressing toward closing the tour.
    """
    if len(unvisited_nodes) == 0:
        raise ValueError("No unvisited nodes")
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if not _check_time():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Extract relevant distances
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    n_candidates = len(unvisited_nodes)
    
    if n_candidates == 1:
        return int(unvisited_nodes[0])
    
    # Compute circumradius of triangle (current, destination, candidate)
    # Circumradius R = (a * b * c) / (4 * Area)
    # where a, b, c are side lengths and Area is the triangle area
    
    # For each candidate, we have triangle with vertices:
    # current_node, destination_node, candidate
    # Side lengths:
    # a = dist(current, destination) = dist_current_to_dest (constant)
    # b = dist(current, candidate) = dist_from_current[i]
    # c = dist(destination, candidate) = dist_to_dest[i]
    
    # Area using Heron's formula or cross product
    # Using the formula: Area = 0.5 * |a * b * sin(C)| where C is angle at current
    # But we need the actual area. Use Heron's formula:
    # s = (a + b + c) / 2
    # Area = sqrt(s * (s-a) * (s-b) * (s-c))
    
    a = dist_current_to_dest
    b = dist_from_current
    c = dist_to_dest
    
    # Handle degenerate cases
    s = (a + b + c) / 2.0
    # Clip to avoid negative values due to floating point
    area_sq = np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    area = np.sqrt(area_sq)
    
    # Circumradius: R = (a * b * c) / (4 * Area)
    # If area is 0 (collinear), circumradius is infinite - penalize heavily
    circumradius = np.where(area > 1e-10, (a * b * c) / (4.0 * area), 1e10)
    
    # Compute angle at current_node using law of cosines
    # cos(C) = (a^2 + b^2 - c^2) / (2 * a * b)
    # where a = dist(current, dest), b = dist(current, candidate), c = dist(dest, candidate)
    # Angle at current is between vectors (dest - current) and (candidate - current)
    
    # cos(angle_at_current) = (b^2 + a^2 - c^2) / (2 * a * b)
    # Wait, let me be more careful. In triangle with vertices A=current, B=dest, C=candidate:
    # Side opposite A is a = dist(B,C) = c
    # Side opposite B is b = dist(A,C) = b  
    # Side opposite C is c = dist(A,B) = a
    # Angle at A (current): cos(A) = (b^2 + a^2 - c^2) / (2*a*b) where here a=dist(A,B), b=dist(A,C), c=dist(B,C)
    
    # So: cos(angle_at_current) = (dist_current_to_dest^2 + dist_from_current^2 - dist_to_dest^2) / (2 * dist_current_to_dest * dist_from_current)
    
    denom = 2.0 * a * b
    cos_angle = np.where(denom > 1e-10, (a**2 + b**2 - c**2) / denom, 0.0)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_at_current = np.arccos(cos_angle)
    
    # Scoring: We want to minimize a combination of:
    # 1. Distance from current (primary: keep tour compact)
    # 2. Circumradius (secondary: prefer tight turns, avoid wide arcs)
    # 3. Angle penalty: penalize very small angles (backtracking) and very large angles (sharp turns)
    #    Ideal angle is around 60-120 degrees
    
    # Normalize components to similar scale
    max_dist = np.maximum(np.max(dist_from_current), 1e-10)
    max_circum = np.maximum(np.max(circumradius), 1e-10)
    
    # Primary score: distance from current
    score_dist = dist_from_current / max_dist
    
    # Secondary: circumradius (tighter is better)
    score_circum = circumradius / max_circum
    
    # Angle penalty: prefer angles in [60°, 120°], penalize outside
    # Convert to radians: 60° = pi/3, 120° = 2*pi/3
    angle_ideal_low = np.pi / 3.0
    angle_ideal_high = 2.0 * np.pi / 3.0
    
    # Penalty: 0 in ideal range, increases quadratically outside
    angle_penalty = np.zeros_like(angle_at_current)
    below = angle_at_current < angle_ideal_low
    above = angle_at_current > angle_ideal_high
    angle_penalty[below] = (angle_ideal_low - angle_at_current[below]) ** 2
    angle_penalty[above] = (angle_at_current[above] - angle_ideal_high) ** 2
    
    # Normalize angle penalty
    max_angle_penalty = np.maximum(np.max(angle_penalty), 1e-10)
    score_angle = angle_penalty / max_angle_penalty
    
    # Combine scores with weights
    # Emphasize distance most, then circumradius, then angle
    w_dist = 0.5
    w_circum = 0.3
    w_angle = 0.2
    
    total_score = w_dist * score_dist + w_circum * score_circum + w_angle * score_angle
    
    # Select the candidate with minimum total score
    best_idx = np.argmin(total_score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
