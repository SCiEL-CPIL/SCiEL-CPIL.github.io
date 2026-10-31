"""Rewritten program: TSP constructive heuristic using Delaunay-based geometric scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def _compute_circumradius(a, b, c):
    """Compute the circumradius of triangle formed by points a, b, c."""
    # Using the formula: R = (a*b*c) / (4*Area)
    # where a, b, c are side lengths
    side_a = np.linalg.norm(b - c)
    side_b = np.linalg.norm(a - c)
    side_c = np.linalg.norm(a - b)
    
    # Area using cross product (2D)
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.abs(cross)
    
    if area < 1e-10:
        # Degenerate triangle, return large value
        return 1e10
    
    R = (side_a * side_b * side_c) / (4.0 * area)
    return R

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Delaunay Triangulation-based Geometric Scoring.
    
    1. Compute Delaunay triangulation of unvisited nodes + current + destination.
    2. For each candidate, compute the circumradius of the triangle (current, destination, candidate).
    3. Score candidates based on:
       - Immediate distance from current node.
       - Circumradius penalty (prefer tight triangles that preserve local convexity).
       - Proximity to destination (weighted more heavily near the end of the tour).
       - Delaunay neighbor count (prefer nodes with fewer neighbors to avoid dense regions).
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
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Get coordinates for Delaunay triangulation
    # We need to reconstruct coordinates from distance matrix
    # For Euclidean instances, we can use MDS or assume we have access to coordinates
    # Since we only have distance_matrix, we'll use a geometric proxy
    
    # Compute circumradius proxy using distances
    # For triangle (current, dest, candidate_i):
    # sides are: d(curr,dest), d(curr,i), d(dest,i)
    d_curr_dest = distance_matrix[current_node, destination_node]
    
    # Circumradius formula: R = (a*b*c) / (4*Area)
    # Heron's formula for area: s = (a+b+c)/2, Area = sqrt(s*(s-a)*(s-b)*(s-c))
    a = d_curr_dest  # side opposite to candidate
    b = d_curr       # side from current to candidate
    c = d_dest       # side from destination to candidate
    
    # Compute Heron's formula vectorized
    s = (a + b + c) / 2.0
    # Handle numerical issues where terms might be negative
    term1 = np.maximum(s, 0)
    term2 = np.maximum(s - a, 0)
    term3 = np.maximum(s - b, 0)
    term4 = np.maximum(s - c, 0)
    
    area_sq = term1 * term2 * term3 * term4
    area_sq = np.maximum(area_sq, 0)
    area = np.sqrt(area_sq)
    
    # Circumradius
    with np.errstate(divide='ignore', invalid='ignore'):
        circumradii = (a * b * c) / (4.0 * area)
    
    # Replace non-finite values with large number
    circumradii = np.where(np.isfinite(circumradii), circumradii, 1e10)
    
    # 4. Compute Delaunay neighbor count proxy
    # Count how many other unvisited nodes are "close" to each candidate
    # Use a threshold based on median distance
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    median_dist = np.median(d_sub[~np.eye(n, dtype=bool)])
    
    # Count neighbors within 1.5 * median distance
    neighbor_counts = np.sum(d_sub < 1.5 * median_dist, axis=1)
    
    # 5. Compute scores
    # Normalize circumradii
    min_circum = np.min(circumradii)
    max_circum = np.max(circumradii)
    if max_circum > min_circum:
        circum_norm = (circumradii - min_circum) / (max_circum - min_circum)
    else:
        circum_norm = np.zeros(n)
    
    # Destination weight: stronger ramp as n decreases
    dest_weight = 0.1 + 2.0 / n
    
    # Circumradius weight: moderate penalty for large triangles
    circum_weight = 0.3
    
    # Neighbor count weight: prefer sparse regions
    neighbor_weight = 0.1
    
    # Immediate cost weight
    curr_weight = 1.0
    
    scores = (curr_weight * d_curr 
              + circum_weight * circum_norm * median_dist
              + dest_weight * d_dest
              + neighbor_weight * neighbor_counts * median_dist / n)
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
