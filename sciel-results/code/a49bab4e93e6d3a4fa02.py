"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin below 120s

def _check_deadline():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Delaunay-inspired geometric heuristic.
    Prioritizes candidates that form "tight" triangles with current and destination,
    avoiding long chords that break local convexity.
    """
    # Check deadline for safety
    if _check_deadline():
        # Fallback to nearest neighbor if time is running out
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    
    # If only one node left, it must be the destination (or just the last one)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If two nodes left, pick the one that minimizes the total tour completion cost
    if n_unvisited == 2:
        c1, c2 = unvisited_nodes[0], unvisited_nodes[1]
        # Option 1: current -> c1 -> c2 -> destination
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        # Option 2: current -> c2 -> c1 -> destination
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1 if cost1 < cost2 else c2)
    
    # General case: Use geometric heuristic based on triangle properties
    # For each candidate c, compute a score based on:
    # 1. Distance from current to c (prefer closer)
    # 2. Circumradius of triangle (current, destination, c) - smaller is better (tighter turn)
    # 3. Angle at c in the triangle - prefer angles that maintain convexity
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_from_dest = distance_matrix[destination_node, unvisited_nodes]
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    for i, c in enumerate(unvisited_nodes):
        d1 = dists_from_current[i]  # dist(current, c)
        d2 = dists_from_dest[i]     # dist(destination, c)
        d3 = dist_current_dest       # dist(current, destination)
        
        # Compute circumradius of triangle (current, destination, c)
        # R = (a * b * c) / (4 * Area)
        # Using Heron's formula for area
        s = (d1 + d2 + d3) / 2.0
        area_sq = s * (s - d1) * (s - d2) * (s - d3)
        
        if area_sq <= 1e-12:
            # Degenerate triangle (collinear points)
            # Use a penalty or fallback
            circumradius = np.inf
        else:
            area = np.sqrt(area_sq)
            circumradius = (d1 * d2 * d3) / (4.0 * area)
        
        # Combine metrics:
        # - d1: direct travel cost (weight 1.0)
        # - circumradius: geometric tightness (weight 0.5, normalized)
        # Normalize circumradius relative to the scale of the problem
        # Use a soft normalization: circumradius / (d1 + d2 + d3 + eps)
        norm_circum = circumradius / (d1 + d2 + d3 + 1e-9)
        
        # Score: lower is better
        # Weighted combination: direct cost + geometric penalty
        score = d1 + 0.3 * norm_circum
        
        if score < best_score:
            best_score = score
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
