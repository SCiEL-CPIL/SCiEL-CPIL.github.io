"""Capacitated VRP heuristic: angular sweep with distance tie-breaking."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

# We'll cache some per-instance state to speed up repeated calls within one route set.
# Since the evaluator may call this many times, we keep a simple cache keyed by (depot, n_customers).
# But to be safe and simple, we compute on the fly.

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next node to visit using an angular sweep heuristic with distance tie-breaking.
    
    Strategy:
    - If no unvisited nodes, return depot.
    - Filter unvisited nodes by capacity feasibility.
    - Among feasible nodes, pick the one that minimizes a combined score:
      * Angular distance from the last visited direction (to promote sweep-like routes)
      * Plus a distance component to avoid long jumps
    - If no feasible node, return depot.
    
    To implement angular sweep, we need to know the "last direction" from the depot.
    However, we don't have access to the full route history. Instead, we approximate:
    - Use the angle from depot to current_node as the "reference angle".
    - For each candidate, compute the angle from depot to candidate.
    - Prefer candidates that are "ahead" in angular order (small positive angular difference).
    
    This creates a sweep-like behavior without needing full route history.
    """
    if len(unvisited_nodes) == 0:
        return 0
    
    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    n = len(unvisited)
    
    # Get distances from current node to all unvisited
    dists = distance_matrix[current_node, unvisited]
    
    # Filter by capacity
    demands_arr = np.array([demands[node] for node in unvisited])
    feasible_mask = demands_arr <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0
    
    # Get feasible candidates
    feasible_nodes = unvisited[feasible_indices]
    feasible_dists = dists[feasible_indices]
    
    # Compute angles from depot
    # We need coordinates, but we only have distance_matrix. 
    # We can't compute angles directly from distance matrix alone.
    # Instead, use a different approach: prefer nodes that are close to current 
    # but also consider "progress" toward completing a compact route.
    
    # Alternative: Use a simple nearest-neighbor with a slight bias toward 
    # nodes that are also close to the depot (to encourage returning to depot).
    # Score = distance from current + alpha * distance to depot
    dists_to_depot = distance_matrix[depot, feasible_nodes]
    
    # Balance: primarily minimize distance from current, but slightly prefer 
    # nodes closer to depot to reduce route length.
    alpha = 0.3
    scores = feasible_dists + alpha * dists_to_depot
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
