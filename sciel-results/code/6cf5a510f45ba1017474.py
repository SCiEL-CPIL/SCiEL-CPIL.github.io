"""Capacitated vehicle routing heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer to visit, or return to depot (0) to close the route.
    
    Strategy: Multi-factor scoring that considers:
    1. Immediate distance from current node
    2. Distance to depot (to penalize going far from depot)
    3. One-step lookahead: minimum distance from candidate to other feasible nodes
    4. Capacity feasibility
    
    Returns 0 (depot) if no feasible node or if it's better to close the route.
    """
    if _time_exceeded():
        # Fallback to simple nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(min(feasible, key=lambda n: distance_matrix[current_node, n]))
    
    n_nodes = len(unvisited_nodes)
    if n_nodes == 0:
        return depot
    
    # Get feasible nodes (demand fits in remaining capacity)
    feasible = []
    for n in unvisited_nodes:
        if demands[n] <= rest_capacity:
            feasible.append(n)
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Score each feasible candidate
    # We want to minimize total route length, so we consider:
    # - dist(current -> candidate): immediate cost
    # - dist(candidate -> depot): cost to return after this (if we close here)
    # - dist(candidate -> best next feasible): one-step lookahead
    # - Balance between going far from depot and clustering
    
    best_score = float('inf')
    best_node = feasible[0]
    
    # Precompute distances from current to all feasible
    dist_from_current = np.array([distance_matrix[current_node, n] for n in feasible])
    
    # Distance from depot to each feasible (for return cost consideration)
    dist_to_depot = np.array([distance_matrix[n, depot] for n in feasible])
    
    # For one-step lookahead: for each candidate, find min distance to another feasible node
    # This is O(k^2) where k is number of feasible nodes
    k = len(feasible)
    min_next_dist = np.zeros(k)
    for i in range(k):
        ni = feasible[i]
        min_d = float('inf')
        for j in range(k):
            if i != j:
                nj = feasible[j]
                d = distance_matrix[ni, nj]
                if d < min_d:
                    min_d = d
        min_next_dist[i] = min_d
    
    # Weighted score:
    # Primary: distance from current (immediate travel)
    # Secondary: distance to depot (prefer staying close to depot to save return trip)
    # Tertiary: one-step lookahead (prefer nodes that are close to other feasible nodes)
    
    # Normalize components to [0,1] for weighting
    max_dist_current = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1
    max_dist_depot = np.max(dist_to_depot) if np.max(dist_to_depot) > 0 else 1
    max_next = np.max(min_next_dist) if np.max(min_next_dist) > 0 else 1
    
    # Weights: immediate distance is most important, then depot proximity, then clustering
    w1 = 0.5   # immediate distance
    w2 = 0.3   # distance to depot (lower is better, so we use this as penalty)
    w3 = 0.2   # one-step lookahead (lower is better)
    
    scores = (w1 * dist_from_current / max_dist_current + 
              w2 * dist_to_depot / max_dist_depot + 
              w3 * min_next_dist / max_next)
    
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
