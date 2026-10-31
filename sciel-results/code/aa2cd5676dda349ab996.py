"""Improved TSP constructive heuristic with lookahead and angle-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache to avoid recomputing expensive lookups
_tsp_cache = {}
_DEADLINE_MARGIN = 1.0  # seconds before the 150s limit we stop trying to improve


def _compute_angle_score(current, candidate, prev_node, dm, n):
    """
    Compute a penalty based on the turning angle at 'current'.
    If prev_node is None (first step), return 0.
    We want to avoid sharp turns that create crossings.
    """
    if prev_node is None:
        return 0.0
    
    # Vectors: current -> prev and current -> candidate
    # But we don't have coordinates, only distances.
    # We can approximate "straightness" using the triangle inequality or
    # by looking at the distance between prev and candidate relative to
    # dist(current, prev) + dist(current, candidate).
    # If dist(prev, candidate) is close to |dist(current, prev) - dist(current, candidate)|,
    # they are roughly collinear (same direction).
    # If dist(prev, candidate) is close to dist(current, prev) + dist(current, candidate),
    # they are in opposite directions (sharp turn).
    
    d_cp = dm[current, prev_node]
    d_cc = dm[current, candidate]
    d_pc = dm[prev_node, candidate]
    
    # Avoid division by zero
    if d_cp < 1e-10 or d_cc < 1e-10:
        return 0.0
        
    # Cosine of the angle at 'current' using Law of Cosines:
    # d_pc^2 = d_cp^2 + d_cc^2 - 2 * d_cp * d_cc * cos(theta)
    # cos(theta) = (d_cp^2 + d_cc^2 - d_pc^2) / (2 * d_cp * d_cc)
    
    cos_theta = (d_cp * d_cp + d_cc * d_cc - d_pc * d_pc) / (2.0 * d_cp * d_cc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # theta = arccos(cos_theta)
    # We want to penalize large angles (turns).
    # A straight line has theta=0 or pi?
    # If moving from prev to current to candidate:
    # Vector u = current - prev. Vector v = candidate - current.
    # Angle between u and v.
    # If they are in the same direction (straight), the angle is 0.
    # If they are opposite (U-turn), the angle is pi.
    # Let's check the cosine formula again.
    # Triangle vertices: P, C, Q.
    # Side PC = d_cp. Side CQ = d_cc. Side PQ = d_pc.
    # Angle at C is between vector C->P and C->Q? No, the path is P->C->Q.
    # So we care about the angle between vector P->C and C->Q.
    # Let vector a = C - P. Let vector b = Q - C.
    # The angle between a and b is what we want to minimize (0 is best).
    # The triangle sides are |a|=d_cp, |b|=d_cc, and |P-Q| = |C-P + C-Q|? No.
    # P, C, Q. Distance P-Q is d_pc.
    # Vector P->Q = Vector P->C + Vector C->Q = a + b.
    # |a+b|^2 = |a|^2 + |b|^2 + 2|a||b|cos(theta)
    # d_pc^2 = d_cp^2 + d_cc^2 + 2 d_cp d_cc cos(theta)
    # cos(theta) = (d_pc^2 - d_cp^2 - d_cc^2) / (2 d_cp d_cc)
    
    cos_theta = (d_pc * d_pc - d_cp * d_cp - d_cc * d_cc) / (2.0 * d_cp * d_cc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # theta is the angle between the incoming direction (P->C) and outgoing direction (C->Q).
    # We want theta to be small (straight).
    # cos(theta) close to 1 means theta close to 0 (good).
    # cos(theta) close to -1 means theta close to pi (bad, U-turn).
    
    # Penalty: 1 - cos(theta) is 0 for straight, 2 for U-turn.
    return 1.0 - cos_theta


def _select_next_node_impl(current_node, destination_node, unvisited_nodes, distance_matrix, prev_node=None, time_budget=None):
    """
    Core selection logic.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # 1. Lookahead: For each candidate, what is the cost to go to the nearest unvisited node AFTER it?
    # We approximate the "future cost" by the distance to the destination (if it's the last one)
    # or by the average distance to other unvisited nodes?
    # A better heuristic:
    # Score(c) = w1 * dist(current, c) + w2 * min_{u in unvisited\{c}} dist(c, u)
    # If c is the last one to visit, min is dist(c, destination).
    
    n_unvisited = len(unvisited_nodes)
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate lookahead cost
    # For each candidate i, find the min distance to any other unvisited node j != i.
    # This is O(N^2) per step, which might be slow for N=50.
    # N=50, steps=50. 50*50*50 = 125k ops. This is fast in NumPy.
    
    # Matrix of distances between unvisited nodes
    # sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # min_dist_to_other = np.min(sub_dm, axis=1)
    # But we need to exclude the node itself.
    # We can set diagonal to inf, then take min.
    
    sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dm, np.inf)
    
    # If only 2 nodes left, the "other" node is the only one.
    # min_dist_to_other[i] is the distance to the other node.
    # If i is the last one, the next step is to destination.
    # The heuristic usually handles this by considering the destination in the lookahead
    # or by a specific case for the last few nodes.
    
    # Let's refine:
    # If len(unvisited) == 2:
    #   Candidates: A, B.
    #   If pick A, next is B (if B is not dest) then Dest. Or if B is Dest, next is Dest.
    #   Actually, the function is called step-by-step.
    #   If 2 nodes left, we pick one. Then 1 node left, we pick it. Then 0 left.
    #   The "lookahead" for the first of the two should consider the cost to go to the second,
    #   and then to the destination.
    
    # General lookahead:
    # Cost if pick candidate c:
    # 1. dist(current, c)
    # 2. min_{u in unvisited\{c}} dist(c, u)  -- this is the "next step" cost
    #    BUT if c is the last one to be visited from unvisited, the next is destination.
    #    We don't know the order yet.
    #    
    # A common heuristic (Floyd-Warshall style or just 1-step lookahead):
    # Score(c) = dist(current, c) + min( dist(c, dest), min_{u in unvisited\{c}} dist(c, u) )
    # This assumes that after visiting c, we will either go to dest (if c is the last) 
    # or go to the closest other unvisited node.
    
    min_to_other = np.min(sub_dm, axis=1)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # The "next cost" for candidate i is min( min_to_other[i], dist_to_dest[i] )?
    # Not exactly. If we pick c, we MUST visit all other unvisited nodes before dest?
    # No, dest is the return point. The tour is Start -> ... -> Unvisited -> Dest -> Start?
    # Wait, the problem says "closed-tour".
    # Usually TSP: Start at 0, visit all, return to 0.
    # Here: select_next_node(current, destination, ...).
    # This implies the tour ends at destination.
    # So the sequence is: current -> ... -> destination.
    # And presumably destination -> current_start? Or is destination the start?
    # "destination_node" suggests it's the end of the path.
    # If it's a closed tour, destination is likely the start node (0).
    
    # If we pick c, the remaining unvisited nodes must be visited.
    # The cost is dist(current, c) + (cost to visit remaining from c).
    # Approximation: dist(current, c) + min_{u in unvisited\{c}} dist(c, u)
    # This is the 1-step lookahead.
    
    # However, if c is the ONLY unvisited node, the next step is destination.
    # So next_cost = dist(c, destination).
    
    # Let's compute next_cost for each candidate i:
    next_costs = np.zeros_like(dists)
    for i in range(n_unvisited):
        if n_unvisited == 1:
            next_costs[i] = distance_matrix[unvisited_nodes[i], destination_node]
        else:
            # Min of distance to other unvisited nodes OR distance to destination?
            # If we go to destination early, we might not visit others.
            # But the algorithm enforces visiting all unvisited nodes.
            # So we cannot go to destination until unvisited is empty.
            # Therefore, the next step MUST be to another unvisited node.
            # UNLESS n_unvisited == 1, then next is destination.
            
            if n_unvisited == 2:
                # If we pick one, the other is the last one.
                # After picking c, we pick the other u. Then we go to destination.
                # So cost = dist(current, c) + dist(c, u) + dist(u, destination)
                # We can compute this exactly for the 2 cases.
                # But the function returns ONE node.
                # We should score c based on the total cost of the remaining path if we pick c.
                
                # Let's generalize the lookahead to 2 steps if n_unvisited is small.
                pass
            
            # For general n_unvisited > 1:
            # Next step is to some u in unvisited\{c}.
            # To minimize, we pick the closest u.
            # So next_cost = min_to_other[i]
            next_costs[i] = min_to_other[i]
    
    # Special case: if n_unvisited == 2, we can be more precise.
    if n_unvisited == 2:
        # Candidates: u0, u1.
        # If pick u0: cost = dist(cur, u0) + dist(u0, u1) + dist(u1, dest)
        # If pick u1: cost = dist(cur, u1) + dist(u1, u0) + dist(u0, dest)
        
        u0, u1 = unvisited_nodes[0], unvisited_nodes[1]
        
        cost_0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost_1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        
        if cost_0 < cost_1:
            return int(u0)
        else:
            return int(u1)

    # Special case: if n_unvisited == 1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # General case:
    # Score = dist(current, c) + next_costs[i]
    # Add angle penalty if prev_node is available
    scores = dists + next_costs
    
    if prev_node is not None:
        # Compute angle penalty for each candidate
        # This is vectorized?
        # d_cp is scalar. d_cc is vector. d_pc is vector.
        d_cp = distance_matrix[current_node, prev_node]
        d_cc = distance_matrix[current_node, unvisited_nodes]
        d_pc = distance_matrix[prev_node, unvisited_nodes]
        
        cos_thetas = (d_pc * d_pc - d_cp * d_cp - d_cc * d_cc) / (2.0 * d_cp * d_cc)
        cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
        angle_penalties = 1.0 - cos_thetas
        
        # Weight the angle penalty
        # How important is it?
        # Let's try a weight of 0.5 * max_distance? Or just 0.5?
        # Distances are Euclidean, likely O(1).
        # Let's try weight 0.1 initially, or maybe 0.5.
        # A sharp turn adds length.
        # Let's use a weight that scales with the distance.
        # Or just a fixed weight.
        # Let's try 0.5.
        scores += 0.5 * angle_penalties

    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Public interface.
    """
    # Check for empty
    if len(unvisited_nodes) == 0:
        return -1
        
    # Simple cache key?
    # The state is defined by current, dest, unvisited set, and matrix.
    # Matrix is large, so we can't cache by matrix.
    # We can cache by (current, dest, frozenset(unvisited)) if the matrix is static per instance.
    # But the matrix is passed in.
    # We can hash the current, dest, and unvisited tuple.
    # If the same state is reached via different paths (unlikely in greedy), we can reuse.
    # But greedy is deterministic, so same state implies same history? No.
    # However, for a single instance, the matrix is fixed.
    # We can cache based on (id(distance_matrix), current, dest, tuple(unvisited)).
    
    # Given the 150s limit and N=50, the computation is fast enough.
    # 50 steps * 50 candidates * O(1) or O(N) lookahead.
    # O(N^2) lookahead is 2500 ops per step. 50*2500 = 125k ops.
    # This is negligible in Python/NumPy.
    
    # We don't have prev_node in the interface.
    # We can infer it? No, the interface doesn't provide history.
    # So we cannot use the angle penalty based on the actual previous node.
    # UNLESS we maintain a global state?
    # The function is called sequentially.
    # We can store the last node in a global variable?
    # This is dangerous if multiple instances are evaluated in parallel or interleaved.
    # But the task says "eight seeded uniform 50-city instances".
    # Likely evaluated sequentially.
    # However, re-entering the function with a different instance would break the state.
    # We can detect a new instance by checking if current_node is 0 (start) or if the matrix changed.
    
    # Let's try to maintain a simple state:
    # global _last_node, _last_matrix_id
    # If _last_matrix_id != id(distance_matrix):
    #     _last_node = None
    #     _last_matrix_id = id(distance_matrix)
    # Else:
    #     prev_node = _last_node
    #     _last_node = current_node
    
    # This is a bit hacky but might work if the evaluator is sequential.
    
    global _last_node, _last_matrix_id
    if _last_matrix_id != id(distance_matrix):
        _last_node = None
        _last_matrix_id = id(distance_matrix)
    
    prev_node = _last_node
    # Update state for next call
    _last_node = current_node
    
    return _select_next_node_impl(current_node, destination_node, unvisited_nodes, distance_matrix, prev_node)

# Initialize globals
_last_node = None
_last_matrix_id = None
# EVOLVE-BLOCK-END
