"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Module-level cache to avoid recomputing expensive structures across calls
_cache = {}


def _compute_mst_weight(nodes, distance_matrix):
    """Compute MST weight for a set of nodes using Prim's algorithm (O(n^2))."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Use nodes as indices into a local array
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find the node with minimum distance
        # We track which node is added
        best_idx = -1
        best_val = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_val:
                best_val = min_dist[i]
                best_idx = i
        
        if best_idx == -1:
            break
            
        in_mst[best_idx] = True
        total_weight += best_val
        
        # Update distances
        node_global = nodes[best_idx]
        for j in range(n):
            if not in_mst[j]:
                node_global_j = nodes[j]
                d = distance_matrix[node_global, node_global_j]
                if d < min_dist[j]:
                    min_dist[j] = d
                    
    return total_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a hybrid heuristic:
    1. Compute MST weight of remaining unvisited nodes.
    2. For each candidate, estimate the change in MST weight if that node were visited.
    3. Combine with distance to current node and distance to destination.
    
    The idea: visiting a node that is a 'leaf' or 'articulation point' in the MST
    structure might reduce future MST weight more, or we want to keep the remaining
    graph well-connected.
    
    Actually, a simpler and faster proxy:
    - We want to pick a node that minimizes distance to current.
    - But also, we want to avoid creating long edges later.
    - A good heuristic is the "2-opt" aware greedy: look ahead 1 step.
    
    Let's try a different structural approach:
    Compute the MST of the *unvisited* nodes.
    If we visit node v, the remaining nodes are unvisited \ {v}.
    The MST of the remaining nodes might be cheaper if v was a 'bridge' or high-degree node?
    
    Actually, the MST weight of the *remaining* graph is what matters for the future.
    If we remove v, the MST weight of the rest might go UP (if v was helping connect) or stay same.
    We want to minimize the *final* tour.
    
    Let's use a simpler but effective heuristic:
    Score(candidate) = dist(current, candidate) + alpha * dist(candidate, destination)
    where alpha is a small factor to bias towards the destination.
    
    However, the directive suggests MST-based.
    Let's compute the MST weight of the current unvisited set.
    Then for each candidate c, compute the MST weight of unvisited \ {c}.
    Prefer candidates that result in a *lower* MST weight for the remainder?
    Or prefer candidates whose removal *increases* the MST weight the least?
    If removing a node increases MST weight a lot, that node was crucial for connectivity.
    Maybe we should visit nodes that are *less* crucial for the MST of the remainder,
    so that the remainder stays well-connected (low MST weight).
    
    So:
    base_mst = MST(unvisited)
    For each c in unvisited:
        new_mst = MST(unvisited \ {c})
        penalty = new_mst - base_mst  # How much does the MST weight increase?
        If penalty is small, c is not critical for the MST of the rest.
        If penalty is large, c is critical.
        
    We want to pick c that minimizes:
    dist(current, c) + beta * max(0, new_mst - base_mst)
    
    This encourages picking nodes that are close AND don't break the connectivity of the remaining graph too much.
    
    But computing MST for each candidate is O(N^3) total per step, which is too slow for N=50?
    N=50, 50 steps. 50 * 50 MSTs of size ~50.
    MST is O(N^2). So 50 * 50 * 2500 = 6.25M ops. This is fast.
    
    Let's implement this.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Edge case: only one node left
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0  # 2 seconds per call max, should be much less
    
    # 1. Compute base MST weight
    base_mst = _compute_mst_weight(unvisited_nodes, distance_matrix)
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # 2. Evaluate each candidate
    for i in range(n_unvisited):
        if time.time() > deadline - 0.5:
            break
            
        c = unvisited_nodes[i]
        
        # Distance from current
        d_current = distance_matrix[current_node, c]
        
        # Distance to destination (optional, but helps close the loop)
        d_dest = distance_matrix[c, destination_node]
        
        # Compute MST of remaining nodes (excluding c)
        remaining = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
        if len(remaining) == 0:
            new_mst = 0.0
        else:
            new_mst = _compute_mst_weight(remaining, distance_matrix)
            
        # Penalty for increasing MST weight of the remainder
        # If new_mst > base_mst, it means removing c made the MST heavier (c was helping).
        # We want to avoid this.
        mst_penalty = max(0.0, new_mst - base_mst)
        
        # Heuristic score
        # Weight the MST penalty heavily?
        # d_current is in same units as mst_penalty (both are sums of distances).
        # Let's try: score = d_current + 0.5 * mst_penalty
        # Or maybe score = d_current + mst_penalty
        
        score = d_current + mst_penalty
        
        if score < best_score:
            best_score = score
            best_node = c
            
    return int(best_node)

# EVOLVE-BLOCK-END
