"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track time budget and cache
_start_time = None
_time_budget = 140.0  # Leave 10s margin for 150s limit
_deadline_reached = False


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _check_deadline():
    global _deadline_reached
    if time.time() - _start_time > _time_budget:
        _deadline_reached = True
    return _deadline_reached


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _deadline_reached
    
    if _check_deadline():
        # Fallback: nearest neighbor to destination or current
        # Just pick nearest to current to be safe and fast
        if len(unvisited_nodes) == 0:
            return -1
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    n = len(unvisited_nodes)
    
    if n == 1:
        return int(unvisited_nodes[0])
    
    # If only a few nodes left, use exact optimization or simple heuristic
    if n <= 5:
        # For small n, we can try to be smarter, but for now, nearest neighbor with a twist
        # Actually, for n<=5, let's just use nearest neighbor to current, it's fast
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Heuristic: Score each candidate node
    # We want to minimize the increase in tour length.
    # Current partial tour ends at current_node.
    # The tour will eventually close by connecting the last node (currently unknown, but destination_node is the start)
    # Wait, the interface says destination_node is passed. In TSP construction, usually we start at a node, build the path, and return to start.
    # So destination_node is likely the starting node of the tour.
    # When we pick a node `u` next:
    # 1. We add edge (current_node, u). Cost: d[current_node, u]
    # 2. We remove the "potential" closing edge? No, the closing edge is only added at the very end.
    # However, a common heuristic is to consider the "penalty" of leaving the destination far away for later.
    
    # Let's try a "Nearest Neighbor with Destination Bias"
    # Score(u) = d(current, u) + alpha * d(u, destination)
    # Or better: d(current, u) - d(current, destination) ? No.
    
    # Let's look at the structure.
    # If we go to u, the current path length increases by d(current, u).
    # The remaining path must connect u -> ... -> destination.
    # A lower bound on the remaining path is d(u, destination).
    # So estimated final length = current_length + d(current, u) + d(u, destination).
    # Since current_length is constant for all candidates, we minimize d(current, u) + d(u, destination).
    
    # This is equivalent to finding the node u that minimizes the sum of distances to current and destination.
    # This creates a "corridor" towards the destination.
    
    # Let's compute this score.
    # dist_current = distance_matrix[current_node, unvisited_nodes]
    # dist_dest = distance_matrix[destination_node, unvisited_nodes]
    # scores = dist_current + dist_dest
    
    # However, if we are close to the end, we should prioritize closing the loop.
    # If n is small, the d(u, destination) term dominates.
    # If n is large, d(current, u) dominates (nearest neighbor).
    
    # Let's try a dynamic alpha?
    # Or just the sum.
    # Sum = d(c, u) + d(d, u). This is the length of the triangle c-u-d minus the base c-d.
    # Minimizing this keeps the path "tight" between current and destination.
    
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Weights: 
    # Early on, we want to explore. Late on, we want to close.
    # But the sum heuristic naturally balances this?
    # If u is far from c and d, score is high.
    # If u is near c but far from d, score is high.
    # If u is near d but far from c, score is high.
    # It prefers nodes that are in the "middle" or near the direct path?
    
    # Actually, for TSP, going towards the destination is good.
    # Let's try: score = dist_current + 0.5 * dist_dest
    # Or just sum.
    
    # Let's try the sum first.
    scores = dist_current + dist_dest
    
    # To avoid getting stuck in local minima or making bad choices late,
    # we can add a small random noise? No, deterministic is better for reproducibility and stability.
    
    # One issue: if destination_node is the same as current_node (first step?), 
    # then scores = 2 * dist_current. It becomes nearest neighbor. Correct.
    
    # If unvisited_nodes is almost empty, dist_dest dominates.
    # Example: 2 nodes left: u1, u2.
    # We pick u that minimizes d(c,u) + d(d,u).
    # Then we go to the other one, then back to d.
    # Total added: d(c, u1) + d(u1, u2) + d(u2, d)
    # vs d(c, u2) + d(u2, u1) + d(u1, d)
    # Our heuristic picks u1 if d(c,u1)+d(d,u1) < d(c,u2)+d(d,u2).
    # This doesn't account for d(u1, u2).
    # But it's a reasonable proxy.
    
    idx = np.argmin(scores)
    return int(unvisited_nodes[idx])

# EVOLVE-BLOCK-END
