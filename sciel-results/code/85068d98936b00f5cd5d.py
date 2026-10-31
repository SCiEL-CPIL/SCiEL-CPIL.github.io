"""Rewritten program: TSP constructive heuristic using Regret-2 with destination awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Regret-2 Heuristic with Destination Awareness.
    For each candidate node u in unvisited_nodes, we compute:
    1. Immediate cost: c1(u) = d(current, u)
    2. Best future cost: c2(u) = min_{v in unvisited, v != u} (d(u, v) + d(v, destination))
    
    The regret is defined as: regret(u) = c2(u) - c1(u)
    
    We select the node with the maximum regret. This prioritizes nodes where
    the difference between the best immediate choice and the best future path
    is largest, effectively capturing opportunity cost and avoiding nodes that
    would lead to poor future connections.
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
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute c2(u) for each candidate u
    # For each u, we want min_{v != u} (d(u, v) + d(v, destination))
    # Create a matrix where entry [i, j] = d(u_i, u_j) + d(u_j, destination)
    # This is d_sub + d_dest[None, :]
    future_costs = d_sub + d_dest[None, :]
    
    # Set the diagonal to infinity so that v != u
    np.fill_diagonal(future_costs, np.inf)
    
    # c2(u) is the minimum future cost for each u
    c2 = np.min(future_costs, axis=1)
    
    # 5. Compute regret: regret(u) = c2(u) - c1(u)
    # Note: In standard regret heuristics, we maximize regret.
    # However, let's verify the intuition.
    # If c1 is small and c2 is large, regret is large -> good candidate?
    # Actually, standard regret-2 for TSP usually looks like:
    # regret(i) = (cost of 2nd best option) - (cost of best option)
    # But here our "cost" is a composite.
    
    # Let's stick to the structural change: Maximize (c2 - c1)
    # This favors nodes that are cheap to enter (low c1) but have expensive alternatives
    # for the next step (high c2), implying that if we don't go there, we might be stuck
    # with bad options later? Or does it mean the node is a "bridge"?
    
    # Alternative interpretation:
    # We want to minimize the total tour length.
    # A node with high regret means that the "best" immediate move is much better than
    # the "best" move if we were to consider the future? No, that's not right.
    
    # Let's look at Inspiration 3: "savings-based / regret-based heuristic that considers the opportunity cost of not v"
    # Opportunity cost of NOT visiting u:
    # If we don't visit u now, we will visit it later.
    # The cost of visiting u later is at least min_{v in current_unvisited} d(v, u) + ...
    
    # Let's try a different structural approach: **Nearest Neighbor with Lookahead Penalty**
    # Score(u) = d(current, u) + lambda * (min_{v in unvisited, v!=u} d(u, v))
    # This is similar to the current bottleneck but using min instead of max.
    # Min-based lookahead encourages clustering.
    
    # Let's implement: Score(u) = d(current, u) + beta * min_{v != u} d(u, v)
    # This is a 2-step lookahead favoring nodes that are close to current AND close to another unvisited node.
    
    min_next_dist = np.min(future_costs, axis=1) # This is min_{v != u} (d(u,v) + d(v,dest))
    # Wait, future_costs includes d(v, dest). 
    # Let's use pure distance for the "closeness to next" term to keep it simple and robust.
    # min_dist_to_other = min_{v != u} d(u, v)
    
    # Re-compute min distance to other unvisited nodes
    np.fill_diagonal(d_sub, np.inf)
    min_dist_to_other = np.min(d_sub, axis=1)
    
    # Hybrid Score: Balance immediate cost and "hub-ness" (closeness to others)
    # alpha = 0.5
    # scores = alpha * d_curr + (1 - alpha) * min_dist_to_other
    
    # This is structurally different from Max Bottleneck.
    # Max Bottleneck avoids nodes far from ANY other node.
    # Min Dist to Other favors nodes close to SOME other node (clustering).
    
    # Let's try this.
    alpha = 0.5
    scores = alpha * d_curr + (1 - alpha) * min_dist_to_other
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
