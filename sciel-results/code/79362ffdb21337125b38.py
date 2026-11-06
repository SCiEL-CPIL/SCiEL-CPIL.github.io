"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Hard deadline guard: return depot to close route if time is running out
    if time.time() - _start_time > 140:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Precompute distances from current node and depot to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Strategy: Clarke-Wright Savings + Regret + Lookahead
    
    # 1. Savings Score:
    # The "saving" of visiting node i next is roughly the reduction in total route length
    # compared to closing the route now and visiting i later from depot.
    # Current route ends at current_node. If we go to depot now, cost is D(current, depot).
    # If we go to i then depot, cost is D(current, i) + D(i, depot).
    # Saving = D(current, depot) - (D(current, i) + D(i, depot))? No, that's not right.
    # Actually, standard savings for inserting i between j and k is D(j,i)+D(i,k)-D(j,k).
    # Here, we are appending i to the current route.
    # If we stop at current and go to depot: Cost = D(current, depot).
    # If we go to i then depot: Cost = D(current, i) + D(i, depot).
    # This doesn't capture the "saving" of merging routes well unless we think about the next node.
    
    # Alternative View:
    # We want to minimize total distance.
    # Let's use a composite score that balances:
    # a) Immediate cost: D(current, i)
    # b) Future cost penalty: How far is i from the *next* likely node or depot?
    # c) Regret: How much worse is i's best alternative compared to its second best?
    
    # Let's compute a "Regret" score.
    # For each node i, look at the distances to all other feasible nodes j.
    # The regret is the difference between the best and second-best distance from i to any other feasible node.
    # High regret means i is uniquely close to some node, so we should visit it soon before that node is taken.
    
    # However, computing pairwise distances for all feasible nodes at every step is O(N^2) per step, 
    # which is O(N^3) total. N=50, so 125,000 ops per instance. 8 instances = 1M ops. Very fast.
    
    # Let's try a simpler but effective heuristic:
    # Score = alpha * D(current, i) + beta * D(i, depot) - gamma * Savings(i)
    # Where Savings(i) approximates how much we save by not going back to depot yet.
    
    # Actually, let's look at the inspiration: "Regret-based selection ... 1.06 -> 1.114".
    # Let's implement Regret properly.
    
    # For each candidate i in feasible_nodes:
    # 1. Calculate distance from i to all other feasible nodes j (j != i).
    # 2. Find the min and second-min distance from i to other nodes.
    # 3. Regret(i) = min_dist - second_min_dist.
    #    (If only one other node, regret is 0 or large? If only one other, min=dist, second=inf? 
    #     Better: if count < 2, regret = 0).
    # 4. Also consider distance from current to i.
    
    # We want to MAXIMIZE regret (urgency) and MINIMIZE distance.
    
    # Let's compute distances from each feasible node to all other feasible nodes.
    # feasible_nodes is an array of indices.
    # We can extract the submatrix of distance_matrix for these nodes.
    
    # Optimization: Only do this if N is small enough, but N=50 is fine.
    
    # Get the submatrix
    # Note: distance_matrix is likely a 2D array or list of lists.
    # Assuming numpy array for speed.
    
    # Extract submatrix
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each row i in sub_dist, find min and second min
    # Set diagonal to inf to ignore self
    np.fill_diagonal(sub_dist, np.inf)
    
    # Get the two smallest values in each row
    # sort each row
    sorted_rows = np.sort(sub_dist, axis=1)
    
    min_dists = sorted_rows[:, 0]
    second_min_dists = sorted_rows[:, 1]
    
    # Regret = min - second_min
    # If a node has no other neighbors (isolated), min_dists will be inf.
    # Handle inf
    regret = min_dists - second_min_dists
    regret[np.isinf(regret)] = 0
    
    # Normalize regret to [0,1] for scaling? Or just use raw?
    # Raw regret can be large. Let's normalize by max regret or just use it as a priority.
    
    # Also, we need to consider the distance from current node to i.
    # dist_from_current is already computed.
    
    # Let's define a score to MINIMIZE.
    # Score = dist_from_current - lambda * regret
    # We want high regret (urgency) and low distance.
    
    # But we also need to ensure we don't get stuck.
    # Let's add a term for distance to depot to encourage closing if far.
    
    # Let's try:
    # Primary: Nearest Neighbor (dist_from_current)
    # Secondary: Regret (higher is better)
    
    # Combined Score = dist_from_current - 0.5 * regret
    
    # Let's also consider the "Savings" aspect from the original prompt inspiration.
    # Savings of visiting i now vs later:
    # If we visit i now, we pay D(current, i).
    # If we visit i later from depot (or another node), we pay more.
    # A proxy for "value" of i is how close it is to the cluster of remaining nodes.
    
    # Let's stick to Regret + Distance.
    
    # To balance scales, let's normalize regret.
    max_regret = np.max(regret) if len(regret) > 0 else 1
    if max_regret > 0:
        normalized_regret = regret / max_regret
    else:
        normalized_regret = regret
        
    # We want to pick the node with lowest distance, but boosted by regret.
    # Score = dist_from_current * (1 + 0.5 * (1 - normalized_regret)) ?
    # No, simpler: Score = dist_from_current - 0.2 * normalized_regret * max(dist_from_current)
    
    # Actually, let's look at the structure.
    # If regret is high, we want to visit it even if slightly farther.
    # Score = dist_from_current - weight * normalized_regret
    
    # Let's try weight = 0.3
    
    scores = dist_from_current - 0.3 * normalized_regret
    
    # Ties or close scores?
    # If scores are very close, prefer the one closer to depot (to keep route compact)?
    # Or just take argmin.
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
