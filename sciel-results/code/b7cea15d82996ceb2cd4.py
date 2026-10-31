"""
Strategy: Online Greedy with Regret-2 (Soft-Min)
Instead of pre-computing a full tour and following it, this strategy makes a
decision at each step based on:
1. Immediate cost: distance from current node to candidate.
2. Regret: How much worse the solution would be if we didn't pick the best
   immediate candidate. Specifically, we look at the cost of the 2nd best
   candidate from the current node and the cost from the candidate to the
   destination (if it were the last node, though here we just look at
   connectivity).
   
   A more robust regret formulation for TSP construction:
   For each candidate v in unvisited:
   cost_v = dist(current, v)
   
   We want to minimize the total tour length. A common heuristic is to look at
   the "regret" of not choosing v. If v is the best choice, the regret is low.
   If v is far from current, but all other options are even worse, we might
   still need to pick something.
   
   However, a simpler and effective structural change is using a "Lookahead"
   or "Regret" score:
   Score(v) = dist(current, v) + alpha * (min_dist_from_v_to_unvisited_excluding_v - min_dist_from_current_to_unvisited_excluding_v)
   
   Actually, let's use a standard Regret-2 heuristic adapted for TSP:
   For a candidate v:
   1. Find the two nearest neighbors of v in the unvisited set (including current? No, usually unvisited).
      Actually, for insertion, we consider the edges (current, v) and (v, next).
      Since we don't know 'next', we can approximate the future cost.
      
   Let's try a different structural approach: **Savings Heuristic (Clarke-Wright) style logic**
   applied online? No, that's for VRP.
   
   Let's go with **Regret-2** as described in literature for TSP construction:
   For each candidate v:
   r(v) = (2nd best distance from current to unvisited) - (best distance from current to unvisited)
   This is constant for all v, so it doesn't help distinguish.
   
   Correct Regret-2 for TSP:
   For each candidate v, consider the cost of inserting v.
   Let d_c_v = dist(current, v).
   Let d_v_u = min_{u in unvisited, u != v} dist(v, u).
   The "regret" is how much we lose by not picking v.
   
   Alternative: **Or-opt** style lookahead.
   
   Let's try a **Greedy with Soft-Min Global Structure** (Inspiration 3 style but implemented cleanly).
   Score(v) = dist(current, v) + beta * (soft_min(dist(v, unvisited)))
   This balances going to a nearby city vs going to a city that is well-connected to the rest (low soft-min distance to others).
   
   Wait, the current program pre-computes a tour. The evaluator calls select_next_node sequentially.
   The pre-computed tour approach is powerful because it uses 2-Opt.
   The "Online" approach must be fast and smart.
   
   Let's implement a **2-Step Lookahead Greedy**:
   For each candidate v:
     cost1 = dist(current, v)
     best_next = min_{u in unvisited, u!=v} dist(v, u)
     cost2 = best_next
     total = cost1 + cost2
     
   This is O(n^2) per step. With n=50, 50 steps, total ops ~ 50*50*50 = 125k. Very fast.
   This is a structural change from "Pre-compute + Follow" to "Local 2-Step Lookahead".
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _START_TIME, _DEADLINE
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 148.0  # Safety margin
        
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1 # Should not happen
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Exact 2-node lookahead
        u0 = unvisited_nodes[0]
        u1 = unvisited_nodes[1]
        # Path: current -> u0 -> u1 -> dest (if dest is not involved in cost yet? 
        # The contract says "while a traveling-salesman tour is built". 
        # Usually the last edge to destination/start is added at the end.
        # The evaluator scores "mean closed-tour length".
        # So we should consider the return to destination if possible, or just minimize path cost.
        # Since destination_node is passed, it implies we are building a path towards it?
        # Or is it a fixed start/end? "Traveling salesman tour" usually implies cycle.
        # The interface has destination_node. In many TSP formulations, start=dest.
        # Let's assume we are building a path and the final closure is handled by the evaluator or we should minimize the path cost.
        # Given the previous program used distance_matrix[unvisited_nodes[0], destination_node], it considers the link to destination.
        
        d_c_0 = distance_matrix[current_node, u0]
        d_c_1 = distance_matrix[current_node, u1]
        d_0_1 = distance_matrix[u0, u1]
        d_0_dest = distance_matrix[u0, destination_node]
        d_1_dest = distance_matrix[u1, destination_node]
        
        # Option 0: current -> u0 -> u1 -> dest
        cost_0 = d_c_0 + d_0_1 + d_1_dest
        # Option 1: current -> u1 -> u0 -> dest
        cost_1 = d_c_1 + d_0_1 + d_0_dest
        
        if cost_0 <= cost_1:
            return int(u0)
        else:
            return int(u1)

    # General case: 2-Step Lookahead
    # For each candidate v in unvisited_nodes:
    # 1. cost_to_v = dist(current, v)
    # 2. cost_from_v = min(dist(v, u) for u in unvisited if u != v)
    #    If no other u, use dist(v, destination_node)
    
    unvisited_arr = unvisited_nodes
    # Vectorized: get distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_arr]
    
    # We need min distance from each v to other unvisited nodes.
    # This is the diagonal of the submatrix, which we want to exclude.
    # Create a mask or set diagonal to inf.
    
    # Get the submatrix of distances between unvisited nodes
    # sub_dist[i, j] = dist(unvisited[i], unvisited[j])
    sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(sub_dist, np.inf)
    
    # For each v (row i), find the min distance to any other unvisited node
    min_dist_to_other = np.min(sub_dist, axis=1)
    
    # If there is only 1 unvisited node left (handled above), this would be inf.
    # If n_unvisited > 1, min_dist_to_other is valid.
    
    # Combine costs:
    # Total estimated cost if we pick v:
    # cost = dist(current, v) + min_dist_to_other[v]
    # This is a simple 2-step lookahead.
    
    # To make it more robust, we can add a small penalty for nodes that are "isolated" 
    # or use a soft-min. But simple min is a good structural change.
    
    total_costs = dists_from_current + min_dist_to_other
    
    # Pick the candidate with the minimum total cost
    best_idx = np.argmin(total_costs)
    return int(unvisited_arr[best_idx])

# EVOLVE-BLOCK-END
