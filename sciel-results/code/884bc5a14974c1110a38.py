"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track the path and avoid recalculating if possible, 
# though the function signature doesn't provide path history explicitly.
# We must infer the path or maintain it. 
# The interface is select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix).
# We do NOT have the full history of visited nodes in the arguments.
# However, we can infer the "start" node if we assume the tour starts at destination_node? 
# No, destination_node is the node we must return to.
# Usually in these constructive heuristics, the first call has unvisited_nodes = all nodes.
# We need to know the start node to calculate the "remainder" cost correctly.
# If we don't know the start node, we can't close the loop correctly for the lookahead.
# 
# Let's look at the interface again: 
# select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
# 
# If this is the first call, current_node is likely the start node.
# If we assume the tour is a cycle, the "destination_node" is the start node.
# 
# Strategy: 
# 1. On the first call, we identify the start node (current_node).
# 2. We maintain a global list of the path constructed so far.
# 3. For each candidate, we estimate the cost of completing the tour from that candidate back to the start node using the remaining unvisited nodes.
# 
# We need to be careful about the "first call" detection.
# We can use a global variable to store the start node and the path.

_start_node = None
_path = []
_call_count = 0

def _estimate_completion_cost(start, end, unvisited, dist_matrix):
    """
    Estimates the cost of a tour from 'start' to 'end' visiting all 'unvisited' nodes 
    using a Nearest Neighbor heuristic.
    """
    if not len(unvisited):
        return dist_matrix[start, end]
    
    current = start
    remaining = list(unvisited)
    total_dist = 0.0
    
    # Fast NN loop
    while remaining:
        # Find nearest neighbor in remaining
        # dist_matrix[current, remaining] is a vector
        # argmin gives index in remaining
        idx = np.argmin(dist_matrix[current, remaining])
        next_node = remaining[idx]
        total_dist += dist_matrix[current, next_node]
        current = next_node
        remaining.pop(idx)
        
    # Close the loop
    total_dist += dist_matrix[current, end]
    return total_dist

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_node, _path, _call_count
    
    # Initialize state on first call
    if _start_node is None:
        _start_node = current_node
        _path = [current_node]
        _call_count = 0
        
    _call_count += 1
    
    # If no unvisited nodes, we shouldn't be called, but handle gracefully
    if len(unvisited_nodes) == 0:
        return current_node # Fallback
        
    best_node = -1
    best_score = float('inf')
    
    # Lookahead Greedy: 
    # For each candidate, estimate the total tour length if we pick it next.
    # The partial tour is _path. 
    # If we pick candidate 'c', the new partial tour is _path + [c].
    # The remaining nodes are unvisited_nodes without 'c'.
    # We estimate the cost to complete from 'c' to destination_node (which is _start_node) 
    # visiting the remaining nodes.
    
    # Optimization: 
    # The cost of the path so far is fixed. 
    # The decision depends on:
    # 1. dist(current_node, c)  [Edge from current to candidate]
    # 2. Estimated completion cost from c to destination_node using remaining nodes.
    
    # Note: destination_node should be _start_node.
    
    for c in unvisited_nodes:
        # Cost to get to c
        edge_cost = distance_matrix[current_node, c]
        
        # Remaining nodes after picking c
        mask = unvisited_nodes != c
        remaining_nodes = unvisited_nodes[mask]
        
        # Estimate completion cost from c to destination_node
        # Using NN heuristic on remaining_nodes
        completion_cost = _estimate_completion_cost(c, destination_node, remaining_nodes, distance_matrix)
        
        total_estimated_cost = edge_cost + completion_cost
        
        if total_estimated_cost < best_score:
            best_score = total_estimated_cost
            best_node = c
            
    # Update state
    _path.append(best_node)
    
    return int(best_node)

# EVOLVE-BLOCK-END
