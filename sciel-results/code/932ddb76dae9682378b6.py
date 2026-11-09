"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Compute distances from current to all candidates
    dist_to_candidate = distance_matrix[current_node, candidates]
    
    # Strategy: 2-Step Lookahead (Savings Heuristic)
    # Instead of just minimizing d(current, next), we minimize the cost of the 
    # segment: d(current, next) + d(next, destination).
    # This accounts for the "return cost" of the next node, preventing the 
    # tour from straying too far from the destination (start node) early on.
    
    # As the tour progresses (n decreases), the weight on the return cost 
    # should increase because the final edge (last node -> destination) 
    # becomes a larger fraction of the total tour length.
    
    # For early stages (large n), we are less worried about the immediate return 
    # to destination because there are many intermediate nodes. 
    # However, pure nearest neighbor is too myopic. 
    # A blend is best.
    
    # We use a dynamic weight that increases as n decreases.
    # When n is large (e.g., 50), weight is small (0.1).
    # When n is small (e.g., 2), weight is high (1.0 or more).
    
    # Let's try a linear interpolation of the weight based on n.
    # weight = 1.0 when n=1, weight = 0.1 when n=50?
    # Or simply: weight = 1.0 / n? No, that's too small for early steps.
    
    # Let's use a fixed weight that is strong enough to guide direction but 
    # not so strong that it ignores local geometry.
    # The "Savings" heuristic in cluster routing often uses:
    # s(i,j) = d(i,dest) + d(j,dest) - d(i,j)
    # We want to maximize savings, which is equivalent to minimizing:
    # d(i,dest) + d(j,dest) - s(i,j) ... wait.
    # Actually, the standard nearest neighbor with lookahead for TSP construction 
    # is: min_j [ d(current, j) + d(j, destination) ]
    
    # This is exactly what we are doing.
    # The question is the weight. 
    # If we use weight=1.0 always, we are minimizing d(current, j) + d(j, dest).
    # This is the "shortest path via one intermediate node" logic.
    
    # Let's test if a constant weight of 1.0 is better than the current variable weights.
    # The current program uses:
    # n <= 10: weight = n/10 (0.1 to 1.0)
    # n <= 20: weight = 0.5
    # n <= 40: weight = 0.15
    # n > 40: weight = 0.0 (pure NN)
    
    # The pure NN at n>40 is likely the weakest point. 
    # Let's apply the lookahead (weight=1.0) for ALL n, or at least a significant weight.
    
    dist_candidate_to_dest = distance_matrix[candidates, destination_node]
    
    # Hypothesis: The "dead end" problem is most severe when we pick a node 
    # that is far from the destination but close to the current node.
    # This is common in early stages.
    # Therefore, a constant weight of 1.0 (full lookahead) might be superior 
    # to the current variable weights, especially in the early/mid stages.
    
    scores = dist_to_candidate + dist_candidate_to_dest
    
    return int(candidates[np.argmin(scores)])
# EVOLVE-BLOCK-END
