"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Calculates priority scores for tasks based on Positional Weight.
    Positional weight is the sum of the task's processing time and all its successors' times.
    Higher weight implies higher priority.
    """
    # Deadline guard: The computation is very fast for N=30, but we add a check just in case.
    # The actual limit is 120s, but we should return quickly.
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    # predecessors is likely a list of lists, where predecessors[i] is a list of task indices that must precede i.
    # Or it might be a different format. The prompt says "instance has times, predecessors, cycle_time".
    # Standard format in these tasks is usually a list of lists of predecessors.
    
    # Build successors list
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        # Ensure predecessors[i] is iterable
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological Sort (Kahn's Algorithm)
    # We need reverse topological order to compute positional weights (sum of self + successors)
    # Or we can do forward topological sort and then process in reverse.
    
    queue = []
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topological_order = []
    while queue:
        # Use a list as a queue. For small N, pop(0) is fine, or use collections.deque
        u = queue.pop(0)
        topological_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If there's a cycle, topological_order will be shorter than n. 
    # The problem statement implies a valid DAG.
    # If not all nodes are processed, we might have a cycle. 
    # For robustness, if len(topological_order) < n, we just use the ones we have or fallback.
    # Assuming valid DAG.
    
    # Calculate Positional Weights
    # w[i] = times[i] + sum(w[j] for j in successors[i])
    weights = np.zeros(n, dtype=float)
    
    # Process in reverse topological order
    for u in reversed(topological_order):
        w_u = times[u]
        for v in successors[u]:
            w_u += weights[v]
        weights[u] = w_u
        
    # If any tasks were not in the topological order (due to cycles, though unlikely),
    # their weights remain 0. We might want to set them to their own time or something.
    # But usually, the input is a valid DAG.
    
    return weights

# EVOLVE-BLOCK-END
