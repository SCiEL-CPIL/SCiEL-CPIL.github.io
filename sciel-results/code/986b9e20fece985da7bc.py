"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Score each assembly task by its positional criticality (Earliest Start Time).
    
    The goal is to minimize the number of stations. This heuristic prioritizes
    tasks that are early in the dependency chain (low Earliest Start Time), 
    which helps in filling the initial stations and establishing the 
    foundation for subsequent tasks. This is implemented by returning the 
    negative of the Earliest Start Time (EST), so that lower EST tasks have 
    higher priority scores.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor list and in-degree
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], (list, tuple, np.ndarray)) else [predecessors[i]]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Compute topological order using Kahn's algorithm
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg_copy[succ] -= 1
            if in_deg_copy[succ] == 0:
                queue.append(succ)
    
    # Compute EST (Earliest Start Time) in topological order
    # EST[i] = max(EST[p] + times[p] for p in predecessors[i])
    # If no predecessors, EST[i] = 0
    # We track Earliest Completion (EC) of predecessors to determine EST.
    EC = np.zeros(n, dtype=float) # Earliest Completion
    EST = np.zeros(n, dtype=float) # Earliest Start
    
    for node in topo_order:
        if in_degree[node] > 0:
            # Find max EC of predecessors
            max_pred_ec = 0.0
            preds = predecessors[node] if isinstance(predecessors[node], (list, tuple, np.ndarray)) else [predecessors[node]]
            for p in preds:
                if EC[p] > max_pred_ec:
                    max_pred_ec = EC[p]
            EST[node] = max_pred_ec
        else:
            EST[node] = 0.0
            
        EC[node] = EST[node] + times[node]
        
    # Prioritize tasks with LOWEST EST (Start Phase)
    # Return negative EST so that higher score = lower EST
    # Normalizing by cycle_time as per directive
    scores = -EST / cycle_time
    
    return scores
# EVOLVE-BLOCK-END
