import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]  # list of lists, preds[i] = list of predecessor indices for task i
    n = len(times)
    
    # Build successor graph
    succs = [set() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].add(i)
    
    # Compute number of immediate successors
    succ_count = np.zeros(n)
    for i in range(n):
        succ_count[i] = len(succs[i])
    
    # Compute longest path from each task to any sink (including own time)
    # Use topological sort (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1  # wait, this is wrong
    
    # Actually, let me recompute. in_degree[i] = number of predecessors of i
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])
    
    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute longest path from each node to a sink
    # Process in reverse topological order
    longest_to_sink = np.zeros(n)
    for u in reversed(topo_order):
        if not succs[u]:
            longest_to_sink[u] = times[u]
        else:
            longest_to_sink[u] = times[u] + max(longest_to_sink[v] for v in succs[u])
    
    # Compute total downstream work (sum of times of all successors, transitive)
    # This is more expensive but could be useful
    # For now, let's use a combination
    
    # Priority: higher is better
    # Weight: successors count heavily, then downstream path, then own time
    score = (
        1.0 * succ_count + 
        0.3 * longest_to_sink + 
        0.1 * times
    )
    
    return score
