import numpy as np
import time
from collections import deque

def priority(instance):
    start_time = time.time()
    deadline = 120  # Safety margin, though computation is trivial
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # If n is 0, return empty
    if n == 0:
        return np.array([])
        
    # 1. Build successor graph
    # successors[i] contains all j such that i is a predecessor of j
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for j in range(n):
        preds = predecessors[j]
        for i in preds:
            successors[i].append(j)
            in_degree[i] += 1 # Wait, in_degree for topological sort usually counts incoming edges.
            # Here, for reverse topo sort (sinks first), we need to know when a node's successors are done.
            # Let's just do standard Topological Sort to get order, then reverse.
            
    # Standard Topological Sort (Kahn's)
    # in_degree[i] = number of predecessors of i
    # We need to reset in_degree logic if we used it above incorrectly.
    # Let's recompute in_degree properly for Kahn's.
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            in_degree[i] += 1 
            # No, in_degree[i] is number of nodes that point TO i.
            # So if i is pred of j, edge i->j.
            # So in_degree[j] += 1.
            
    # Correct In-Degree Calculation:
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            in_degree[j] += 1
            
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # It's possible the graph has cycles (invalid input for ALBP), but assuming valid DAG.
    # If topo_order length < n, there's a cycle. We should handle gracefully or assume valid.
    # Assuming valid DAG as per standard ALBP problems.
    
    # 2. Calculate Positional Weights
    # w(i) = t(i) + sum(w(j) for j in successors(i))
    # Process in reverse topological order (sinks to sources)
    pos_weights = times.copy()
    
    # Reverse topo order
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
            
    # 3. Refine Priority
    # Add a small bonus for tasks with more immediate successors.
    # This helps "branching" tasks to be scheduled earlier, potentially
    # unlocking more tasks in subsequent stations and reducing the total
    # number of stations needed.
    # The weight 0.1 is small enough to not override RPW significantly
    # but large enough to break ties in favor of high-degree nodes.
    
    # Check time
    if time.time() - start_time > deadline - 5:
         # Fallback to simple LPT if somehow slow (unlikely)
         return times
         
    # Return refined priority
    return pos_weights + 0.1 * np.array([len(succs) for succs in successors])
