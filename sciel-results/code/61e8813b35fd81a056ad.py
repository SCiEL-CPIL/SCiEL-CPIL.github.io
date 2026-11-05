# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    successors = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Calculate Topological Order
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Calculate Downstream Work (Ranked Positional Weight)
    # downstream[i] = times[i] + sum(downstream[s] for s in successors[i])
    # Process in reverse topological order
    downstream = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        total = times[i]
        for s in successors[i]:
            total += downstream[s]
        downstream[i] = total
        
    # Calculate Upstream Work
    # upstream[i] = times[i] + sum(upstream[p] for p in predecessors[i])
    # Process in topological order
    upstream = np.zeros(n, dtype=float)
    for i in topo_order:
        total = times[i]
        for p in predecessors[i]:
            total += upstream[p]
        upstream[i] = total
        
    # Buckley's Method: Sum of Upstream and Downstream Work
    priorities = downstream + upstream
    
    return priorities
# EVOLVE-BLOCK-END
