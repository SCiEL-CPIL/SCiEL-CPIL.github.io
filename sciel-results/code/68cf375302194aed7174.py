"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Compute priority scores using Bottom Level (BL) heuristic.
    
    The Bottom Level of a task is the length of the longest path from that task
    to the end of the precedence graph, including the task's own duration.
    Tasks with higher Bottom Levels are on the critical path and should be
    scheduled as early as possible to minimize the total number of stations.
    
    This approach differs from sum-based weights by focusing on the maximum
    delay constraint rather than the total workload unlocked.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topological sort failed (cycle), fallback to times
    if len(topo_order) < n:
        return times.copy()
    
    # Compute Bottom Level (BL)
    # BL[i] = times[i] + max(BL[succ] for succ in successors[i])
    # If no successors, BL[i] = times[i]
    bl = np.zeros(n, dtype=float)
    
    # Process in reverse topological order
    for node in reversed(topo_order):
        max_succ_bl = 0.0
        for succ in successors[node]:
            if bl[succ] > max_succ_bl:
                max_succ_bl = bl[succ]
        
        bl[node] = times[node] + max_succ_bl
        
    return bl
# EVOLVE-BLOCK-END
