"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using Critical Path Density.
    
    Strategy:
    1. Calculate Longest Downstream Path (LDP) for each task.
    2. Calculate Critical Path Density (CPD) = time[i] / LDP[i].
    3. Combine CPD with immediate successor count to prioritize high-fanout bottlenecks.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                if pred is not None:
                    successors[pred].append(i)
    
    # Compute in-degrees
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                if pred is not None:
                    in_degree[i] += 1
    
    # Topological sort
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Handle cycles gracefully
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
    
    # Compute Longest Downstream Path (LDP)
    ldp = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        succs = successors[node]
        if not succs:
            ldp[node] = times[node]
        else:
            max_succ_ldp = max(ldp[s] for s in succs)
            ldp[node] = times[node] + max_succ_ldp
    
    # Compute Critical Path Density (CPD)
    # CPD = time[i] / LDP[i]
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cpd = np.where(ldp > 0, times / ldp, 0.0)
    
    # Compute successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Combine CPD with successor count
    # Score = CPD * (1 + succ_count)
    # This rewards tasks that are dense bottlenecks and have high fanout
    priority_scores = cpd * (1.0 + succ_count)
    
    return priority_scores
# EVOLVE-BLOCK-END
