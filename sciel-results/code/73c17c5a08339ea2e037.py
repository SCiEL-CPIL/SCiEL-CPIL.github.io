"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def priority(instance):
    start_time = _time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological order (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes in topo, there's a cycle; fall back to LPT
    if len(topo) != n:
        return times.copy()
    
    # Compute longest downstream path (rank position) in reverse topological order
    # rank[i] = times[i] + max(rank[j] for j in successors[i], default 0)
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in successors[u]:
            if rank[v] > max_succ:
                max_succ = rank[v]
        rank[u] = times[u] + max_succ
    
    # Deadline check
    if _time.time() - start_time > DEADLINE:
        return rank
    
    return rank
# EVOLVE-BLOCK-END
