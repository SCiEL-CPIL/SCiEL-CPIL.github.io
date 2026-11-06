import numpy as np
import time
from collections import deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 5.0
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)
    
    # Topological order using Kahn's algorithm
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_deg[i] += 1
    
    q = deque()
    for i in range(n):
        if in_deg[i] == 0:
            q.append(i)
    
    topo = []
    indeg_work = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)
    
    # Fallback if topological sort failed
    if len(topo) < n:
        return times.copy()
    
    # Compute in reverse topological order
    # 1. Critical path: longest path from node to any sink (including self)
    cp = np.zeros(n, dtype=float)
    # 2. Downstream work: sum of times of all transitive successors
    downstream_work = np.zeros(n, dtype=float)
    # 3. Depth: number of tasks in longest chain after node
    depth = np.zeros(n, dtype=float)
    # 4. Upstream path: longest path from any source to node (including self)
    up = np.zeros(n, dtype=float)
    
    # Reverse topological pass for cp, downstream_work, depth
    for i in reversed(range(len(topo))):
        u = topo[i]
        if time.time() > deadline:
            break
        max_succ_cp = 0.0
        dw = 0.0
        max_depth = 0.0
        for v in successors[u]:
            if cp[v] > max_succ_cp:
                max_succ_cp = cp[v]
            dw += times[v] + downstream_work[v]
            if 1 + depth[v] > max_depth:
                max_depth = 1 + depth[v]
        cp[u] = times[u] + max_succ_cp
        downstream_work[u] = dw
        depth[u] = max_depth
    
    # Forward topological pass for upstream path
    for i in range(len(topo)):
        u = topo[i]
        if time.time() > deadline:
            break
        max_pred_up = 0.0
        for pred in predecessors[u]:
            if up[pred] > max_pred_up:
                max_pred_up = up[pred]
        up[u] = times[u] + max_pred_up
    
    # Normalize each metric to [0,1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(cp)
    dw_norm = normalize(downstream_work)
    depth_norm = normalize(depth)
    up_norm = normalize(up)
    
    # Weighted combination:
    # Critical path is most important (identifies tasks on longest chains)
    # Downstream work captures total remaining workload (branching complexity)
    # Depth captures chain length (tie-breaker for similar cp)
    # Upstream path identifies tasks that are bottlenecks from the start
    # 
    # Weights: cp (0.4), dw (0.3), up (0.2), depth (0.1)
    # This emphasizes critical path and downstream work, with upstream and depth as secondary
    score = 0.4 * cp_norm + 0.3 * dw_norm + 0.2 * up_norm + 0.1 * depth_norm
    
    return score
