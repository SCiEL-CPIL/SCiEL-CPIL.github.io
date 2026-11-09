"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Build successor lists and compute reverse position weight (RPW)
    # RPW[i] = sum of times of all successors of i (including i itself)
    # This captures "how much work remains after this task"
    
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            in_degree[i] += 1

    # Topological order (Kahn's algorithm)
    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo = []
    in_deg_copy = in_degree[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    # If topological sort didn't cover all nodes (cycle?), fall back to original order
    if len(topo) < n:
        topo = list(range(n))

    # Compute RPW: for each task, sum of times of itself and all its successors
    # Process in reverse topological order
    rpw = np.zeros(n, dtype=float)
    for u in reversed(topo):
        rpw[u] = times[u]
        for v in succ[u]:
            rpw[u] += rpw[v]

    # Priority: combine RPW (more successors -> higher priority to unblock) 
    # with task time (longer tasks first)
    # Normalize both to [0, 1] range
    rpw_min, rpw_max = rpw.min(), rpw.max()
    t_min, t_max = times.min(), times.max()
    
    if rpw_max > rpw_min:
        rpw_norm = (rpw - rpw_min) / (rpw_max - rpw_min)
    else:
        rpw_norm = np.zeros(n)
    
    if t_max > t_min:
        t_norm = (times - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros(n)

    # Weighted combination: emphasize RPW slightly more since it captures
    # the "critical path" aspect, but also keep task size
    priority_scores = 0.6 * rpw_norm + 0.4 * t_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
