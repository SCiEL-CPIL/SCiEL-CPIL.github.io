"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy list scheduler.
    The scheduler picks the highest-priority eligible task that fits the cycle time.
    We want to minimize the number of stations, so we want a priority that packs
    well. A good heuristic combines:
      1) Longest path to end (critical path length) - do long chains early
      2) Number of immediate successors - do tasks that unlock many others early
      3) Task time itself - longer tasks first (LPT)
    
    We compute a combined score. Higher score = higher priority.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build adjacency: succs[i] = list of tasks that depend on i
    succs = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in preds[j]:
            succs[i].append(j)
            in_degree[j] += 1
    
    # Compute longest path from each node to the end (reverse topological order)
    # longest_end[i] = times[i] + max(longest_end[s] for s in succs[i])
    # Use topological sort (Kahn's algorithm)
    from collections import deque
    
    # Topological order
    in_deg = in_degree.copy()
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    # If not all nodes processed, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use times
        return times.copy()
    
    # Compute longest path to end in reverse topological order
    longest_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if succs[u]:
            longest_end[u] = times[u] + np.max([longest_end[v] for v in succs[u]])
        else:
            longest_end[u] = times[u]
    
    # Count immediate successors
    num_succs = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Also compute longest path from start to each node (for additional context)
    longest_start = np.zeros(n, dtype=float)
    for u in topo_order:
        if preds[u]:
            longest_start[u] = times[u] + np.max([longest_start[p] for p in preds[u]])
        else:
            longest_start[u] = times[u]
    
    # Combined priority:
    # - Primary: longest_end (critical path remaining)
    # - Secondary: num_succs (unlocking more tasks)
    # - Tertiary: times (longer tasks first)
    #
    # Normalize each component to [0, 1] range for fair combination
    max_le = np.max(longest_end) if n > 0 else 1.0
    max_ns = np.max(num_succs) if n > 0 else 1.0
    max_t = np.max(times) if n > 0 else 1.0
    
    if max_le == 0: max_le = 1.0
    if max_ns == 0: max_ns = 1.0
    if max_t == 0: max_t = 1.0
    
    norm_le = longest_end / max_le
    norm_ns = num_succs / max_ns
    norm_t = times / max_t
    
    # Weighted combination: emphasize longest path, then successors, then time
    # Weights: 0.5 for longest_end, 0.3 for num_succs, 0.2 for times
    score = 0.5 * norm_le + 0.3 * norm_ns + 0.2 * norm_t
    
    return score
# EVOLVE-BLOCK-END
