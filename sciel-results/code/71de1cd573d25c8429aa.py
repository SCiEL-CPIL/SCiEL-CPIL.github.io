import numpy as np
import time
from collections import deque

def priority(instance):
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor map: succ[i] = list of tasks that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            succ[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    in_degree = [len(predecessors[i]) for i in range(n)]
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If not all nodes, there's a cycle - fallback
    if len(topo_order) != n:
        return times
    
    # Compute longest path from each node to a terminal (reverse topological order)
    # lp[i] = times[i] + max(lp[s] for s in succ[i]) if succ[i] else times[i]
    lp = [0.0] * n
    for node in reversed(topo_order):
        if succ[node]:
            lp[node] = times[node] + max(lp[s] for s in succ[node])
        else:
            lp[node] = times[node]
    
    # Compute weighted precedence: sum of times of task and all transitive successors
    # wp[i] = times[i] + sum(wp[s] for s in succ[i])
    wp = [0.0] * n
    for node in reversed(topo_order):
        w = times[node]
        for s in succ[node]:
            w += wp[s]
        wp[node] = w
    
    # Compute number of immediate successors
    num_succ = [len(succ[i]) for i in range(n)]
    
    # Compute depth: number of levels from node to deepest terminal
    depth = [0] * n
    for node in reversed(topo_order):
        if succ[node]:
            depth[node] = 1 + max(depth[s] for s in succ[node])
        else:
            depth[node] = 0
    
    # Use Rank-based normalization for robustness against outliers
    # Higher lp/wp/succ should result in higher priority score.
    lp_arr = np.array(lp)
    wp_arr = np.array(wp)
    succ_arr = np.array(num_succ, dtype=float)
    
    # np.argsort(np.argsort(x)) gives the rank (0 is smallest).
    # We want the largest value to have the highest score.
    # So we normalize the rank to [0, 1] where the largest element gets 1.0 (or close to it).
    lp_ranks = np.argsort(np.argsort(lp_arr))
    wp_ranks = np.argsort(np.argsort(wp_arr))
    succ_ranks = np.argsort(np.argsort(succ_arr))
    
    lp_norm = lp_ranks / max(1, n - 1)
    wp_norm = wp_ranks / max(1, n - 1)
    succ_norm = succ_ranks / max(1, n - 1)
    
    # Weighted combination: favor LP more heavily as primary critical-path signal
    # WP provides secondary packing signal; succ count as tiebreaker
    priority_scores = 0.7 * lp_norm + 0.25 * wp_norm + 0.05 * succ_norm
    
    # Check time budget
    if time.time() - t_start > deadline:
        return priority_scores
    
    return priority_scores
