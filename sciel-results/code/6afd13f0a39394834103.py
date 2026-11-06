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
    
    # Combine heuristics:
    # Primary: longest path (lp) - tasks with longer remaining paths should be scheduled early
    # Secondary: weighted precedence (wp) - tasks with more total successor work
    # Tertiary: number of successors - tasks that unlock more other tasks
    # Quaternary: depth - tasks deeper in the precedence graph
    # Quinary: task time itself
    
    # Normalize each component to [0, 1] range for fair combination
    lp_min, lp_max = min(lp), max(lp)
    lp_range = lp_max - lp_min if lp_max > lp_min else 1.0
    
    wp_min, wp_max = min(wp), max(wp)
    wp_range = wp_max - wp_min if wp_max > wp_min else 1.0
    
    depth_min, depth_max = min(depth), max(depth)
    depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
    
    time_min, time_max = min(times), max(times)
    time_range = time_max - time_min if time_max > time_min else 1.0
    
    priority_scores = np.zeros(n)
    for i in range(n):
        lp_norm = (lp[i] - lp_min) / lp_range
        wp_norm = (wp[i] - wp_min) / wp_range
        succ_norm = num_succ[i] / max(1, max(num_succ))
        depth_norm = (depth[i] - depth_min) / depth_range
        time_norm = (times[i] - time_min) / time_range
        
        # Weighted combination
        priority_scores[i] = (0.4 * lp_norm + 
                            0.25 * wp_norm + 
                            0.15 * succ_norm + 
                            0.1 * depth_norm + 
                            0.1 * time_norm)
    
    # Check time budget
    if time.time() - t_start > deadline:
        return priority_scores
    
    return priority_scores
