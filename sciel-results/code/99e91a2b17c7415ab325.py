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
    
    # Compute "slack" potential: how much room a task has in the cycle time
    # relative to its longest path. Tasks with less slack are more critical.
    slack = [cycle_time - lp[i] for i in range(n)]
    
    # Compute a "bottleneck score": tasks that have high out-degree AND high LP
    # are critical bottlenecks because they gate many successors
    bottleneck = [num_succ[i] * lp[i] for i in range(n)]
    
    # Compute "criticality": combination of LP and position in the DAG
    # Critical tasks are those on the critical path (high LP) and early in the DAG
    criticality = [lp[i] * (1.0 + depth[i]) for i in range(n)]
    
    # Normalize using min-max scaling with rank-based fallback for robustness
    lp_arr = np.array(lp)
    wp_arr = np.array(wp)
    succ_arr = np.array(num_succ, dtype=float)
    slack_arr = np.array(slack)
    bottleneck_arr = np.array(bottleneck)
    crit_arr = np.array(criticality)
    
    # Use rank normalization for robustness, but blend with min-max to preserve magnitude info
    def rank_norm(x):
        ranks = np.argsort(np.argsort(x))
        return ranks / max(1, len(x) - 1)
    
    def minmax_norm(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn < 1e-12:
            return np.ones_like(x) * 0.5
        return (x - mn) / (mx - mn)
    
    # Blend rank and min-max: 0.7 rank + 0.3 minmax preserves both order and relative magnitude
    lp_score = 0.7 * rank_norm(lp_arr) + 0.3 * minmax_norm(lp_arr)
    wp_score = 0.7 * rank_norm(wp_arr) + 0.3 * minmax_norm(wp_arr)
    succ_score = 0.7 * rank_norm(succ_arr) + 0.3 * minmax_norm(succ_arr)
    slack_score = 0.7 * rank_norm(-slack_arr) + 0.3 * minmax_norm(-slack_arr)  # lower slack = higher priority
    bottleneck_score = 0.7 * rank_norm(bottleneck_arr) + 0.3 * minmax_norm(bottleneck_arr)
    crit_score = 0.7 * rank_norm(crit_arr) + 0.3 * minmax_norm(crit_arr)
    
    # Weighted combination: emphasize LP, criticality, and bottleneck
    # LP: longest path - tasks that take the longest to complete
    # Criticality: LP * depth - tasks that are both long and deep in the DAG
    # Bottleneck: out-degree * LP - tasks that gate many successors
    # WP: total work - tasks with more total dependent work
    # Succ: out-degree - tasks that unlock more successors
    # Slack: cycle_time - LP - tasks with less slack are more constrained
    priority_scores = (0.25 * lp_score + 
                       0.20 * crit_score + 
                       0.15 * bottleneck_score + 
                       0.15 * wp_score + 
                       0.15 * succ_score + 
                       0.10 * slack_score)
    
    # Check time budget
    if time.time() - t_start > deadline:
        return priority_scores
    
    return priority_scores
