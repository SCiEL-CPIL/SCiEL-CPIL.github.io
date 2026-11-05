"""Rewritten program using Critical Path Density + successor count heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each task for a priority-based assembly line balancing heuristic.
    The evaluator fills stations by repeatedly picking the highest-priority
    eligible task that fits the remaining cycle time.
    We want to minimize the number of stations.
    
    Strategy: Critical Path Density + Successor Fan-out
    - Compute the longest path from each task to any successor (downstream length).
    - Critical path density = task_time / (task_time + downstream_length)
      This rewards tasks that are long relative to what follows them.
    - Also reward tasks with many immediate successors (high fan-out), as placing
      them early avoids fragmenting downstream work across many stations.
    - Combine: score = density * (1 + 0.3 * log(1 + num_successors))
    
    This should be computed in O(n^2) or better using topological order.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build adjacency: successors[i] = list of tasks that depend on i
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute downstream_length[i] = max over successors s of (times[s] + downstream_length[s]), 0 if no successors
    # Use topological order (reverse: process sinks first)
    # We can do a simple DFS/memoized approach or iterative topological sort.
    
    # Topological sort using Kahn's algorithm
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1  # p is a predecessor of i, so i has an incoming edge from p
    
    # Actually, let's build the graph properly.
    # predecessors[i] is the list of tasks that must come before i.
    # So edge p -> i for each p in predecessors[i].
    
    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If topo_order doesn't include all nodes, there's a cycle; fall back to simple heuristic
    if len(topo_order) != n:
        # Cycle detected; return times as fallback
        return times.copy()
    
    # Compute downstream_length in reverse topological order
    downstream_length = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        node = topo_order[i]
        if successors[node]:
            max_dl = 0.0
            for s in successors[node]:
                val = times[s] + downstream_length[s]
                if val > max_dl:
                    max_dl = val
            downstream_length[node] = max_dl
        # else stays 0
    
    # Compute density: times[i] / (times[i] + downstream_length[i])
    denom = times + downstream_length
    # Avoid division by zero
    denom_safe = np.where(denom > 0, denom, 1.0)
    density = times / denom_safe
    
    # Count immediate successors
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Combine: score = density * (1 + 0.3 * log(1 + succ_count))
    score = density * (1.0 + 0.3 * np.log1p(succ_count))
    
    return score
# EVOLVE-BLOCK-END
