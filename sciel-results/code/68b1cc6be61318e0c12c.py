"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    
    The evaluator fills stations with the highest-priority eligible task that fits
    the cycle time. Lower total number of stations is better.
    
    We use a priority score that combines:
    1. Longest path in the successor DAG (positional priority)
    2. Task duration (longer tasks first)
    3. Number of immediate successors (tasks with more dependents get higher priority)
    
    The idea: prioritize tasks that are "bottlenecks" - they have long durations,
    are early in the dependency chain (longest path to end), and have many successors.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph from predecessors
    # predecessors[i] = list of task indices that must precede task i
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Compute longest path from each task to the end (in topological order)
    # This is the "position" priority - tasks that are earlier in the chain
    # should generally be scheduled first to leave room for their successors
    
    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        # Sort by index for determinism
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Longest path from each node to sink (reverse topological order)
    # dist[i] = longest sum of times from i to any terminal node
    dist = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist[v])
        dist[u] = times[u] + max_succ
    
    # Number of immediate successors
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Total number of successors (transitive)
    # Compute transitive closure count via DP in reverse topological order
    total_succ_count = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0
        for v in succ[u]:
            s += 1 + total_succ_count[v]
        total_succ_count[u] = s
    
    # Deadline guard
    if time.time() - start_time > 100:
        # Fallback: just use task times
        return times
    
    # Composite priority score
    # We want higher priority = scheduled earlier (higher score chosen first)
    # 
    # Components:
    # 1. dist: tasks with longer downstream path should go first
    # 2. times: longer tasks should go first (fill stations efficiently)
    # 3. n_succ: tasks with more immediate successors should go first
    # 4. total_succ_count: tasks with more total successors should go first
    # 5. rank: weighted sum of successor priorities (bottleneck detection)
    #
    # Normalize each component to [0, 1] and combine with weights
    
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_dist = normalize(dist)
    norm_times = normalize(times)
    norm_n_succ = normalize(n_succ)
    norm_total_succ = normalize(total_succ_count)
    
    # Compute rank: weighted sum of successor times (bottleneck indicator)
    # Tasks whose successors are long should be scheduled early to avoid
    # leaving them waiting
    rank = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0.0
        for v in succ[u]:
            s += times[v]
        rank[u] = s
    
    norm_rank = normalize(rank)
    
    # Weighting: emphasize downstream path length, task duration, and rank
    # These three are the most critical for ALB
    w_dist = 0.35
    w_times = 0.30
    w_n_succ = 0.08
    w_total_succ = 0.12
    w_rank = 0.15
    
    score = (w_dist * norm_dist + 
             w_times * norm_times + 
             w_n_succ * norm_n_succ + 
             w_total_succ * norm_total_succ +
             w_rank * norm_rank)
    
    # Add a small tie-breaker based on original time to break symmetry
    # Higher time gets slightly higher priority as tie-breaker
    score = score + 1e-6 * times
    
    if time.time() - start_time > 120:
        return times
    
    return score
# EVOLVE-BLOCK-END
