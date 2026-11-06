"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score tasks by a combination of their own processing time, the number of
    transitive successors, and the weighted downstream time (sum of times of
    all transitive successors).
    
    Rationale:
    - High processing time: classic LPT heuristic, fills stations efficiently.
    - High successor count: tasks that unlock many downstream tasks should be
      scheduled early so the greedy packer has more eligible options later.
    - High weighted downstream time: tasks that are bottlenecks for a large
      total amount of downstream work should be scheduled early to free up
      capacity for those downstream tasks.
    
    We use a weighted sum with normalized features.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)

    # Build successor graph
    # predecessors: list of lists, predecessors[i] = list of task indices that must come before i
    preds = instance["predecessors"]
    
    # Build adjacency: succ[i] = list of tasks that depend on i (i must come before succ[i])
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    
    # Compute transitive successor counts and weighted downstream times via BFS/DFS from each node
    # For 30 tasks, this is cheap.
    trans_succ_count = np.zeros(n, dtype=float)
    weighted_downstream = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                weighted_downstream[i] += times[node]
                for nxt in succ[node]:
                    if nxt not in visited:
                        stack.append(nxt)
        trans_succ_count[i] = len(visited)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5.0:
        # Fallback to simple time-based priority
        return times.copy()
    
    max_succ = trans_succ_count.max() if n > 0 else 1.0
    max_downstream = weighted_downstream.max() if n > 0 else 1.0
    
    # Normalize features to [0, 1]
    norm_succ = trans_succ_count / max_succ if max_succ > 0 else np.zeros(n)
    norm_downstream = weighted_downstream / max_downstream if max_downstream > 0 else np.zeros(n)
    
    # Weights: keep time dominant, add modest boosts for bottleneck features
    alpha_count = 0.3   # Boost for high-fanout tasks
    alpha_time = 0.4    # Boost for high downstream workload
    
    scores = times + alpha_count * norm_succ + alpha_time * norm_downstream
    
    return scores
# EVOLVE-BLOCK-END
