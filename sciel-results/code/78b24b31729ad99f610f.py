# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successors
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)

    # Calculate in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j]:
            in_degree[j] = len(predecessors[j])

    # Topological sort (Kahn's algorithm)
    queue = []
    for j in range(n):
        if in_degree[j] == 0:
            queue.append(j)

    top_order = []
    while queue:
        u = queue.pop(0)
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Reverse topological order
    rev_top_order = list(reversed(top_order))

    # Calculate longest path from each node to the end (critical path length)
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            if durations[j] + longest_path_to_end[j] > max_succ:
                max_succ = durations[j] + longest_path_to_end[j]
        longest_path_to_end[i] = durations[i] + max_succ

    # Resource pressure: use max ratio (bottleneck resource) and sum ratio
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        for k in range(len(capacities)):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
        max_ratio[j] = mx
        sum_ratio[j] = s

    # Normalize components
    max_cp = np.max(longest_path_to_end) if n > 0 else 1.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = longest_path_to_end / max_cp

    max_mr = np.max(max_ratio) if n > 0 else 1.0
    if max_mr == 0:
        max_mr = 1.0
    norm_max_ratio = max_ratio / max_mr

    max_sr = np.max(sum_ratio) if n > 0 else 1.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sum_ratio = sum_ratio / max_sr

    # Combine: primary driver is critical path (tail), boosted by resource contention.
    # Use max_ratio as the primary resource signal (bottleneck), with sum_ratio as secondary.
    # score = norm_cp * (1.0 + 0.6 * norm_max_ratio + 0.4 * norm_sum_ratio)
    score = norm_cp * (1.0 + 0.6 * norm_max_ratio + 0.4 * norm_sum_ratio)

    return score
# EVOLVE-BLOCK-END
