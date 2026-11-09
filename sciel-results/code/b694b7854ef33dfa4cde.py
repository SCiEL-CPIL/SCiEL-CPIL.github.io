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

    # Resource pressure:
    # For each activity, compute a measure of how much it consumes resources.
    # We use the sum of (demand / capacity) for each resource, which captures
    # the relative intensity of resource usage.
    resource_pressure = np.zeros(n, dtype=float)
    for j in range(n):
        total_pressure = 0.0
        for k in range(len(capacities)):
            if capacities[k] > 0:
                total_pressure += demands[j][k] / capacities[k]
        resource_pressure[j] = total_pressure

    # Normalize components
    max_cp = np.max(longest_path_to_end) if n > 0 else 1.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = longest_path_to_end / max_cp

    max_press = np.max(resource_pressure) if n > 0 else 1.0
    if max_press == 0:
        max_press = 1.0
    norm_press = resource_pressure / max_press

    # Combine using multiplicative boost:
    # The idea is that activities that are both on the critical path AND
    # resource-heavy should be prioritized much more strongly.
    # Score = norm_cp * (1 + norm_press)
    # This gives a boost to activities with high resource pressure,
    # while keeping critical path as the primary driver.
    score = norm_cp * (1.0 + norm_press)

    return score
# EVOLVE-BLOCK-END
