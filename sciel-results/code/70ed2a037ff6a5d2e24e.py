"""Priority function for resource-constrained project scheduling (serial SGS)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """Return a priority score per activity (higher = schedule earlier).

    Combines:
      - Critical path length (longest path from activity to project end)
      - Resource demand intensity (total demand / max capacity)
      - Successor count (activities with many successors are bottlenecks)
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    # Build successor lists
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)

    # Compute critical path length (longest path to end) via reverse topological order
    # in_degree for topological sort
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            in_deg[j] += 1

    # Topological order (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # If topo is incomplete (cycle), fall back to index order
    if len(topo) < n:
        topo = list(range(n))

    # Reverse topological order for backward pass
    rtopo = list(reversed(topo))

    # Longest path from each node to end (including its own duration)
    cp_len = np.zeros(n, dtype=float)
    for u in rtopo:
        cp_len[u] = durations[u]
        for v in successors[u]:
            if cp_len[v] + durations[u] > cp_len[u]:
                cp_len[u] = durations[u] + cp_len[v]

    # Resource demand intensity: sum of demands / max capacity across resources
    # demands shape: (n, R), capacities shape: (R,)
    if demands.ndim == 1:
        # Single resource case
        demand_intensity = demands / max(capacities, 1e-9)
    else:
        # Normalize each resource's demand by its capacity, then sum
        cap_safe = np.where(capacities > 0, capacities, 1e-9)
        demand_intensity = np.sum(demands / cap_safe, axis=1)

    # Successor count (bottleneck indicator)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Normalize each component to [0, 1]
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    cp_norm = norm(cp_len)
    demand_norm = norm(demand_intensity)
    succ_norm = norm(succ_count)

    # Weighted combination: critical path most important, then demand, then successors
    score = 0.55 * cp_norm + 0.30 * demand_norm + 0.15 * succ_norm

    return score
# EVOLVE-BLOCK-END
