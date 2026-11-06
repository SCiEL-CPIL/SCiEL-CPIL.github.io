"""New program for Resource-Constrained Project Scheduling (serial SGS).

Approach: Bottom-Level (BL) Priority combined with Resource Demand Pressure.
Unlike the seed which uses CPM Slack (backward pass), this uses the longest
path to the end (forward depth from node to sink) multiplied by a resource
scarcity factor. This prioritizes activities that lead to long remaining
chains and consume scarce resources.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Predecessor list per activity.
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    # Normalise predecessors to lists of ints.
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        if isinstance(p, (list, tuple, np.ndarray)):
            p = [int(x) for x in p]
        else:
            p = [int(p)]
        # Filter out self and out-of-range.
        p = [x for x in p if 0 <= x < n and x != i]
        pred_lists.append(p)

    # Successor list per activity.
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in pred_lists[i]:
            succ_lists[j].append(i)

    # ---- Topological Sort (Kahn's Algorithm) ----
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in pred_lists[i]:
            in_degree[i] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle potential cycles (should not happen in RCPSP)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))

    # ---- Bottom Level (BL) Calculation ----
    # BL[i] is the length of the longest path from i to any sink.
    BL = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ_bl = 0.0
        if succ_lists[i]:
            max_succ_bl = max(BL[j] for j in succ_lists[i])
        BL[i] = durations[i] + max_succ_bl

    # ---- Resource Pressure Calculation ----
    demands = np.asarray(instance.get("demands", None), dtype=float)
    capacities = np.asarray(instance.get("capacities", None), dtype=float)

    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and demands.size > 0:
        if demands.ndim == 1:
            demands = demands.reshape(n, 1)
        if demands.shape[0] != n:
            demands = np.zeros((n, 1), dtype=float)

        R = demands.shape[1]
        if capacities is None or np.asarray(capacities).size == 0:
            caps = np.ones(R, dtype=float)
        else:
            caps = np.asarray(capacities, dtype=float).reshape(-1)
            if caps.size == 1:
                caps = np.full(R, float(caps[0]), dtype=float)
            elif caps.size < R:
                tmp = np.ones(R, dtype=float)
                tmp[:caps.size] = caps
                caps = tmp
            else:
                caps = caps[:R]
            caps = np.maximum(caps, 1e-9)

        # Per-activity resource usage ratio
        ratio = demands / caps[np.newaxis, :]  # (n, R)

        # Weight by resource scarcity (how much total demand is on this resource)
        resource_load = np.sum(ratio, axis=0)  # (R,)
        load_max = float(np.max(resource_load)) if R > 0 else 1.0
        if load_max <= 1e-12:
            load_weight = np.ones(R, dtype=float)
        else:
            load_weight = resource_load / load_max  # (R,)

        # Activity pressure: weighted sum of ratios
        resource_pressure = np.sum(ratio * load_weight[np.newaxis, :], axis=1)  # (n,)

        # Normalize pressure to [0, 1]
        rp_max = float(np.max(resource_pressure)) if n > 0 else 1.0
        if rp_max <= 1e-12:
            resource_pressure = np.zeros(n, dtype=float)
        else:
            resource_pressure = resource_pressure / rp_max

    # ---- Composite Priority ----
    # Score = BL * (1 + ResourcePressure)
    score = BL * (1.0 + resource_pressure)

    # Tie-breaker: Higher duration first
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-12:
        dur_norm = durations / dur_max
        score += 0.01 * dur_norm

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
