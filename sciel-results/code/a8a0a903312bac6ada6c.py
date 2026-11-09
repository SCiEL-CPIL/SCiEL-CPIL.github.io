"""Direct approach for resource-constrained project scheduling priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_DEADLINE = 150.0
_START_TIME = None


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    if "demand" in instance:
        demands = np.asarray(instance["demand"], dtype=float)
    elif "demands" in instance:
        demands = np.asarray(instance["demands"], dtype=float)
    else:
        demands = np.zeros((n, 0))

    if "capacity" in instance:
        capacities = np.asarray(instance["capacity"], dtype=float)
    elif "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = np.array([float('inf')])

    if "predecessors" in instance:
        predecessors = instance["predecessors"]
    elif "predecessor_lists" in instance:
        predecessors = instance["predecessor_lists"]
    else:
        predecessors = [[] for _ in range(n)]

    # Normalize shapes
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
        capacities = capacities.reshape(1)
    m = len(capacities)
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
        capacities = capacities[:m]
        m = len(capacities)

    # Build graph
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            if p >= 0 and p < n:
                in_degree[i] += 1
                successors[p].append(i)

    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)

    if len(topo_order) < n:
        in_topo = set(topo_order)
        for i in range(n):
            if i not in in_topo:
                topo_order.append(i)

    # Compute head (earliest start) and tail (latest finish distance)
    head = np.zeros(n)
    for node in topo_order:
        best = 0.0
        for p in predecessors[node]:
            if p >= 0 and p < n:
                path = head[p] + durations[p]
                if path > best:
                    best = path
        head[node] = best

    tail = np.zeros(n)
    for node in reversed(topo_order):
        best = 0.0
        for s in successors[node]:
            path = durations[s] + tail[s]
            if path > best:
                best = path
        tail[node] = best

    # Critical path length
    cp_length = max(head[i] + durations[i] + tail[i] for i in range(n)) if n > 0 else 0.0

    # Total float (slack)
    total_float = cp_length - head - durations - tail
    total_float = np.maximum(total_float, 0.0)

    # Resource intensity features
    max_demand_ratio = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0 and np.isfinite(capacities[r]):
            ratio = demands[:, r] / capacities[r]
            max_demand_ratio = np.maximum(max_demand_ratio, ratio)

    # Weighted demand (bottleneck awareness)
    weighted_demand = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0 and np.isfinite(capacities[r]):
            weighted_demand += demands[:, r] / capacities[r] * durations

    dur = durations
    succ_count = np.array([len(s) for s in successors], dtype=float)
    pred_count = np.array([len(p) for p in predecessors], dtype=float)

    # Normalize features
    def normalize(x):
        x = np.asarray(x, dtype=float)
        x_max = x.max()
        x_min = x.min()
        rng = x_max - x_min
        if rng > 1e-12:
            return (x - x_min) / rng
        return np.zeros_like(x)

    # Priority scoring with stronger emphasis on total float and tail
    norm_float_inv = 1.0 - normalize(total_float)
    norm_weighted_demand = normalize(weighted_demand)
    norm_duration = normalize(dur)
    norm_tail = normalize(tail)
    norm_succ = normalize(succ_count)
    norm_max_ratio = normalize(max_demand_ratio)

    # Composite priority score - stronger float and tail emphasis
    scores = (0.45 * norm_float_inv +
              0.22 * norm_tail +
              0.15 * norm_weighted_demand +
              0.10 * norm_duration +
              0.05 * norm_succ +
              0.03 * norm_max_ratio)

    return scores
# EVOLVE-BLOCK-END
