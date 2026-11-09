"""Resource-constrained project scheduling priority for a serial SGS.

Combines clean CPM ranking (true max/min, inverted total float, tail, duration,
resource pressure) with robust instance normalization and deadline guards.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

_DEADLINE = 150.0   # evaluator wall-clock kill limit
_SAFETY = 8.0       # return this many seconds before the limit
_START_TIME = None

def _soft_min(x, k=1.0):
    """Soft-min via log-sum-exp with temperature 1/k."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    m = np.min(x)
    if not np.isfinite(m):
        return np.zeros_like(x)
    shifted = (m - x) / k
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -500, 500)
    e = np.exp(shifted)
    s = np.sum(e)
    if s <= 0 or not np.isfinite(s):
        return m
    return m - k * np.log(s)


def _pred_list(x):
    """Normalize a predecessor entry to a list of ints."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    return [int(x)]


def _normalize(x):
    """Min-max normalize to [0, 1]; all-zero if degenerate."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _soft_max(x, k=1.0):
    """Soft-max via log-sum-exp with temperature 1/k."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    m = np.max(x)
    if not np.isfinite(m):
        return np.zeros_like(x)
    shifted = (x - m) / k
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -500, 500)
    e = np.exp(shifted)
    s = np.sum(e)
    if s <= 0 or not np.isfinite(s):
        return m
    return m + k * np.log(s)

def _cpm(instance, durations, predecessors, n):
    """Compute head (ES), tail (bottom-level rank), total float, project end."""
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            if 0 <= p < n and p != i:
                in_degree[i] += 1
                successors[p].append(i)

    # Topological order (Kahn)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    tmp = in_degree[:]
    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in successors[node]:
            tmp[s] -= 1
            if tmp[s] == 0:
                queue.append(s)
    if len(topo) < n:  # cycle / disconnected fallback
        seen = set(topo)
        for i in range(n):
            if i not in seen:
                topo.append(i)

    # Forward pass: earliest start using soft-max for smoother gradient
    head = np.zeros(n, dtype=float)
    for node in topo:
        preds = [p for p in predecessors[node] if 0 <= p < n]
        if not preds:
            head[node] = 0.0
        else:
            vals = np.array([head[p] + durations[p] for p in preds])
            # Use soft-max with small temperature for smoothness
            head[node] = float(_soft_max(vals, k=0.5))

    project_end = float(np.max(head + durations)) if n else 0.0

    # Tail / bottom-level rank: longest path from node to a sink (soft-max)
    tail = np.zeros(n, dtype=float)
    for node in reversed(topo):
        succs = [s for s in successors[node] if 0 <= s < n]
        if not succs:
            tail[node] = 0.0
        else:
            vals = np.array([durations[s] + tail[s] for s in succs])
            tail[node] = float(_soft_max(vals, k=0.5))

    # Backward pass: latest start using soft-min for smoother gradient
    ls = np.full(n, project_end, dtype=float)
    for node in reversed(topo):
        succs = [s for s in successors[node] if 0 <= s < n]
        if not succs:
            ls[node] = project_end - durations[node]
        else:
            vals = np.array([ls[s] - durations[node] for s in succs])
            # Also include the project_end constraint
            vals = np.append(vals, project_end - durations[node])
            ls[node] = float(_soft_min(vals, k=0.5))

    total_float = np.maximum(ls - head, 0.0)
    return head, tail, total_float, project_end, successors


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard (fast path)
    if time.time() - _START_TIME > _DEADLINE - _SAFETY:
        n = len(instance.get("durations", []))
        return np.zeros(n)

    # Extract instance data (robust key handling)
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.asarray([], dtype=float)

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
        capacities = np.array([float("inf")])

    if "predecessors" in instance:
        preds_raw = instance["predecessors"]
    elif "predecessor_lists" in instance:
        preds_raw = instance["predecessor_lists"]
    else:
        preds_raw = [[] for _ in range(n)]
    predecessors = [_pred_list(preds_raw[i]) for i in range(n)]

    # Normalize demand/capacity shapes
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
        capacities = capacities.reshape(1)
    m = len(capacities)
    if demands.shape[1] < m:
        demands = np.hstack([demands, np.zeros((n, m - demands.shape[1]))])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
        capacities = capacities[:m]
        m = len(capacities)

    # CPM features (true max/min)
    try:
        head, tail, total_float, project_end, successors = _cpm(
            instance, durations, predecessors, n)
    except Exception:
        head = np.zeros(n, dtype=float)
        tail = durations.copy()
        total_float = np.zeros(n, dtype=float)
        project_end = float(np.max(durations)) if n else 0.0
        successors = [[] for _ in range(n)]

    # Deadline guard (mid path)
    if time.time() - _START_TIME > _DEADLINE - _SAFETY:
        scores = 0.5 * (1.0 - _normalize(total_float)) + 0.5 * _normalize(tail)
        return np.asarray(scores, dtype=float)

    # Resource features
    max_demand_ratio = np.zeros(n)
    weighted_demand = np.zeros(n)
    for r in range(m):
        c = capacities[r]
        if c > 0 and np.isfinite(c):
            ratio = demands[:, r] / c
            max_demand_ratio = np.maximum(max_demand_ratio, ratio)
            weighted_demand += ratio * durations
    demand_intensity = np.sum(demands, axis=1) if m > 0 else np.zeros(n)

    succ_count = np.array([len(s) for s in successors], dtype=float)
    dur = durations

    # Composite priority (higher = schedule first)
    # Adjusted weights: emphasize critical path features more
    scores = (
        0.50 * (1.0 - _normalize(total_float))   # low float -> critical -> first
        + 0.25 * _normalize(tail)                # long tail -> first
        + 0.15 * _normalize(dur)                 # long duration -> first
        + 0.05 * _normalize(demand_intensity)    # heavy load -> first
        + 0.03 * _normalize(max_demand_ratio)    # bottleneck pressure
        + 0.02 * _normalize(head)                # early start -> first
    )

    return np.asarray(scores, dtype=float)
# EVOLVE-BLOCK-END
