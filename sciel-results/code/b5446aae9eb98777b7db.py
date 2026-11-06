import numpy as np
import time
from collections import deque

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 2.0

def _elapsed():
    return time.time() - _T0

def _topo_order(n, preds):
    """Return topological order and successor list."""
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            indeg[i] += 1
            succ[p].append(i)
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:  # cycle fallback
        order = list(range(n))
    return order, succ

def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Structural change: Tail-driven priority (longest path to end) as the dominant
    signal, refined with resource scarcity pressure and total float. This shifts
    from a float-dominant model to a tail-dominant model, which better captures
    the makespan impact of each activity in resource-constrained settings.
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)

    if demands.ndim == 1:
        if n > 0:
            demands = demands.reshape(n, -1)
        else:
            demands = demands.reshape(0, 0)
    R = demands.shape[1] if demands.size else 0

    order, succ = _topo_order(n, preds)

    # Tail: longest duration path from i to end (inclusive of i's duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if tail[s] > best:
                best = tail[s]
        tail[i] = durations[i] + best

    # Head: longest duration path from start to i (inclusive)
    head = np.zeros(n, dtype=float)
    for i in order:
        best = 0.0
        for p in preds[i]:
            if head[p] > best:
                best = head[p]
        head[i] = durations[i] + best

    # Total float: critical path length - head - tail + duration
    cp_len = max(tail) if n > 0 else 0.0
    total_float = cp_len - head - tail + durations

    # Resource scarcity pressure: for each activity, measure how much it consumes
    # scarce resources. Use demand/capacity ratio summed across resources.
    if R > 0 and capacities.size > 0:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        # Per-activity resource pressure: sum of (demand_i_r / capacity_r)
        resource_pressure = np.sum(demands / cap, axis=1)
        rp_max = resource_pressure.max() if n > 0 else 1.0
        if rp_max > 0:
            resource_norm = resource_pressure / rp_max
        else:
            resource_norm = np.zeros(n, dtype=float)
    else:
        resource_norm = np.zeros(n, dtype=float)

    # Normalize tail to [0, 1]
    t_max = tail.max() if n > 0 else 1.0
    if t_max > 0:
        tail_norm = tail / t_max
    else:
        tail_norm = np.zeros(n, dtype=float)

    # Normalize total float (invert: lower float = higher priority)
    tf_max = total_float.max() if n > 0 else 1.0
    if tf_max > 0:
        float_norm = 1.0 - total_float / tf_max
    else:
        float_norm = np.ones(n, dtype=float)

    # Normalize head to [0, 1] (weaker signal)
    h_max = head.max() if n > 0 else 1.0
    if h_max > 0:
        head_norm = head / h_max
    else:
        head_norm = np.zeros(n, dtype=float)

    # Combine: Tail dominates (strongest makespan predictor),
    # resource pressure second, float third, head as weak tie-breaker.
    # This structural shift from float-dominant to tail-dominant is the key change.
    w_tail = 1.0
    w_resource = 0.4
    w_float = 0.25
    w_head = 0.05

    prio = w_tail * tail_norm + w_resource * resource_norm + w_float * float_norm + w_head * head_norm

    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)

    return np.asarray(prio, dtype=float)
