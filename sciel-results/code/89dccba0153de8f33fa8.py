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
    
    Heuristic: combine
      - tail (longest path from activity to end, i.e., critical-path length)
      - resource scarcity pressure (how much scarce resources it consumes)
      - successor count (activities that unblock more downstream work)
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

    # Successor count: how many direct successors (unblocking potential)
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = len(succ[i])

    # Resource scarcity pressure: sum over resources of (demand / capacity)
    if R > 0 and capacities.size:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        # Normalize capacities to [0,1] range for consistency
        cap_max = cap.max() if cap.size else 1.0
        if cap_max > 0:
            cap_norm = cap / cap_max
        else:
            cap_norm = cap
        pressure = np.sum(demands / cap_norm, axis=1)
        # Normalize pressure to be on similar scale as tail
        p_max = pressure.max() if pressure.size else 1.0
        if p_max > 0:
            pressure_norm = pressure / p_max
        else:
            pressure_norm = pressure
    else:
        pressure_norm = np.zeros(n, dtype=float)

    # Normalize tail to [0,1]
    t_max = tail.max() if tail.size else 1.0
    if t_max > 0:
        tail_norm = tail / t_max
    else:
        tail_norm = tail

    # Normalize successor count
    sc_max = succ_count.max() if succ_count.size else 1.0
    if sc_max > 0:
        succ_norm = succ_count / sc_max
    else:
        succ_norm = succ_count

    # Combine: critical path dominates, then resource pressure, then unblocking
    # Weights: 1.0 for tail, 0.4 for pressure, 0.1 for successor count
    prio = 1.0 * tail_norm + 0.4 * pressure_norm + 0.1 * succ_norm

    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(durations, dtype=float)

    return np.asarray(prio, dtype=float)
