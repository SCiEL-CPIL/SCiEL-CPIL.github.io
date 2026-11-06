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
      - total float (lower float = more critical)
      - resource scarcity pressure
      - successor count (unblocking potential)
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    preds = instance.get("predecessors", instance.get("predecessor", [[] for _ in range(n)]))
    
    if "demands" in instance:
        demands = instance["demands"]
    else:
        demands = np.zeros((n, 0))
    
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = np.array([])
    
    demands = np.asarray(demands, dtype=float)
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
    
    # Number of successors (direct)
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = len(succ[i])
    
    # Number of predecessors (direct)
    pred_count = np.zeros(n, dtype=float)
    for i in range(n):
        pred_count[i] = len(preds[i])
    
    # Depth: longest path from start to i (inclusive of i's duration)
    depth = np.zeros(n, dtype=float)
    for i in order:
        best = 0.0
        for p in preds[i]:
            if depth[p] > best:
                best = depth[p]
        depth[i] = durations[i] + best
    
    # Forward pass (earliest start)
    es = np.zeros(n, dtype=float)
    for node in order:
        for s in succ[node]:
            candidate = es[node] + durations[node]
            if candidate > es[s]:
                es[s] = candidate
    
    # Project duration
    if n > 0:
        project_end = max(es[i] + durations[i] for i in range(n))
    else:
        project_end = 0.0
    
    # Backward pass (latest start)
    ls = np.zeros(n, dtype=float)
    for node in reversed(order):
        lf = project_end
        for s in succ[node]:
            if ls[s] < lf:
                lf = ls[s]
        ls[node] = lf - durations[node]
    
    # Total float = LS - ES
    tf = ls - es
    # Clip to non-negative for numerical stability
    tf = np.maximum(tf, 0.0)
    
    # Resource scarcity pressure
    if R > 0 and capacities.size > 0:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        pressure = np.sum(demands / cap, axis=1)
    else:
        pressure = np.zeros(n, dtype=float)
    
    # Normalize all components to [0, 1]
    def normalize(arr):
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    tail_norm = normalize(tail)
    # For total float, lower float = higher priority, so invert
    tf_norm = 1.0 - normalize(tf)
    pressure_norm = normalize(pressure)
    succ_norm = normalize(succ_count)
    
    # Duration priority (longer activities get slight boost)
    dur_norm = normalize(durations)
    
    # Resource demand magnitude (total demand across all resources)
    if R > 0:
        total_demand = np.sum(demands, axis=1)
        demand_norm = normalize(total_demand)
    else:
        demand_norm = np.zeros(n, dtype=float)
    
    # Combine: critical path dominates, then float, then resource pressure, then unblocking
    # Weights tuned for RCPSP: tail and float are most important
    prio = 1.0 * tail_norm + 0.5 * tf_norm + 0.3 * pressure_norm + 0.1 * succ_norm + 0.1 * dur_norm + 0.2 * demand_norm
    
    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(durations, dtype=float)
    
    return np.asarray(prio, dtype=float)
