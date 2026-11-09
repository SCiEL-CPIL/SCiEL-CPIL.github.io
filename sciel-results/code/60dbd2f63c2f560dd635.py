"""Priority heuristic for RCPSP serial schedule-generation scheme."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute a priority score per activity for a serial schedule-generation scheme.
    Higher score => started earlier.

    Design (CPM-based):
      - Total float (slack): critical activities (low float) get high priority.
      - Tail length: activities with long downstream chains get high priority.
      - Resource intensity: demand/capacity ratio captures scarcity pressure.
    """
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s wall-clock kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Normalize predecessors
    preds_raw = instance["predecessors"]
    preds = []
    for i in range(n):
        p = preds_raw[i]
        if isinstance(p, np.ndarray):
            preds.append([int(x) for x in p])
        elif isinstance(p, (list, tuple)):
            preds.append([int(x) for x in p])
        elif p is None:
            preds.append([])
        else:
            try:
                preds.append([int(x) for x in p])
            except TypeError:
                preds.append([])

    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)

    # --- Forward pass (earliest start / finish) in topological order ---
    # Compute in-degree for Kahn's topological sort
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                in_deg[j] += 1

    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    indeg_work = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)

    # If there are cycles (shouldn't be), fall back to identity order
    if len(topo) < n:
        topo = list(range(n))

    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for u in topo:
        es = 0.0
        for p in preds[u]:
            if 0 <= p < n:
                if EF[p] > es:
                    es = EF[p]
        ES[u] = es
        EF[u] = es + durations[u]

    project_end = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass (latest start / finish) in reverse topological order ---
    LF = np.full(n, project_end, dtype=float)
    LS = np.zeros(n, dtype=float)
    for u in reversed(topo):
        lf = project_end
        for s in succs[u]:
            if 0 <= s < n:
                cand = LS[s]
                if cand < lf:
                    lf = cand
        LF[u] = lf
        LS[u] = lf - durations[u]

    # Total float
    total_float = LS - ES
    # Guard against tiny negatives from float error
    total_float = np.maximum(total_float, 0.0)

    # --- Tail length: longest path from activity to project end ---
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for s in succs[u]:
            if 0 <= s < n:
                cand = durations[s] + tail[s]
                if cand > best:
                    best = cand
        tail[u] = durations[u] + best

    # --- Resource intensity: demand/capacity ratio ---
    try:
        demands = instance["demands"]
        capacities = instance["capacities"]
    except (KeyError, TypeError):
        demands = None
        capacities = None

    resource_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        # demands shape: (n, R) or (n,) ; capacities shape: (R,) or (R,1)
        try:
            if demands.ndim == 1:
                d = demands.reshape(n, 1)
            else:
                d = demands
            cap = capacities.reshape(-1)
            # ratio per activity = max over resources of demand/capacity
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = d / cap[None, :]
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            resource_score = np.max(ratio, axis=1) if ratio.ndim > 1 else ratio
        except Exception:
            resource_score = np.zeros(n, dtype=float)

    # --- Normalize each component to [0, 1] ---
    def norm(x):
        x = np.asarray(x, dtype=float)
        mx = float(np.max(x)) if x.size else 0.0
        mn = float(np.min(x)) if x.size else 0.0
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / rng

    # For float: we want LOW float => HIGH priority, so use (1 - norm(float))
    float_norm = norm(total_float)
    tail_norm = norm(tail)
    res_norm = norm(resource_score)

    # Weights: emphasize total float and tail (per measured knowledge), plus resource pressure
    w_float = 0.45
    w_tail = 0.22
    w_res = 0.33

    score = w_float * (1.0 - float_norm) + w_tail * tail_norm + w_res * res_norm

    # Deadline guard: if we're close to the limit, just return what we have
    if time.time() - t0 > DEADLINE:
        return np.asarray(score, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
