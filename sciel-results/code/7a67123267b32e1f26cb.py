"""Rewritten priority heuristic: critical-path (bottom-level) + contention tie-break."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def priority(instance):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Predecessor list per activity (list of lists)
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    else:
        preds = [list(p) for p in preds]

    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            succs[i].append(j)

    # Resource data (optional)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    # Compute bottom-level (BL): longest path from activity to any sink,
    # including the activity's own duration. Process in reverse topological order.
    # Use Kahn's algorithm on the reversed graph (sinks first).
    indeg = [0] * n
    for j in range(n):
        for i in preds[j]:
            indeg[j] += 1  # not needed; we compute out-degree instead

    # out-degree = number of successors
    outdeg = [len(succs[i]) for i in range(n)]

    # queue of sinks (outdeg == 0)
    from collections import deque
    queue = deque([i for i in range(n) if outdeg[i] == 0])

    BL = durations.copy().astype(float)  # initialize with own duration

    processed = [False] * n
    while queue:
        if time.time() > DEADLINE:
            break
        u = queue.popleft()
        processed[u] = True
        # for each predecessor p of u, BL[p] = max(BL[p], dur[p] + BL[u])
        for p in preds[u]:
            cand = durations[p] + BL[u]
            if cand > BL[p]:
                BL[p] = cand
            outdeg[p] -= 1
            if outdeg[p] == 0:
                queue.append(p)

    # If some nodes were not processed (cycles), fill BL with durations to be safe.
    for i in range(n):
        if not processed[i]:
            # fallback: sum along unprocessed is hard; just use own duration
            pass

    # Top-level (TL): longest path from any source to activity (including its duration).
    indeg = [len(preds[i]) for i in range(n)]
    queue = deque([i for i in range(n) if indeg[i] == 0])
    TL = durations.copy().astype(float)
    processed2 = [False] * n
    while queue:
        if time.time() > DEADLINE:
            break
        u = queue.popleft()
        processed2[u] = True
        for s in succs[u]:
            cand = TL[u] + durations[s]
            if cand > TL[s]:
                TL[s] = cand
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)

    # Resource contention score: activities demanding a scarce resource get a small boost.
    # contention[i] = sum over resources of demand[i][r] / capacity[r]  (higher = more scarce)
    contention = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        try:
            D = np.asarray(demands, dtype=float)
            C = np.asarray(capacities, dtype=float)
            if D.ndim == 1:
                D = D.reshape(n, 1)
            if C.ndim == 1:
                C = C.reshape(1, -1)
            # avoid divide by zero
            safeC = np.where(C > 0, C, 1.0)
            # per-activity sum of demand/capacity across resources
            contention = np.sum(D / safeC, axis=1)
        except Exception:
            contention = np.zeros(n, dtype=float)

    # Final priority: bottom-level dominates; use TL and contention as tie-breakers.
    # Normalize each component to [0,1] to combine on comparable scales.
    def norm(x):
        x = np.asarray(x, dtype=float)
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    bl_n = norm(BL)
    tl_n = norm(TL)
    ct_n = norm(contention)

    # Weighted combination: BL is the primary critical-path signal.
    priority_scores = 1.0 * bl_n + 0.2 * tl_n + 0.1 * ct_n

    return priority_scores
# EVOLVE-BLOCK-END
