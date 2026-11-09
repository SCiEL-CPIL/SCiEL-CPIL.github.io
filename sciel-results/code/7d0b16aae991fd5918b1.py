import numpy as np
import time
from collections import deque

def _greedy_stations(times, preds_list, succs, order, prio, ct):
    n = len(times)
    remaining = [0] * n
    for j in range(n):
        for i in preds_list[j]:
            remaining[j] += 1
    eligible = [i for i in order if remaining[i] == 0]
    if not eligible:
        eligible = list(order)
    unassigned = set(range(n))
    station = 0
    while unassigned:
        station += 1
        load = 0.0
        for i in order:
            if i not in unassigned:
                continue
            if remaining[i] != 0:
                continue
            t = times[i]
            if load + t <= ct:
                load += t
                unassigned.discard(i)
                for v in succs[i]:
                    remaining[v] -= 1
                if not eligible:
                    eligible = [i for i in order if remaining[i] == 0]
        if not unassigned:
            break
        # No task fit this station (all remaining times exceed remaining capacity).
        # Assign the best eligible task that fits the full cycle time; if none,
        # force the best eligible task to keep progress.
        best = -1
        bestp = -1e100
        bestt = -1.0
        for i in order:
            if i not in unassigned or remaining[i] != 0:
                continue
            if times[i] > ct:
                continue
            p = prio[i]
            if p > bestp or (p == bestp and times[i] > bestt):
                bestp, best, bestt = p, i, times[i]
        if best < 0:
            for i in order:
                if i not in unassigned or remaining[i] != 0:
                    continue
                p = prio[i]
                if p > bestp or (p == bestp and times[i] > bestt):
                    bestp, best, bestt = p, i, times[i]
        if best < 0:
            break
        unassigned.discard(best)
        for v in succs[best]:
            remaining[v] -= 1
        if not eligible:
            eligible = [i for i in order if remaining[i] == 0]
    return station

def _topo(times, preds_list, succs):
    n = len(times)
    indeg = [0] * n
    for j in range(n):
        indeg[j] = len(preds_list[j])
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        order.extend([i for i in range(n) if i not in set(order)])
    return order

def priority(instance):
    start = time.time()
    deadline = 138.0  # safety margin before the 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds_raw = instance["predecessors"]
    preds_list = [list(p) for p in preds_raw]
    n = len(times)
    if n == 0:
        return np.array([])

    ct = float(instance.get("cycle_time", 40.0))

    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds_list[j]:
            succs[i].append(j)

    order = _topo(times, preds_list, succs)
    ridx = {i: k for k, i in enumerate(order)}
    rorder = order[::-1]

    # Positional weight: t(i) + sum over successors
    pw = times.copy()
    for u in rorder:
        s = times[u]
        for v in succs[u]:
            s += pw[v]
        pw[u] = s

    # Level: longest path from a source to i (in topological order)
    level = np.zeros(n, dtype=float)
    for u in order:
        m = 0.0
        for i in preds_list[u]:
            if level[i] > m:
                m = level[i]
        level[u] = m + 1.0

    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)

    # Structural tie-breakers (negligible weight vs the learned component)
    base = pw + 0.0001 * (n - succ_count) + 0.00001 * level + 0.000001 * times

    # Learn: for each task, how many stations does greedy use when that task is
    # promoted to the top priority? Then reward tasks that reduce station count.
    scores = np.zeros(n, dtype=float)
    for k in range(n):
        if time.time() > deadline - 5.0:
            break
        cand = base.copy()
        cand[k] = cand.max() + 1.0
        scores[k] = float(_greedy_stations(times, preds_list, succs, order, cand, ct))

    if scores.size and np.ptp(scores) > 1e-9:
        best = scores.min()
        learned = (best - scores) * 1000.0  # >=0, larger = better for the objective
    else:
        learned = np.zeros(n, dtype=float)

    final = base + learned

    if time.time() > deadline - 1.0:
        return final
    return final
