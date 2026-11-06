"""Rewritten program: adaptive priority for single-model assembly line balancing.
The scheduler (fixed by the task) picks, per station, the highest-priority
eligible task that fits the remaining cycle time. We only control priorities.
We construct several analytically-motivated priority vectors, simulate the
greedy fill for each (cheap for 30 tasks), and return the best one.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _topo_order(n, pred):
    """Kahn topological order. pred[i] = list of predecessor indices of i."""
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            indeg[i] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for i in range(n):
            if u in pred[i]:
                indeg[i] -= 1
                if indeg[i] == 0:
                    q.append(i)
    if len(order) != n:
        # fallback
        order = list(range(n))
    return order


def _build_graph(n, pred, succ):
    return pred, succ


def _downstream_load(n, succ, times):
    """For each task, sum of durations of all transitive successors (including self)."""
    load = times.astype(float).copy()
    # process in reverse topological order so successors are processed first
    # build reverse topo
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in succ[u]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != n:
        order = list(range(n))
    for u in reversed(order):
        for s in succ[u]:
            load[u] += load[s]
    return load


def _longest_path_end(n, succ, times):
    """Longest path from each node to a sink (max sum of durations along a path, including self)."""
    lp = times.astype(float).copy()
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in succ[u]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != n:
        order = list(range(n))
    for u in reversed(order):
        best = 0.0
        for s in succ[u]:
            if lp[s] > best:
                best = lp[s]
        lp[u] = times[u] + best
    return lp


def _successor_count(n, succ):
    """Count of transitive successors."""
    cnt = np.zeros(n, dtype=float)
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in succ[u]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != n:
        order = list(range(n))
    for u in reversed(order):
        c = 0.0
        for s in succ[u]:
            c += 1.0 + cnt[s]
        cnt[u] = c
    return cnt


def _simulate(times, pred, cycle_time, priority):
    """Greedy list scheduling: at each station pick highest-priority eligible task
    that fits remaining cycle time. Returns number of stations."""
    n = len(times)
    remaining = times.astype(float).copy()
    done = np.zeros(n, dtype=bool)
    # eligible = all predecessors done
    # We pick by priority among those whose preds are all done and remaining<=cycle_time
    stations = 0
    # precompute pred sets
    pred_sets = [set(pred[i]) for i in range(n)]
    while not done.all():
        stations += 1
        cap = cycle_time
        # find eligible
        best = -1
        best_p = -np.inf
        for i in range(n):
            if done[i]:
                continue
            if remaining[i] > cap:
                continue
            # check preds done
            ok = True
            for p in pred_sets[i]:
                if not done[p]:
                    ok = False
                    break
            if not ok:
                continue
            if priority[i] > best_p:
                best_p = priority[i]
                best = i
        if best == -1:
            # no task fits: pick the smallest remaining eligible to avoid deadlock
            best = -1
            best_sz = np.inf
            for i in range(n):
                if done[i]:
                    continue
                ok = True
                for p in pred_sets[i]:
                    if not done[p]:
                        ok = False
                        break
                if not ok:
                    continue
                if remaining[i] < best_sz:
                    best_sz = remaining[i]
                    best = i
            if best == -1:
                # should not happen
                break
        done[best] = True
        cap -= remaining[best]
        # fill same station with more tasks greedily (highest priority among newly-eligible fitting)
        while True:
            best2 = -1
            best_p2 = -np.inf
            for i in range(n):
                if done[i]:
                    continue
                if remaining[i] > cap:
                    continue
                ok = True
                for p in pred_sets[i]:
                    if not done[p]:
                        ok = False
                        break
                if not ok:
                    continue
                if priority[i] > best_p2:
                    best_p2 = priority[i]
                    best2 = i
            if best2 == -1:
                break
            done[best2] = True
            cap -= remaining[best2]
    return stations


def priority(instance):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s (and 150s kill)

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    pred = instance["predecessors"]
    # pred may be list of lists or array
    pred_lists = [list(p) for p in pred]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)

    cycle_time = instance["cycle_time"]

    # Build graph metrics
    dl = _downstream_load(n, succ, times)          # downstream load (incl self)
    lp = _longest_path_end(n, succ, times)         # longest path to sink
    sc = _successor_count(n, succ)                 # transitive successor count

    tmax = times.max() if n else 1.0
    if tmax == 0:
        tmax = 1.0

    # Candidate priority vectors
    candidates = []

    # 1) duration only (baseline)
    candidates.append(times.copy())
    # 2) downstream load
    candidates.append(dl.copy())
    # 3) longest path
    candidates.append(lp.copy())
    # 4) successor count
    candidates.append(sc.copy())
    # 5) blends
    candidates.append(0.5 * times + 0.5 * dl)
    candidates.append(0.4 * times + 0.6 * dl)
    candidates.append(0.6 * times + 0.4 * dl)
    candidates.append(0.5 * times + 0.3 * dl + 0.2 * sc)
    candidates.append(0.3 * times + 0.5 * dl + 0.2 * lp)
    candidates.append(0.2 * times + 0.4 * dl + 0.2 * lp + 0.2 * sc)
    # normalize blends to comparable scale (not required, but helps tie-breaks)
    # 6) "commitment" — duration as fraction of cycle time (packs tightly)
    candidates.append(times / max(cycle_time, 1e-9))
    candidates.append(times / max(cycle_time, 1e-9) + 0.5 * dl / (dl.max() + 1e-9))

    # Evaluate each candidate with the greedy simulator
    best_prio = candidates[0]
    best_stations = _simulate(times, pred_lists, cycle_time, best_prio)

    for cand in candidates[1:]:
        if time.time() > deadline:
            break
        s = _simulate(times, pred_lists, cycle_time, cand)
        if s < best_stations:
            best_stations = s
            best_prio = cand

    # If time remains, try a few more randomized blends
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(12345)
        base = np.stack([times, dl, lp, sc, times / max(cycle_time, 1e-9)], axis=1)
        for _ in range(200):
            if time.time() > deadline:
                break
            w = rng.dirichlet(np.ones(base.shape[1]))
            cand = base @ w
            s = _simulate(times, pred_lists, cycle_time, cand)
            if s < best_stations:
                best_stations = s
                best_prio = cand

    return np.asarray(best_prio, dtype=float)
# EVOLVE-BLOCK-END
