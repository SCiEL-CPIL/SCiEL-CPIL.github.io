"""Assembly line balancing: learn instance-specific priority weights via local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 145.0  # safety margin under 150s wall-clock


def _features(instance):
    """Compute a matrix of per-task features (n, F)."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", None)
    preds = preds if preds is not None else [[] for _ in range(n)]
    preds = [list(p) for p in preds]

    # longest path from start (sum of times of predecessors + self)
    longest_from_start = np.zeros(n)
    memo = {}
    def dfs(i):
        if i in memo:
            return memo[i]
        val = times[i]
        for p in preds[i]:
            val = max(val, dfs(p) + times[i])
        memo[i] = val
        return val
    for i in range(n):
        longest_from_start[i] = dfs(i)

    # longest path to end (self + successors), via reverse graph
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    longest_to_end = np.zeros(n)
    memo2 = {}
    def dfs2(i):
        if i in memo2:
            return memo2[i]
        best = 0.0
        for s in succ[i]:
            best = max(best, dfs2(s))
        val = times[i] + best
        memo2[i] = val
        return val
    for i in range(n):
        longest_to_end[i] = dfs2(i)

    # number of direct successors
    n_succ = np.array([len(s) for s in succ], dtype=float)
    # number of direct predecessors
    n_pred = np.array([len(p) for p in preds], dtype=float)

    # depth = number of predecessors in chain (longest count)
    depth = np.zeros(n)
    memo3 = {}
    def dfs3(i):
        if i in memo3:
            return memo3[i]
        best = 0
        for p in preds[i]:
            best = max(best, dfs3(p) + 1)
        memo3[i] = best
        return best
    for i in range(n):
        depth[i] = dfs3(i)

    feats = np.column_stack([
        times,
        longest_from_start,
        longest_to_end,
        n_succ,
        n_pred,
        depth,
    ])
    return feats, preds, succ


def _simulate(feats, weights, preds, succ, times, cycle_time):
    """Greedy packing: at each step pick highest-priority eligible task that fits.
    Returns number of stations. Returns inf on infeasibility."""
    n = len(times)
    ct = cycle_time
    # eligibility: a task is eligible when all its predecessors are placed.
    placed = np.zeros(n, dtype=bool)
    # remaining predecessor count
    rem_pred = np.array([len(p) for p in preds], dtype=int)
    # current station load
    load = 0.0
    stations = 1
    # priority score per task
    score = feats @ weights
    # tie-breaker: higher score first; use a stable argmax over eligible
    for _ in range(n):
        # eligible = not placed and rem_pred == 0
        eligible = (~placed) & (rem_pred == 0)
        if not eligible.any():
            return float("inf")
        # among eligible, pick one that fits current load; if none fits, open new station
        fits = eligible & ((load + times) <= ct)
        if fits.any():
            idxs = np.where(fits)[0]
            # highest score
            best = idxs[np.argmax(score[idxs])]
        else:
            # open a new station, take the highest-priority eligible
            stations += 1
            load = 0.0
            idxs = np.where(eligible)[0]
            best = idxs[np.argmax(score[idxs])]
        placed[best] = True
        load += times[best]
        # decrement rem_pred of successors
        for s in succ[best]:
            rem_pred[s] -= 1
    return stations


def _local_search(feats, preds, succ, times, cycle_time, time_budget):
    """Search over weight vectors to minimize station count."""
    F = feats.shape[1]
    # normalize features to comparable scale
    col_std = feats.std(axis=0)
    col_std[col_std == 0] = 1.0
    feats_n = feats / col_std

    def sim(w):
        return _simulate(feats_n, w, preds, succ, times, cycle_time)

    # initial: each feature alone, plus a combined
    candidates = []
    for f in range(F):
        w = np.zeros(F)
        w[f] = 1.0
        candidates.append(w)
    # combined equal
    candidates.append(np.ones(F))
    # longest-path heavy
    w = np.zeros(F); w[1] = 1.0; w[0] = 0.5; candidates.append(w)

    best_w = candidates[0]
    best_score = sim(best_w)

    # local search: perturb best weights
    start = time.time()
    iters = 0
    while time.time() < time_budget:
        iters += 1
        # random direction
        direction = np.random.randn(F)
        for step in (0.5, 0.25, 0.1, 0.05):
            w_new = best_w + step * direction
            s = sim(w_new)
            if s < best_score:
                best_score = s
                best_w = w_new
                break
        # also try coordinate ascent
        for f in range(F):
            for delta in (0.1, -0.1, 0.3, -0.3):
                w_new = best_w.copy()
                w_new[f] += delta
                s = sim(w_new)
                if s < best_score:
                    best_score = s
                    best_w = w_new
        if iters % 50 == 0:
            # occasional larger random restart from a feature
            w = np.zeros(F)
            w[np.random.randint(F)] = 1.0
            s = sim(w)
            if s < best_score:
                best_score = s
                best_w = w
    return best_w, best_score


def priority(instance):
    """Return one score per task."""
    start = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)

    # fallback if too little time
    if time.time() > _DEADLINE - 5.0:
        return times.copy()

    feats, preds, succ = _features(instance)
    F = feats.shape[1]

    # budget for local search: leave margin for final return
    time_budget = min(_DEADLINE - 3.0, start + 100.0)

    best_w = None
    best_score = float("inf")
    try:
        best_w, best_score = _local_search(feats, preds, succ, times, cycle_time, time_budget)
    except Exception:
        best_w = None

    if best_w is None:
        # fallback: longest task first
        return times.copy()

    # final score: use normalized features for stability
    col_std = feats.std(axis=0)
    col_std[col_std == 0] = 1.0
    feats_n = feats / col_std
    return feats_n @ best_w
# EVOLVE-BLOCK-END
