"""Stronger B2[2] set search for {1..1000}.

Strategy:
  - Fast vectorized greedy construction with multiple diverse orderings.
  - A score-driven greedy that picks the best feasible candidate each step.
  - Local improvement via single-swap (remove one, add one or more).
  - Portfolio of randomized runs within a strict time budget.
  - Deadline guard with safety margin.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate B2[2]: every sum has at most two representations a<=b."""
    if not chosen:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=1000, time_fn=None, stop_fn=None):
    """Greedy: iterate candidates in order; add if B2[2] preserved."""
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen_np = np.empty(0, dtype=np.int64)
    for x in order:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if chosen_np.size == 0:
            new_sums = np.array([2 * x], dtype=np.int64)
        else:
            new_sums = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        new_counts = counts[new_sums] + 1
        if new_counts.max() <= 2:
            chosen.append(x)
            chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
            counts[new_sums] += 1
    return chosen


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                   batch_size=150, score_mode="min_tightness"):
    """Score-driven greedy: at each step, evaluate a batch of random candidates
    and pick the one that minimizes the resulting maximum count (tightness)."""
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)

    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if len(pool) <= batch_size:
            batch_list = list(pool)
            evaluate_all = True
        else:
            batch_list = rng.sample(list(pool), batch_size)
            evaluate_all = False

        arr = np.array(batch_list, dtype=np.int64)
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)

        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]

        if feasible.size == 0:
            if evaluate_all:
                break
            else:
                for b in batch_list:
                    pool.discard(b)
                continue

        # Pick candidate minimizing the max count it would create
        best_local = feasible[int(np.argmin(cand_max[feasible]))]
        x = int(arr[best_local])
        chosen.append(x)
        pool.discard(x)

        if chosen_np.size == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[ns] += 1

    return chosen


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iters=20, rng=None):
    """Try single-swap improvements: remove one element, try to add better ones."""
    if rng is None:
        rng = random.Random()
    if not chosen:
        return chosen
    best = list(chosen)
    best_set = set(best)
    candidates = [v for v in range(1, cap + 1) if v not in best_set]

    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)

    for _ in range(max_iters):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        idx = rng.randrange(len(best))
        removed = best[idx]
        trial = [best[j] for j in range(len(best)) if j != idx]
        trial_set = set(trial)

        # Recompute counts for trial
        tc = np.asarray(trial, dtype=np.int64)
        tn = len(tc)
        if tn > 0:
            ii2, jj2 = np.triu_indices(tn)
            ts = tc[ii2] + tc[jj2]
            tcounts = np.bincount(ts, minlength=2 * cap + 1).astype(np.int32)
        else:
            tcounts = np.zeros(2 * cap + 1, dtype=np.int32)

        # Try to add candidates
        added = 0
        ta = np.asarray(trial, dtype=np.int64) if trial else np.empty(0, dtype=np.int64)
        for cand in candidates:
            if cand in trial_set:
                continue
            if ta.size == 0:
                ns = np.array([2 * cand], dtype=np.int64)
            else:
                ns = np.concatenate([ta + cand, np.array([2 * cand], dtype=np.int64)])
            if (tcounts[ns] + 1).max() <= 2:
                trial.append(cand)
                trial_set.add(cand)
                tcounts[ns] += 1
                ta = np.concatenate([ta, np.array([cand], dtype=np.int64)])
                added += 1
                if time_fn is not None and time_fn() <= 0:
                    break

        if len(trial) > len(best):
            best = trial
            best_set = set(best)
            candidates = [v for v in range(1, cap + 1) if v not in best_set]

    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0

    cap = 1000
    full = list(range(1, cap + 1))
    best = []
    best_len = -1
    rng = random.Random(42)

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 5.0

    # Phase 1: portfolio of diverse greedy constructions
    iteration = 0
    while _time_left() > 12.0:
        iteration += 1
        mode = iteration % 5

        if mode == 0:
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 1:
            # Scored greedy
            res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                 rng=rng, batch_size=150)
        elif mode == 2:
            # Rotated ascending
            start = rng.randrange(cap)
            order = list(range(start + 1, cap + 1)) + list(range(1, start + 1))
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 3:
            # Interleaved low/high
            lows = list(range(1, cap // 2 + 1))
            highs = list(range(cap // 2 + 1, cap + 1))
            rng.shuffle(lows)
            rng.shuffle(highs)
            order = []
            li = hi = 0
            while li < len(lows) or hi < len(highs):
                if li < len(lows):
                    order.append(lows[li]); li += 1
                if hi < len(highs):
                    order.append(highs[hi]); hi += 1
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        else:
            # Descending
            order = full[::-1]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)

        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: local improvement
    if _time_left() > 8.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left,
                                  stop_fn=stopped, max_iters=15, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    # Final validation
    if best and not _valid(best):
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple

    return best
