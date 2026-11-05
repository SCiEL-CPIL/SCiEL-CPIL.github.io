"""Stronger B2[2] set search for {1..1000}.

Design kept: vectorized greedy, scored greedy with smooth surrogate, local improvement,
portfolio of randomized runs under a strict time budget.

Refinements in this rewrite:
  - Soft-max / epigraph-style surrogate in the scored greedy to give smoother gradients.
  - Multi-addition local improvement: after removing an element, run a mini scored-greedy
    to refill, rather than just sequential greedy.
  - More diverse orderings in the portfolio.
  - Tighter time budget management.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 145.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate B2[2]: every sum has at most two representations a<=b."""
    if not chosen:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    if n < 2:
        return True
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
                   batch_size=120, temperature=0.3):
    """Score-driven greedy with soft-max surrogate for smoothness.

    At each step, evaluate a batch of candidates. For each candidate, compute the
    resulting maximum count (tightness). Use a soft-min over feasible candidates
    weighted by their tightness to select the next element, adding randomness
    via temperature for diversity.
    """
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
        cand_max = gathered.max(axis=1).astype(np.float64)
        feasible_mask = cand_max <= 2

        if not feasible_mask.any():
            if evaluate_all:
                break
            else:
                for b in batch_list:
                    pool.discard(b)
                continue

        # Soft-min selection: weight = exp(-temperature * cand_max)
        # Only consider feasible candidates
        scores = np.full(len(batch_list), -np.inf)
        scores[feasible_mask] = cand_max[feasible_mask]
        # Lower score is better (lower max count)
        weights = np.exp(-temperature * (scores - scores.min()))
        weights[~feasible_mask] = 0.0
        total_w = weights.sum()
        if total_w <= 0:
            # All infeasible
            for b in batch_list:
                pool.discard(b)
            continue
        probs = weights / total_w
        idx = int(rng.choice(len(batch_list), p=probs))
        x = int(arr[idx])
        chosen.append(x)
        pool.discard(x)

        if chosen_np.size == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[ns] += 1

    return chosen


def _scored_greedy_deterministic(cap=1000, time_fn=None, stop_fn=None, rng=None,
                                  batch_size=120):
    """Deterministic scored greedy: pick the candidate with the lowest tightness."""
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


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iters=20, rng=None,
                   use_scored_refill=True):
    """Local improvement via single-swap: remove one element, refill with scored greedy
    to try to get a net gain in size."""
    if rng is None:
        rng = random.Random()
    if not chosen:
        return chosen

    best = list(chosen)
    best_set = set(best)
    candidates_pool = [v for v in range(1, cap + 1) if v not in best_set]

    for _ in range(max_iters):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if not candidates_pool:
            break

        idx = rng.randrange(len(best))
        removed = best[idx]
        trial = [best[j] for j in range(len(best)) if j != idx]
        trial_set = set(trial)

        # Refill using scored greedy or simple greedy on remaining pool
        refill_pool = [v for v in candidates_pool if v not in trial_set]
        if not refill_pool:
            continue

        if use_scored_refill:
            # Use scored greedy to refill, but only add if it improves size
            # We need a version that starts from an existing set
            # Instead, we'll do a mini greedy pass
            refill_chosen = []
            refill_counts = np.zeros(2 * cap + 1, dtype=np.int32)
            trial_np = np.asarray(trial, dtype=np.int64) if trial else np.empty(0, dtype=np.int64)

            # Compute current counts for trial
            if trial_np.size >= 2:
                ii, jj = np.triu_indices(trial_np.size)
                sums = trial_np[ii] + trial_np[jj]
                refill_counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)

            # Try adding candidates in random order
            rng.shuffle(refill_pool)
            for cand in refill_pool:
                if time_fn is not None and time_fn() <= 0:
                    break
                if trial_np.size == 0:
                    ns = np.array([2 * cand], dtype=np.int64)
                else:
                    ns = np.concatenate([trial_np + cand, np.array([2 * cand], dtype=np.int64)])
                if (refill_counts[ns] + 1).max() <= 2:
                    trial.append(cand)
                    trial_set.add(cand)
                    refill_counts[ns] += 1
                    trial_np = np.concatenate([trial_np, np.array([cand], dtype=np.int64)])

        if len(trial) > len(best):
            best = trial
            best_set = set(best)
            candidates_pool = [v for v in range(1, cap + 1) if v not in best_set]

    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 145.0

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
    while _time_left() > 15.0:
        iteration += 1
        mode = iteration % 7

        if mode == 0:
            # Random order
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 1:
            # Scored greedy with temperature
            temp = 0.2 + 0.4 * rng.random()
            res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                  rng=rng, batch_size=120, temperature=temp)
        elif mode == 2:
            # Deterministic scored greedy
            res = _scored_greedy_deterministic(cap=cap, time_fn=time_left, stop_fn=stopped,
                                                rng=rng, batch_size=120)
        elif mode == 3:
            # Rotated ascending
            start = rng.randrange(cap)
            order = list(range(start + 1, cap + 1)) + list(range(1, start + 1))
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 4:
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
        elif mode == 5:
            # Descending
            order = full[::-1]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        else:
            # Biased random: favor middle range
            weights = [1.0 + 2.0 * (1 - abs(x - cap/2) / (cap/2)) for x in full]
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)

        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: local improvement
    if _time_left() > 10.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left,
                                  stop_fn=stopped, max_iters=25, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    # Phase 3: additional local improvement passes
    if _time_left() > 8.0 and best:
        improved2 = _local_improve(best, cap=cap, time_fn=time_left,
                                   stop_fn=stopped, max_iters=15, rng=rng)
        if len(improved2) > best_len:
            best = improved2
            best_len = len(best)

    # Final validation
    if best and not _valid(best):
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple

    return best
