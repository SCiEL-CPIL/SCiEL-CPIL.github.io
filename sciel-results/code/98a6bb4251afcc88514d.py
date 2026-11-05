"""Improved B2[2] set search for {1..1000}.

Strategy:
  - Fast vectorized greedy construction with pre-allocated arrays.
  - Score-driven greedy with conflict-aware scoring.
  - Remove-one-add-two local search for net +1 growth.
  - Portfolio of diverse randomized constructions within time budget.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate that every sum has at most two representations a<=b."""
    n = len(chosen)
    if n == 0:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=1000, time_fn=None, stop_fn=None):
    """Fast greedy build with pre-allocated arrays."""
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen_np = np.empty(0, dtype=np.int64)
    
    for x in order:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        if chosen_np.size == 0:
            new_sums = np.array([2 * x], dtype=np.int64)
        else:
            new_sums = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        
        if (counts[new_sums] + 1).max() <= 2:
            chosen.append(x)
            chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
            counts[new_sums] += 1
    
    return chosen


def _scored_greedy(candidates, cap=1000, time_fn=None, stop_fn=None, batch=150):
    """Score-driven greedy: pick candidate minimizing future conflict cost."""
    pool = set(candidates)
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        if len(pool) <= batch:
            batch_list = list(pool)
            evaluate_all = True
        else:
            batch_list = random.sample(list(pool), batch)
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
        
        # Score: minimize number of sums that will reach count 2 (tightness)
        tight = (gathered[feasible] == 2).sum(axis=1)
        # Secondary: minimize total count increase
        total = gathered[feasible].sum(axis=1)
        combined = tight * 100 + total
        best_local = feasible[int(np.argmin(combined))]
        x = int(arr[best_local])
        
        chosen.append(x)
        pool.discard(x)
        
        if chosen_np.size == 0:
            new_sums_x = np.array([2 * x], dtype=np.int64)
        else:
            new_sums_x = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[new_sums_x] += 1
    
    return chosen


def _remove_one_add_two(chosen, cap=1000, time_fn=None, stop_fn=None, max_tries=200):
    """Try removing one element and adding two to get net +1 growth."""
    if not chosen:
        return chosen
    
    best = list(chosen)
    best_set = set(best)
    candidates = [v for v in range(1, cap + 1) if v not in best_set]
    
    for _ in range(max_tries):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        # Pick random element to remove
        idx = random.randrange(len(best))
        removed = best[idx]
        trial = [best[j] for j in range(len(best)) if j != idx]
        
        c = np.asarray(trial, dtype=np.int64)
        n = len(c)
        if n == 0:
            counts_t = np.zeros(2 * cap + 1, dtype=np.int32)
        else:
            ii, jj = np.triu_indices(n)
            sums = c[ii] + c[jj]
            counts_t = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
        
        # Greedily add as many as possible
        added = []
        trial_np = c.copy() if n > 0 else np.empty(0, dtype=np.int64)
        
        for cand in candidates:
            if cand == removed:
                continue
            if len(trial_np) == 0:
                ns = np.array([2 * cand], dtype=np.int64)
            else:
                ns = np.concatenate([trial_np + cand, np.array([2 * cand], dtype=np.int64)])
            if (counts_t[ns] + 1).max() <= 2:
                trial.append(cand)
                trial_np = np.concatenate([trial_np, np.array([cand], dtype=np.int64)])
                counts_t[ns] += 1
                added.append(cand)
        
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
    
    def time_left():
        return _time_left()
    
    def stopped():
        return _time_left() < 5.0
    
    best = []
    best_len = -1
    rng = random.Random(42)
    
    iteration = 0
    while _time_left() > 20.0:
        iteration += 1
        mode = iteration % 4
        
        if mode == 0:
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 1:
            start = rng.randrange(cap)
            order = list(range(start + 1, cap + 1)) + list(range(1, start + 1))
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 2:
            # Score-driven greedy
            cands = full[:]
            rng.shuffle(cands)
            res = _scored_greedy(cands, cap=cap, time_fn=time_left, stop_fn=stopped, batch=120)
        else:
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
        
        if len(res) > best_len:
            best_len = len(res)
            best = res
    
    # Local improvement: remove-one-add-two
    if _time_left() > 15.0 and best:
        improved = _remove_one_add_two(best, cap=cap, time_fn=time_left, stop_fn=stopped, max_tries=100)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Final validation
    if best and not _valid(best):
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple
    
    return best
