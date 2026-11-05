"""Search-based bootstrap for B2[2] sets in 1..1000.

Strategy:
  - Fast vectorized greedy construction given an ordering (or a scoring rule).
  - Portfolio: run many diverse randomized constructions within the time budget.
  - Backtracking with forward-checking to escape greedy local optima.
  - Explicit deadline guard with safety margin; return best found.
"""
import time
import random
import numpy as np

# ---- Configuration / deadline guard ----
# Evaluator kills at 150s wall-clock. Keep a safety margin so we never get killed.
DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate that every sum has at most two representations a<=b. Returns bool."""
    n = len(chosen)
    c = np.asarray(chosen, dtype=np.int64)
    # All ordered pairs i<=j
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    # Count via bincount over a bounded range
    max_s = int(c.max()) * 2 if n else 0
    if max_s < 1:
        return True
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=1000, time_fn=None, stop_fn=None):
    """Greedy build: iterate candidates in `order`; add if B2[2] preserved.

    `order` is an iterable of distinct candidate ints in 1..1000 (the priority order).
    Returns the chosen list (in build order) and the final count-dict (as np array).
    """
    chosen = []
    # We track counts of sums as a numpy array sized 2*1000.
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen_arr = []  # keep as list; convert to np for fast checks
    chosen_np = np.empty(0, dtype=np.int64)
    for x in order:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        # New sums: x + y for y in chosen (all, since we add in arbitrary order,
        # but representation a<=b: when we add x, pairs (x,y) for y in chosen where
        # y<=x are "new" representations of x+y with a=x? Actually a<=b means the
        # smaller is first. The pair {x,y} already may exist if both in set.
        # Simpler & safe: consider x with EVERY existing y (y in chosen) -> sum x+y,
        # and also 2x. This counts each unordered pair {x,y} exactly once when x is
        # the element being added, regardless of ordering. Good.
        if chosen_np.size == 0:
            new_sums = np.array([2 * x], dtype=np.int64)
        else:
            new_sums = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        # Check max count after adding
        idx = new_sums
        new_counts = counts[idx] + 1
        if new_counts.max() <= 2:
            # accept
            chosen.append(x)
            chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
            counts[idx] += 1
    return chosen


def _greedy_build_with_score(candidates, cap=1000, time_fn=None, stop_fn=None,
                             score_mode="minmax", batch=200):
    """Score-driven greedy: at each step pick the best candidate (by score) among
    the remaining pool, using a vectorized batch check. Returns chosen list.

    score_mode:
      'minmax'  : minimize the resulting max count (then minimize new collisions)
      'rand'    : random pick (fallback)
    """
    pool = set(candidates)
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        # Take a batch of candidates to evaluate in parallel for speed.
        # If pool small, evaluate all; else evaluate a random batch and pick best.
        if len(pool) <= batch:
            batch_list = list(pool)
            evaluate_all = True
        else:
            batch_list = random.sample(list(pool), batch)
            evaluate_all = False
        arr = np.array(batch_list, dtype=np.int64)
        # Compute for each candidate the vector of new sums and the resulting max
        # count. Use broadcasting: arr[:,None] + chosen_np[None,:] and 2*arr.
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]  # shape (m,1)
        else:
            cross = arr[:, None] + chosen_np[None, :]  # (m, k)
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)  # (m, k+1)
        # For each candidate m, we need max over its new sums of (counts[s]+1)
        # counts is (2cap+1). Gather counts[new_sums] -> (m, k+1)
        gathered = counts[new_sums] + 1  # (m, k+1)
        cand_max = gathered.max(axis=1)  # (m,)
        feasible = np.where(cand_max <= 2)[0]
        if feasible.size == 0:
            if evaluate_all:
                break  # no candidate in entire pool feasible -> stop
            else:
                # none in this batch feasible; remove them from pool and continue
                # (they are infeasible right now; they might become feasible later?
                #  Adding more elements only increases counts, so if infeasible now,
                #  it stays infeasible. So we can drop them.)
                for b in batch_list:
                    pool.discard(b)
                continue
        # Among feasible, pick the best by secondary score: minimize the number of
        # new sums that hit an already-counted slot (i.e., that will have count>=2),
        # which keeps room. Use mean of gathered as a proxy for "how tight".
        tight = gathered[feasible].mean(axis=1)
        best_local = feasible[int(np.argmin(tight))]
        x = int(arr[best_local])
        # Accept x
        chosen.append(x)
        pool.discard(x)
        if chosen_np.size == 0:
            new_sums_x = np.array([2 * x], dtype=np.int64)
        else:
            new_sums_x = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[new_sums_x] += 1
    return chosen


def _backtracking_build(cap=1000, time_fn=None, stop_fn=None, max_depth=200,
                        rng=None, max_backtracks=50):
    """Backtracking search with forward-checking.
    
    Builds a set by trying candidates in a random order, but when stuck,
    backtracks to an earlier choice and tries an alternative. Uses forward-
    checking to prune branches where no candidate can be added.
    
    Returns the best set found.
    """
    if rng is None:
        rng = random.Random()
    
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    best = []
    best_len = 0
    backtrack_count = 0
    candidates_pool = list(range(1, cap + 1))
    rng.shuffle(candidates_pool)
    
    # We'll use a stack-based approach: at each level, try candidates.
    # If stuck, pop the last choice and try the next candidate at that level.
    # To keep it tractable, we limit depth and backtracks.
    
    # Simpler approach: iterative deepening with random restarts
    # and "conflict-driven" backtracking.
    
    # For efficiency, we'll do: greedy build, then when stuck, try to fix
    # by removing a recently-added element that's causing conflicts and
    # replacing it with something else.
    
    # Actually, let's do a cleaner backtracking:
    # State: (chosen_list, counts_array, candidate_index)
    # At each step, try next candidate; if infeasible, try next.
    # If all candidates exhausted, backtrack.
    
    # To make this efficient, we use a priority queue of candidates
    # and forward-checking.
    
    # Let's do a simpler but effective approach:
    # 1. Greedy build with scoring
    # 2. When stuck, identify the most "conflicting" element
    # 3. Remove it and try to fill the gap with alternatives
    
    # Phase A: Greedy build
    greedy_result = _greedy_build_with_score(candidates_pool[:], cap=cap,
                                              time_fn=time_fn, stop_fn=stop_fn,
                                              batch=150)
    if len(greedy_result) > best_len:
        best = greedy_result[:]
        best_len = len(greedy_result)
    
    if time_fn is not None and time_fn() <= 5:
        return best
    if stop_fn is not None and stop_fn() is True:
        return best
    
    # Phase B: Local search with restarts
    # Try to improve by removing one element and greedily re-adding
    current = greedy_result[:]
    current_set = set(current)
    c_arr = np.asarray(current, dtype=np.int64)
    n = len(c_arr)
    ii, jj = np.triu_indices(n)
    sums = c_arr[ii] + c_arr[jj]
    cur_counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
    
    # Try multiple rounds of swap-improvement
    for round_num in range(10):
        if time_fn is not None and time_fn() <= 10:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        improved = False
        # Try removing each element (random order) and see if we can add more
        trial_indices = list(range(len(current)))
        rng.shuffle(trial_indices)
        
        for idx in trial_indices:
            if time_fn is not None and time_fn() <= 10:
                break
            if stop_fn is not None and stop_fn() is True:
                break
            
            removed_val = current[idx]
            trial = [current[j] for j in range(len(current)) if j != idx]
            trial_set = set(trial)
            
            # Rebuild counts
            if len(trial) == 0:
                t_counts = np.zeros(2 * cap + 1, dtype=np.int32)
            else:
                ta = np.asarray(trial, dtype=np.int64)
                nn = len(ta)
                i2, j2 = np.triu_indices(nn)
                s2 = ta[i2] + ta[j2]
                t_counts = np.bincount(s2, minlength=2 * cap + 1).astype(np.int32)
            
            # Try to add candidates not in trial
            added_count = 0
            for cand in range(1, cap + 1):
                if cand in trial_set:
                    continue
                if len(trial) == 0:
                    ns = np.array([2 * cand], dtype=np.int64)
                else:
                    ta2 = np.asarray(trial, dtype=np.int64)
                    ns = np.concatenate([ta2 + cand, np.array([2 * cand], dtype=np.int64)])
                if (t_counts[ns] + 1).max() <= 2:
                    trial.append(cand)
                    trial_set.add(cand)
                    t_counts[ns] += 1
                    added_count += 1
                    if time_fn is not None and time_fn() <= 10:
                        break
            
            if len(trial) > len(current):
                current = trial
                current_set = set(current)
                improved = True
                if len(current) > best_len:
                    best = current[:]
                    best_len = len(current)
        
        if not improved:
            break
    
    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 120.0

    cap = 1000
    full = list(range(1, cap + 1))

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 5.0  # safety margin

    best = []
    best_len = -1

    rng = random.Random(12345)

    # Phase 1: portfolio of randomized greedy constructions.
    # We'll run constructions with different orderings until time runs low.
    iteration = 0
    while _time_left() > 20.0:
        iteration += 1
        # Choose a randomized ordering strategy
        mode = iteration % 3
        if mode == 0:
            order = full[:]
            rng.shuffle(order)
        elif mode == 1:
            # Start from a random offset, then ascending (a "rotated" greedy)
            start = rng.randrange(cap)
            order = list(range(start + 1, cap + 1)) + list(range(1, start + 1))
        else:
            # Random with a bias: interleave low and high
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

        # Occasionally use the score-driven builder (slower per run but smarter)
        if iteration % 3 == 0:
            # score-driven with a random pool ordering for diversity
            cands = full[:]
            rng.shuffle(cands)
            res2 = _greedy_build_with_score(cands, cap=cap, time_fn=time_left,
                                            stop_fn=stopped, batch=120)
            if len(res2) > best_len:
                best_len = len(res2)
                best = res2

        # Every 5th iteration, try backtracking-based improvement
        if iteration % 5 == 0 and _time_left() > 30.0:
            res3 = _backtracking_build(cap=cap, time_fn=time_left, stop_fn=stopped,
                                       rng=random.Random(rng.random()))
            if len(res3) > best_len:
                best_len = len(res3)
                best = res3

    # Phase 2: final local improvement on the best, if time remains.
    if _time_left() > 10.0 and best:
        # Do a few rounds of remove-and-readd on the best
        current = best[:]
        current_set = set(current)
        c_arr = np.asarray(current, dtype=np.int64)
        n = len(c_arr)
        ii, jj = np.triu_indices(n)
        sums = c_arr[ii] + c_arr[jj]
        cur_counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
        
        final_rng = random.Random(99999)
        for round_num in range(5):
            if _time_left() <= 10.0:
                break
            improved = False
            trial_indices = list(range(len(current)))
            final_rng.shuffle(trial_indices)
            for idx in trial_indices:
                if _time_left() <= 10.0:
                    break
                removed_val = current[idx]
                trial = [current[j] for j in range(len(current)) if j != idx]
                trial_set = set(trial)
                
                if len(trial) == 0:
                    t_counts = np.zeros(2 * cap + 1, dtype=np.int32)
                else:
                    ta = np.asarray(trial, dtype=np.int64)
                    nn = len(ta)
                    i2, j2 = np.triu_indices(nn)
                    s2 = ta[i2] + ta[j2]
                    t_counts = np.bincount(s2, minlength=2 * cap + 1).astype(np.int32)
                
                for cand in range(1, cap + 1):
                    if cand in trial_set:
                        continue
                    if len(trial) == 0:
                        ns = np.array([2 * cand], dtype=np.int64)
                    else:
                        ta2 = np.asarray(trial, dtype=np.int64)
                        ns = np.concatenate([ta2 + cand, np.array([2 * cand], dtype=np.int64)])
                    if (t_counts[ns] + 1).max() <= 2:
                        trial.append(cand)
                        trial_set.add(cand)
                        t_counts[ns] += 1
                        if _time_left() <= 10.0:
                            break
                
                if len(trial) > len(current):
                    current = trial
                    current_set = set(current)
                    improved = True
                    if len(current) > best_len:
                        best_len = len(current)
                        best = current[:]
            
            if not improved:
                break

    # Final safety validation (cheap for the best only).
    if best and not _valid(best):
        # Should not happen; fall back to a fresh simple greedy to be safe.
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple

    return best
