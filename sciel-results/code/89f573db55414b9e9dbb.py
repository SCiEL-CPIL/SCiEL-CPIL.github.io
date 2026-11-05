import time
import random
import numpy as np

DEADLINE_SECONDS = 150.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    n = len(chosen)
    if n == 0:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2 if n else 0
    if max_s < 1:
        return True
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _build_counts(chosen, cap=1000):
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    if not chosen:
        return counts
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(len(c))
    sums = c[ii] + c[jj]
    np.add.at(counts, sums, 1)
    return counts


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, seed=42):
    rng = random.Random(seed)
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)

    while pool:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        pool_list = list(pool)
        if len(pool_list) <= 300:
            batch = pool_list
        else:
            batch = rng.sample(pool_list, 300)
        arr = np.array(batch, dtype=np.int64)
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)
        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]
        if feasible.size == 0:
            for b in batch:
                pool.discard(b)
            continue
        maxes = cand_max[feasible]
        min_max = maxes.min()
        tight_candidates = feasible[maxes == min_max]
        tight_counts = (gathered[tight_candidates] == 2).sum(axis=1)
        best_local = tight_candidates[np.argmin(tight_counts)]
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


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iter=50000):
    """
    Local search: try to add elements, or swap elements to maintain/improve size.
    """
    if not chosen:
        return chosen

    current = set(chosen)
    counts = _build_counts(list(current), cap)
    best_size = len(current)
    best_set = set(current)

    rng = random.Random(123)
    candidates = [x for x in range(1, cap + 1) if x not in current]

    no_improve = 0
    max_no_improve = 1000

    for it in range(max_iter):
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break

        # Strategy 1: Try to add a random candidate if possible
        if candidates:
            try_add = rng.sample(candidates, min(10, len(candidates)))
            added = False
            for x in try_add:
                chosen_arr = np.asarray(list(current), dtype=np.int64)
                new_sums = np.concatenate([chosen_arr + x, np.array([2 * x], dtype=np.int64)])
                if (counts[new_sums] + 1).max() <= 2:
                    current.add(x)
                    candidates.remove(x)
                    counts[new_sums] += 1
                    best_size = len(current)
                    best_set = set(current)
                    no_improve = 0
                    added = True
                    break
            if added:
                continue

        # Strategy 2: Try swaps to create space for future additions
        if current:
            rem = rng.choice(list(current))
            other = [y for y in current if y != rem]
            other_arr = np.asarray(other, dtype=np.int64)

            removed_sums = np.concatenate([other_arr + rem, np.array([2 * rem], dtype=np.int64)])

            tent_counts = counts.copy()
            tent_counts[removed_sums] -= 1

            found_swap = False
            if candidates:
                swap_candidates = rng.sample(candidates, min(50, len(candidates)))
                for x in swap_candidates:
                    if x == rem:
                        continue
                    new_sums = np.concatenate([other_arr + x, np.array([2 * x], dtype=np.int64)])
                    if (tent_counts[new_sums] + 1).max() <= 2:
                        current.remove(rem)
                        current.add(x)
                        if rem in candidates:
                            candidates.remove(rem)
                        if x in candidates:
                            candidates.remove(x)

                        counts = tent_counts
                        counts[new_sums] += 1

                        found_swap = True
                        no_improve = 0
                        break

            if not found_swap:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

    if len(best_set) > len(current):
        return list(best_set)
    return list(current)


def _greedy_direct(pool_order, cap=1000, time_fn=None, stop_fn=None):
    """Direct greedy: iterate in given order, add if valid."""
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    for x in pool_order:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
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


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 150.0

    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 10.0

    best = []
    best_len = -1

    # Phase 1: Multiple scored greedy runs with different seeds
    for seed in range(20):
        if _time_left() < 15.0:
            break
        res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=seed)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    if _time_left() > 20.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    # Phase 3: Try simple structured patterns (quick checks)
    patterns = [
        list(range(1, cap + 1, 2)),  # every 2nd
        [i for i in range(1, cap + 1) if i % 3 != 0],  # skip multiples of 3
        [i for i in range(1, cap + 1) if i % 4 in (1, 2)],  # 2 per 4
    ]
    for pat in patterns:
        if _valid(pat):
            if len(pat) > best_len:
                best_len = len(pat)
                best = pat

    # Phase 4: Try greedy with sorted order and reverse
    if _time_left() > 15.0:
        res_asc = _greedy_direct(list(range(1, cap + 1)), cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(res_asc) > best_len:
            best_len = len(res_asc)
            best = res_asc
    if _time_left() > 15.0:
        res_desc = _greedy_direct(list(range(cap, 0, -1)), cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(res_desc) > best_len:
            best_len = len(res_desc)
            best = res_desc

    # Final validation
    if best and not _valid(best):
        fallback = _greedy_direct(list(range(1, cap + 1)), cap=cap)
        if _valid(fallback):
            best = fallback

    return best
