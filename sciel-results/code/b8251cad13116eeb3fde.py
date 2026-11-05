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
    pool = list(range(1, cap + 1))
    pool_set = set(pool)
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)

    while pool:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        # Batch size: larger when few candidates remain, smaller early for diversity
        if len(pool) <= 150:
            batch = pool[:]
        elif len(pool) <= 500:
            batch = rng.sample(pool, 200)
        else:
            batch = rng.sample(pool, 100)
        
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
                if b in pool_set:
                    pool_set.discard(b)
                    pool.remove(b)
            continue
        
        # Score: prefer candidates that minimize the maximum sum count,
        # then minimize the number of sums that would hit exactly 2
        maxes = cand_max[feasible]
        min_max = maxes.min()
        tight_candidates = feasible[maxes == min_max]
        tight_counts = (gathered[tight_candidates] == 2).sum(axis=1)
        
        # Additional tie-break: prefer smaller values (more room for larger later)
        if tight_counts.size > 1:
            min_tc = tight_counts.min()
            final_cands = tight_candidates[tight_counts == min_tc]
            vals = arr[final_cands]
            best_idx = final_cands[np.argmin(vals)]
        else:
            best_idx = tight_candidates[0]
        
        x = int(arr[best_idx])
        chosen.append(x)
        pool_set.discard(x)
        pool.remove(x)
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
    current_list = sorted(current)
    current_np = np.asarray(current_list, dtype=np.int64)
    counts = _build_counts(current_list, cap)
    best_size = len(current)
    best_set = set(current)

    rng = random.Random(123)
    candidates = [x for x in range(1, cap + 1) if x not in current]
    cand_set = set(candidates)

    no_improve = 0
    max_no_improve = 500

    for it in range(max_iter):
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break

        # Strategy 1: Try to add a random candidate if possible
        if candidates:
            try_add = rng.sample(candidates, min(20, len(candidates)))
            added = False
            for x in try_add:
                new_sums = np.concatenate([current_np + x, np.array([2 * x], dtype=np.int64)])
                if (counts[new_sums] + 1).max() <= 2:
                    current.add(x)
                    current_list.append(x)
                    current_np = np.concatenate([current_np, np.array([x], dtype=np.int64)])
                    candidates.remove(x)
                    cand_set.discard(x)
                    counts[new_sums] += 1
                    best_size = len(current)
                    best_set = set(current)
                    no_improve = 0
                    added = True
                    break
            if added:
                continue

        # Strategy 2: Try swaps to create space for future additions
        if current and candidates:
            rem = rng.choice(list(current))
            idx = current_list.index(rem)
            other_np = np.delete(current_np, idx)
            other_list = current_list[:idx] + current_list[idx+1:]

            removed_sums = np.concatenate([other_np + rem, np.array([2 * rem], dtype=np.int64)])

            tent_counts = counts.copy()
            tent_counts[removed_sums] -= 1

            found_swap = False
            swap_candidates = rng.sample(candidates, min(80, len(candidates)))
            for x in swap_candidates:
                new_sums = np.concatenate([other_np + x, np.array([2 * x], dtype=np.int64)])
                if (tent_counts[new_sums] + 1).max() <= 2:
                    # Perform swap
                    current.remove(rem)
                    current.add(x)
                    current_list[idx] = x
                    current_list.sort()
                    current_np = np.asarray(current_list, dtype=np.int64)
                    
                    if rem in cand_set:
                        cand_set.discard(rem)
                    if x in cand_set:
                        cand_set.discard(x)
                    # Update candidates list
                    if rem in candidates:
                        candidates.remove(rem)
                    if x in candidates:
                        candidates.remove(x)
                    else:
                        candidates.append(rem)
                        cand_set.add(rem)

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


def _polish(chosen, cap=1000, time_fn=None, stop_fn=None):
    """
    Final polish: try to add elements in sorted order, then try swaps.
    This is a tighter, more systematic local search.
    """
    if not chosen:
        return chosen
    
    current = set(chosen)
    current_list = sorted(current)
    current_np = np.asarray(current_list, dtype=np.int64)
    counts = _build_counts(current_list, cap)
    
    rng = random.Random(999)
    candidates = [x for x in range(1, cap + 1) if x not in current]
    cand_set = set(candidates)
    
    # Phase A: Try adding in ascending order
    improved = True
    while improved:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        improved = False
        for x in sorted(candidates):
            new_sums = np.concatenate([current_np + x, np.array([2 * x], dtype=np.int64)])
            if (counts[new_sums] + 1).max() <= 2:
                current.add(x)
                current_list.append(x)
                current_np = np.concatenate([current_np, np.array([x], dtype=np.int64)])
                candidates.remove(x)
                cand_set.discard(x)
                counts[new_sums] += 1
                improved = True
                break
    
    # Phase B: Random swaps
    no_improve = 0
    max_no_improve = 200
    for _ in range(5000):
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        if not candidates or not current:
            break
        
        rem = rng.choice(list(current))
        idx = current_list.index(rem)
        other_np = np.delete(current_np, idx)
        
        removed_sums = np.concatenate([other_np + rem, np.array([2 * rem], dtype=np.int64)])
        tent_counts = counts.copy()
        tent_counts[removed_sums] -= 1
        
        found = False
        swap_cands = rng.sample(candidates, min(100, len(candidates)))
        for x in swap_cands:
            new_sums = np.concatenate([other_np + x, np.array([2 * x], dtype=np.int64)])
            if (tent_counts[new_sums] + 1).max() <= 2:
                current.remove(rem)
                current.add(x)
                current_list[idx] = x
                current_list.sort()
                current_np = np.asarray(current_list, dtype=np.int64)
                
                if rem in cand_set:
                    cand_set.discard(rem)
                if x in cand_set:
                    cand_set.discard(x)
                if rem in candidates:
                    candidates.remove(rem)
                if x in candidates:
                    candidates.remove(x)
                else:
                    candidates.append(rem)
                    cand_set.add(rem)
                
                counts = tent_counts
                counts[new_sums] += 1
                found = True
                no_improve = 0
                break
        
        if not found:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return list(current)


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
    for seed in range(30):
        if _time_left() < 15.0:
            break
        res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=seed)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    if _time_left() > 25.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    # Phase 3: Try simple structured patterns (quick checks)
    patterns = [
        list(range(1, cap + 1, 2)),  # every 2nd
        [i for i in range(1, cap + 1) if i % 3 != 0],  # skip multiples of 3
        [i for i in range(1, cap + 1) if i % 4 in (1, 2)],  # 2 per 4
        [i for i in range(1, cap + 1) if i % 5 in (1, 2, 3)],  # 3 per 5
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

    # Phase 5: Final polish pass
    if _time_left() > 10.0 and best:
        polished = _polish(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(polished) > best_len:
            best_len = len(polished)
            best = polished

    # Final validation
    if best and not _valid(best):
        fallback = _greedy_direct(list(range(1, cap + 1)), cap=cap)
        if _valid(fallback):
            best = fallback

    return best
