import time
import random

def construct():
    deadline = time.time() + 85.0
    N = 256
    MAX_SIZE = 60

    def check_valid(vals):
        """Check cardinality-separated subset sums using bitset DP per cardinality."""
        if not vals:
            return True
        n = len(vals)
        # sums[k] is a Python int bitset: bit s set means sum s achievable with exactly k elements
        sums = [0] * (n + 1)
        sums[0] = 1
        for v in vals:
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        # Check pairwise disjointness across cardinalities
        seen = 0
        for i in range(n + 1):
            if sums[i] & seen:
                return False
            seen |= sums[i]
        return True

    def try_greedy(pool, max_size, reverse=True):
        result = []
        for v in sorted(pool, reverse=reverse):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_grow(initial, all_vals):
        cur = list(initial)
        cur_set = set(cur)
        for v in sorted(all_vals):
            if v in cur_set:
                continue
            if len(cur) >= MAX_SIZE:
                break
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
        return cur

    pool = list(range(1, N + 1))
    best = []

    # Baseline: superincreasing
    sup = [1]
    for i in range(1, 60):
        nxt = sum(sup) + 1
        if nxt <= N:
            sup.append(nxt)
        else:
            break
    if check_valid(sup):
        best = list(sup)

    # Greedy descending
    desc = try_greedy(pool, MAX_SIZE, reverse=True)
    if len(desc) > len(best):
        best = desc

    if time.time() > deadline:
        return [int(x) for x in best]

    # Greedy ascending
    asc = try_greedy(pool, MAX_SIZE, reverse=False)
    if len(asc) > len(best):
        best = asc

    # Grow from superincreasing
    grown = try_grow(sup, pool)
    if len(grown) > len(best):
        best = grown

    # Local search: try to add elements via swaps
    def local_search(cur, pool, max_size, time_limit):
        cur = list(cur)
        cur_set = set(cur)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each remaining element
            for v in sorted(pool):
                if v in cur_set:
                    continue
                if len(cur) >= max_size:
                    break
                cand = cur + [v]
                if check_valid(cand):
                    cur = cand
                    cur_set.add(v)
                    improved = True
            # Try swaps: remove one, add one
            if len(cur) < max_size and not improved:
                for v_out in sorted(cur, key=lambda x: x, reverse=True):
                    for v_in in sorted(pool):
                        if v_in in cur_set:
                            continue
                        if v_in == v_out:
                            continue
                        cand = [x for x in cur if x != v_out] + [v_in]
                        if check_valid(cand):
                            cur = cand
                            cur_set = set(cur)
                            improved = True
                            break
                    if improved:
                        break
        return cur

    remaining = deadline - time.time()
    if remaining > 5:
        ls_time = min(15.0, remaining * 0.4)
        ls_result = local_search(best, pool, MAX_SIZE, time_limit=ls_time)
        if len(ls_result) > len(best):
            best = ls_result

    # Randomized restarts
    remaining = deadline - time.time()
    if remaining > 3:
        rng = random.Random(42)
        iters = min(50, int(remaining * 10))
        for _ in range(iters):
            if time.time() > deadline - 1:
                break
            shuffled = pool[:]
            rng.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= MAX_SIZE:
                    break
                candidate = result + [v]
                if check_valid(candidate):
                    result = candidate
            if len(result) > len(best):
                best = result
            # Try to grow
            grown = try_grow(result, pool)
            if len(grown) > len(best):
                best = grown

    return [int(x) for x in best]
