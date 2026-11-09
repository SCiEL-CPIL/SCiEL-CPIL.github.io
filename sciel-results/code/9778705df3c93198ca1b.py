import time
import random
import sys
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # explicit safety margin for the 60s / 90s limit

    max_val = 4096
    max_sum = 3 * max_val  # 12288

    best_set = []
    best_size = 0

    def new_sums_for(x, arr):
        """All triple sums that involve the new element x and members of arr (sorted)."""
        n = len(arr)
        a = np.array(arr, dtype=np.int64)
        parts = [3 * x]
        if n:
            parts.append(2 * x + a)
            parts.append(x + 2 * a)
            if n > 1:
                ii, jj = np.triu_indices(n)
                parts.append(x + a[ii] + a[jj])
        return np.concatenate(parts)

    def try_add_fast(x, arr, used_sums):
        """Try to add x to sorted arr; return (ok, new_sums_array).

        Checks both against existing sums AND internal collisions among the
        new sums (the latter is what the current program was missing)."""
        ns = new_sums_for(x, arr)
        if not ns.isdisjoint(used_sums):
            return False, ns
        if len(np.unique(ns)) != len(ns):
            return False, ns
        return True, ns

    def add_sums(used_sums, ns):
        used_sums.update(int(v) for v in ns)

    def greedy_build(order, deadline, dense):
        current = []
        used = set()
        for x in order:
            if time.time() > deadline - 1.0:
                break
            if dense:
                # Densest-subsequence step: score a window of candidates at once
                # and pick the one with the fewest collisions (0 => add).
                idx = order.index(x) if x in order else 0
                window = order[idx:idx + 128]
                best_x, best_ns = None, None
                best_coll = None
                for cand in window:
                    ns = new_sums_for(cand, current)
                    coll = len(ns) - len(np.unique(ns))
                    coll += int(np.intersect1d(ns, np.fromiter(used, dtype=np.int64), assume_unique=False).size)
                    if best_coll is None or coll < best_coll:
                        best_coll, best_x, best_ns = coll, cand, ns
                    if coll == 0:
                        break
                if best_x is not None and best_coll == 0:
                    used_sums = add_sums(used, best_ns)
                    current.append(best_x)
            else:
                ok, ns = try_add_fast(x, current, used)
                if ok:
                    add_sums(used, ns)
                    current.append(x)
        return current

    def build_used_sums(arr):
        used = set()
        a = np.array(arr, dtype=np.int64)
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    used.add(int(a[i] + a[j] + a[k]))
        return used

    def local_search(initial_set, deadline):
        current = sorted(initial_set)
        used = build_used_sums(current)
        in_set = set(current)
        candidates = [v for v in range(1, max_val + 1) if v not in in_set]

        for _ in range(200):
            if time.time() > deadline - 1.5:
                break
            improved = False
            n = len(current)
            for i in range(n):
                if time.time() > deadline - 1.5:
                    break
                removed = current[i]
                temp = current[:i] + current[i + 1:]
                temp_used = build_used_sums(temp)
                # candidates closest to removed keep density
                if len(candidates) > 200:
                    d = [abs(c - removed) for c in candidates]
                    idx_sorted = sorted(range(len(candidates)), key=lambda k: d[k])
                    trial = [candidates[k] for k in idx_sorted[:200]]
                else:
                    trial = candidates
                found = False
                for cand in trial:
                    if time.time() > deadline - 1.5:
                        break
                    ok, ns = try_add_fast(cand, temp, temp_used)
                    if ok:
                        add_sums(temp_used, ns)
                        current = temp + [cand]
                        current.sort()
                        used = temp_used
                        in_set.add(cand)
                        candidates.remove(cand)
                        improved = True
                        found = True
                        break
                if not found:
                    pass  # keep original
            if not improved:
                break
        return current

    # Diverse orderings
    base = list(range(1, max_val + 1))
    orders = [base[:], base[::-1]]
    for mod in [3, 5, 7, 11, 13]:
        orders.append(sorted(base, key=lambda x: (x % mod, x)))
    rng = random.Random(42)
    for _ in range(6):
        c = base[:]
        rng.shuffle(c)
        orders.append(c)
    for bs, be in [(1, 100), (100, 500), (500, 2000), (2000, 4096)]:
        c = base[:]
        rng.shuffle(c)
        biased = [x for x in c if bs <= x <= be]
        rest = [x for x in c if not (bs <= x <= be)]
        orders.append(biased + rest)

    for idx, order in enumerate(orders):
        if time.time() > deadline - 3.0:
            break
        # alternate: dense selection on some orders, pure greedy on others
        result = greedy_build(order, deadline, dense=(idx % 3 == 0))
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            if best_size >= 30 and time.time() < deadline - 8.0:
                improved = local_search(best_set, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]

    return best_set
