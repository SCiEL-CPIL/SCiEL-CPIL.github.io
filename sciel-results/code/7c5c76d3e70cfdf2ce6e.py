def construct():
    import time
    import random
    import numpy as np

    deadline = time.time() + 145.0
    LIMIT = 10000
    OFFSET = LIMIT + 1
    ARR_SIZE = 2 * LIMIT + 1

    def instance(size, seed, limit=LIMIT):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit=40.0):
        start_time = time.time()
        n = len(vals)
        vals_set = set(vals)
        idx_of = {v: i for i, v in enumerate(vals)}
        v = np.array(vals, dtype=np.int64)

        # membership array for O(1) lookups: mem[val + OFFSET] = index or -1
        mem = np.full(ARR_SIZE, -1, dtype=np.int32)
        for i, val in enumerate(vals):
            mem[val + OFFSET] = i

        def inS(idx):
            # idx is an index into vals; return True if in S_mask
            return S_mask[idx]

        S_mask = None  # will be set in closures

        def can_add(x_idx, S_mask):
            x = v[x_idx]
            two_x = 2 * x
            if -LIMIT <= two_x <= LIMIT:
                j = mem[two_x + OFFSET]
                if j >= 0 and S_mask[j]:
                    return False
            y_idx = np.nonzero(S_mask)[0]
            if len(y_idx) == 0:
                return True
            y = v[y_idx]
            # x + y in S?
            s = x + y
            # filter to valid range
            valid = (s >= -LIMIT) & (s <= LIMIT)
            if np.any(valid):
                j = mem[s[valid] + OFFSET]
                j = j[j >= 0]
                if len(j) > 0 and np.any(S_mask[j]):
                    return False
            # x - y in S?
            d = x - y
            valid = (d >= -LIMIT) & (d <= LIMIT)
            if np.any(valid):
                j = mem[d[valid] + OFFSET]
                j = j[j >= 0]
                if len(j) > 0 and np.any(S_mask[j]):
                    return False
            return True

        def greedy(order):
            S_mask = np.zeros(n, dtype=bool)
            for x_idx in order:
                if can_add(x_idx, S_mask):
                    S_mask[x_idx] = True
            return S_mask

        def mask_to_set(S_mask):
            return set(vals[i] for i in range(n) if S_mask[i])

        def local_search(S_mask, rng, max_iters=20000):
            S_mask = S_mask.copy()
            size = int(S_mask.sum())
            best = S_mask.copy()
            best_size = size
            stuck = 0
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                improved = False
                not_in = np.nonzero(~S_mask)[0]
                if len(not_in) > 0:
                    rng.shuffle(not_in)
                    for x_idx in not_in:
                        if can_add(x_idx, S_mask):
                            S_mask[x_idx] = True
                            size += 1
                            improved = True
                            if size > best_size:
                                best_size = size
                                best = S_mask.copy()
                                stuck = 0
                            break
                if improved:
                    continue
                in_idx = np.nonzero(S_mask)[0]
                if len(in_idx) > 0 and len(not_in) > 0:
                    rng.shuffle(in_idx)
                    rng.shuffle(not_in)
                    for y_idx in in_idx:
                        S_mask[y_idx] = False
                        size -= 1
                        added = False
                        for x_idx in not_in:
                            if x_idx == y_idx:
                                continue
                            if can_add(x_idx, S_mask):
                                S_mask[x_idx] = True
                                size += 1
                                added = True
                                if size > best_size:
                                    best_size = size
                                    best = S_mask.copy()
                                break
                        if not added:
                            S_mask[y_idx] = True
                            size += 1
                            stuck += 1
                        else:
                            if size <= best_size:
                                S_mask[x_idx] = False
                                S_mask[y_idx] = True
                                size = best_size
                                stuck += 1
                            else:
                                stuck = 0
                        break
                else:
                    stuck += 1
                if stuck > 300:
                    break
            return best

        best_mask = np.zeros(n, dtype=bool)
        best_size = 0

        def consider(S_mask):
            nonlocal best_mask, best_size
            sz = int(S_mask.sum())
            if sz > best_size and is_sum_free(mask_to_set(S_mask)):
                best_size = sz
                best_mask = S_mask.copy()

        odds_mask = np.array([vals[i] % 2 != 0 for i in range(n)], dtype=bool)
        consider(odds_mask)
        pos_mask = np.array([vals[i] > 0 for i in range(n)], dtype=bool)
        consider(pos_mask)
        neg_mask = np.array([vals[i] < 0 for i in range(n)], dtype=bool)
        consider(neg_mask)

        order_odds_evens = [i for i in range(n) if vals[i] % 2 != 0] + \
                           [i for i in range(n) if vals[i] % 2 == 0]
        consider(greedy(order_odds_evens))

        orders = [
            sorted(range(n), key=lambda i: -abs(vals[i])),
            sorted(range(n), key=lambda i: vals[i]),
            sorted(range(n), key=lambda i: -vals[i]),
            sorted(range(n), key=lambda i: abs(vals[i])),
            sorted(range(n), key=lambda i: vals[i] % 13),
        ]
        for order in orders:
            if time.time() - start_time > time_limit * 0.25:
                break
            consider(greedy(order))

        rng = random.Random(12345)
        starts = [best_mask.copy()]
        for _ in range(6):
            m = np.zeros(n, dtype=bool)
            sample = rng.sample(range(n), max(1, n // 2))
            m[sample] = True
            S = mask_to_set(m)
            S = set(x for x in S if all((x + y) not in S and (x - y) not in S for y in S if y != x) and 2 * x not in S)
            m2 = np.zeros(n, dtype=bool)
            for x in S:
                m2[idx_of[x]] = True
            starts.append(m2)

        for s_mask in starts:
            if time.time() - start_time > time_limit:
                break
            improved = local_search(s_mask, rng, max_iters=15000)
            consider(improved)

        if time.time() - start_time < time_limit:
            improved = local_search(best_mask, rng, max_iters=30000)
            consider(improved)

        return mask_to_set(best_mask)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 70.0
        S = solve_instance(vals, t_lim)
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
        results.append(sorted(S))
        if time.time() > deadline:
            break
    while len(results) < 3:
        results.append([])
    return results[:3]
