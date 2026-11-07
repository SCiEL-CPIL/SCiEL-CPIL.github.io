"""Structural rewrite: efficient backtracking with bitset sums and conflict masks."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    DEADLINE = start + 85.0

    N = 256
    MAX_SIZE = 30
    MAX_SUM = MAX_SIZE * N  # 7680

    # Precompute conflict masks using bitsets
    # conflict_mask[s]: bitset of all t != s such that s|t or t|s
    conflict_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        m = 2 * s
        while m <= MAX_SUM:
            mask |= (1 << m)
            m += s
        # Proper divisors
        for d in range(1, int(s ** 0.5) + 1):
            if s % d == 0:
                if d != s:
                    mask |= (1 << d)
                other = s // d
                if other != s and other != d:
                    mask |= (1 << other)
        conflict_mask[s] = mask

    def count_divisors(n):
        c = 0
        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c

    candidates = list(range(1, N + 1))
    div_counts = {x: count_divisors(x) for x in candidates}

    best_set = [31, 32, 34]
    best_size = 3

    def validate(elements):
        if len(elements) > MAX_SIZE:
            return False
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        sl = sorted(sums)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[j] % sl[i] == 0:
                    return False
        return True

    def try_add(x, sums_set, sums_bitset):
        """
        Try to add element x. Returns (new_sums_set, new_sums_bitset) if valid, else None.
        """
        new_sums = set()
        for s in sums_set:
            ns = s + x
            if ns in sums_set or ns in new_sums:
                return None
            new_sums.add(ns)
        new_sums.add(x)

        # Check divisibility: new sums vs existing sums
        for ns in new_sums:
            if (sums_bitset & conflict_mask[ns]) != 0:
                return None

        # Check new sums vs each other
        ns_list = list(new_sums)
        for i in range(len(ns_list)):
            for j in range(i + 1, len(ns_list)):
                a, b = ns_list[i], ns_list[j]
                if b % a == 0 or a % b == 0:
                    return None

        new_bitset = sums_bitset
        for s in new_sums:
            new_bitset |= (1 << s)
        return (sums_set | new_sums, new_bitset)

    def backtrack(current, sums_set, sums_bitset, cand_list, time_limit):
        nonlocal best_set, best_size

        if time.time() > time_limit:
            return

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

        if len(current) >= MAX_SIZE or not cand_list:
            return

        # Upper bound pruning
        if len(current) + len(cand_list) <= best_size:
            return

        for i, x in enumerate(cand_list):
            if time.time() > time_limit:
                return

            result = try_add(x, sums_set, sums_bitset)
            if result is None:
                continue

            new_sums_set, new_sums_bitset = result
            current.append(x)
            backtrack(current, new_sums_set, new_sums_bitset, cand_list[i + 1:], time_limit)
            current.pop()

    # Multiple search runs with different orderings
    rng = random.Random(777)
    orderings = [
        sorted(candidates, key=lambda x: (div_counts[x], -x)),  # few divisors, large
        sorted(candidates, reverse=True),  # descending
        sorted(candidates, key=lambda x: (-div_counts[x], x)),  # many divisors first (to fail fast?)
        candidates[:],  # will shuffle
    ]
    for _ in range(6):
        s = candidates[:]
        rng.shuffle(s)
        orderings.append(s)

    time_budget_per_run = 18.0
    for order in orderings:
        if time.time() > DEADLINE - 8:
            break
        budget = min(time_budget_per_run, DEADLINE - time.time() - 8)
        if budget < 3:
            break
        backtrack([], set(), 0, order, time.time() + budget)

    # Local search to improve
    if time.time() < DEADLINE - 12:
        def local_improve(base, tlimit):
            nonlocal best_set, best_size
            cur = list(base)
            n = len(cur)
            sums = set()
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += cur[i]
                sums.add(s)
            bitset = 0
            for s in sums:
                bitset |= (1 << s)

            if not validate(cur):
                return

            changed = True
            while changed and time.time() < tlimit:
                changed = False
                for i in range(len(cur)):
                    if time.time() > tlimit:
                        break
                    saved = cur.pop(i)
                    # Recompute sums without saved
                    m = len(cur)
                    new_sums = set()
                    for mask in range(1, 1 << m):
                        s = 0
                        for j in range(m):
                            if mask & (1 << j):
                                s += cur[j]
                        new_sums.add(s)
                    nb = 0
                    for s in new_sums:
                        nb |= (1 << s)

                    for x in range(1, N + 1):
                        if x in cur:
                            continue
                        res = try_add(x, new_sums, nb)
                        if res is not None:
                            cur.append(x)
                            changed = True
                            break
                    if not changed:
                        cur.insert(i, saved)

                if len(cur) > best_size:
                    best_size = len(cur)
                    best_set = list(cur)

        ls_time = min(15.0, DEADLINE - time.time() - 8)
        if ls_time > 3:
            local_improve(best_set, time.time() + ls_time)

    if not validate(best_set):
        best_set = [31, 32, 34]

    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
