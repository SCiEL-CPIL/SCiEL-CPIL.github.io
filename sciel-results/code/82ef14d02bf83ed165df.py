import time
import random
import math

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    best_size = 0
    best_set = []

    def rebuild_counts(S):
        counts = [0] * (2 * N + 1)
        for i, x in enumerate(S):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                counts[s] += 1
        return counts

    def can_add(x, S, counts):
        for y in S:
            if counts[x + y] >= 2:
                return False
        if counts[2 * x] >= 2:
            return False
        return True

    def greedy_build(order):
        S = []
        counts = [0] * (2 * N + 1)
        for x in order:
            if x in S:
                continue
            valid = True
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
            if valid and counts[2 * x] >= 2:
                valid = False
            if valid:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S, counts

    def try_order(order):
        nonlocal best_size, best_set
        S, _ = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # Pressure-based adaptive greedy: at each step pick the valid candidate
    # that minimizes a smooth pressure surrogate (log-sum-exp of how close each
    # affected sum is to the cap). This is a smooth surrogate for the hard
    # "max count <= 2" constraint, discouraging clustering.
    def pressure_greedy(seed, max_steps=None):
        rng = random.Random(seed)
        S = []
        counts = [0] * (2 * N + 1)
        remaining = list(range(1, N + 1))
        if max_steps is None:
            max_steps = len(remaining)

        for _ in range(max_steps):
            if not remaining:
                break
            best_x = None
            best_score = float('inf')
            # Sample a subset of candidates when set is large for speed
            if len(remaining) > 200 and len(S) > 50:
                sample = rng.sample(remaining, 200)
            else:
                sample = remaining

            for x in sample:
                # quick validity check
                ok = True
                for y in S:
                    if counts[x + y] >= 2:
                        ok = False
                        break
                if not ok:
                    continue
                if counts[2 * x] >= 2:
                    continue
                # pressure: sum over affected sums of exp((count-1)/scale)
                # higher count -> higher pressure; smooth via exp
                pressure = 0.0
                for y in S:
                    c = counts[x + y]
                    pressure += math.exp((c - 1) / 2.0)
                c2 = counts[2 * x]
                pressure += math.exp((c2 - 1) / 2.0)
                # also mild preference for smaller x to fill gaps
                score = pressure + 0.001 * x
                if score < best_score:
                    best_score = score
                    best_x = x

            if best_x is None:
                break

            # commit
            for y in S:
                counts[best_x + y] += 1
            counts[2 * best_x] += 1
            S.append(best_x)
            remaining.remove(best_x)

        return S, counts

    def local_search_swap(S, counts):
        """2-opt: try swapping each element for a non-member that fits,
        accept if set size preserved and total pressure decreases."""
        if time.time() > deadline - 8:
            return S, counts
        S = list(S)
        counts = list(counts)
        in_set = set(S)
        remaining = [x for x in range(1, N + 1) if x not in in_set]

        def total_pressure(counts):
            p = 0.0
            for c in counts:
                if c > 0:
                    p += math.exp((c - 1) / 2.0)
            return p

        improved = True
        rounds = 0
        while improved and rounds < 15 and time.time() < deadline - 8:
            improved = False
            rounds += 1
            base_pressure = total_pressure(counts)
            for i, elem in enumerate(S):
                if time.time() > deadline - 8:
                    break
                # remove elem
                new_counts = list(counts)
                for y in S:
                    if y == elem:
                        continue
                    s = elem + y
                    if new_counts[s] > 0:
                        new_counts[s] -= 1
                s2 = 2 * elem
                if new_counts[s2] > 0:
                    new_counts[s2] -= 1
                new_S = S[:i] + S[i + 1:]

                # try to find a replacement
                for cand in remaining:
                    if cand in in_set:
                        continue
                    ok = True
                    for y in new_S:
                        if new_counts[cand + y] >= 2:
                            ok = False
                            break
                    if ok and new_counts[2 * cand] >= 2:
                        ok = False
                    if not ok:
                        continue
                    # compute pressure if we add cand
                    trial_pressure = 0.0
                    for y in new_S:
                        c = new_counts[cand + y]
                        trial_pressure += math.exp((c - 1) / 2.0)
                    c2 = new_counts[2 * cand]
                    trial_pressure += math.exp((c2 - 1) / 2.0)
                    # pressure of rest is base_pressure minus elem's contribution
                    # approximate: accept if trial_pressure < base_pressure * 0.99
                    if trial_pressure < base_pressure * 0.995:
                        # commit swap
                        for y in new_S:
                            new_counts[cand + y] += 1
                        new_counts[2 * cand] += 1
                        new_S.append(cand)
                        S = new_S
                        counts = new_counts
                        in_set = set(S)
                        remaining = [x for x in range(1, N + 1) if x not in in_set]
                        improved = True
                        break

        return S, counts

    def local_search_insert(S, counts):
        """Try to insert any remaining element that fits, repeatedly."""
        if time.time() > deadline - 8:
            return S, counts
        S = list(S)
        counts = list(counts)
        in_set = set(S)
        remaining = [x for x in range(1, N + 1) if x not in in_set]

        changed = True
        while changed and time.time() < deadline - 8:
            changed = False
            for cand in remaining:
                if cand in in_set:
                    continue
                if can_add(cand, S, counts):
                    for y in S:
                        counts[cand + y] += 1
                    counts[2 * cand] += 1
                    S.append(cand)
                    in_set.add(cand)
                    remaining.remove(cand)
                    changed = True
        return S, counts

    # 1. Natural order
    try_order(list(range(1, N + 1)))

    # 2. Reverse order
    if time.time() < deadline - 30:
        try_order(list(range(N, 0, -1)))

    # 3. Primes first
    if time.time() < deadline - 30:
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n ** 0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        primes = [x for x in range(1, N + 1) if is_prime(x)]
        composites = [x for x in range(1, N + 1) if not is_prime(x)]
        try_order(primes + composites)

    # 4. Pressure-based adaptive greedy with several seeds
    for seed in [0, 1, 2, 3, 42, 123, 999]:
        if time.time() > deadline - 25:
            break
        S, _ = pressure_greedy(seed)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # 5. Random shuffles
    rng = random.Random(7)
    shuffles = 0
    while time.time() < deadline - 20 and shuffles < 40:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        try_order(order)
        shuffles += 1

    # 6. Polish best set with insertion + swap local search
    if time.time() < deadline - 10 and best_set:
        counts = rebuild_counts(best_set)
        S = local_search_insert(best_set, counts)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            counts = rebuild_counts(best_set)
        S, counts = local_search_swap(best_set, counts)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    return best_set
