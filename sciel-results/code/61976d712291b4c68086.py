import time
import random
import bisect

def construct():
    start_time = time.time()
    limit = 80.0  # Safety margin before 90s kill

    # Pre-compute primes
    primes = [x for x in range(2, 129) if all(x % i for i in range(2, int(x**0.5) + 1))]
    primes_set = set(primes)

    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    all_nums = list(range(2, 129))
    sqfree_comp = [x for x in all_nums if x not in primes_set and is_square_free(x)]

    pool = primes + sqfree_comp
    pool.sort()
    pool_set = set(pool)

    def is_valid_full(seq):
        n = len(seq)
        prods = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    return False
                prods.add(prod)
        return True

    best_S = None
    best_len = 0

    def try_improve(S, start_time, limit):
        nonlocal best_S, best_len
        improved = True
        while improved and time.time() - start_time < limit:
            improved = False
            S_set = set(S)
            non_primes_in_S = [x for x in S if x not in primes_set]
            missing = [x for x in pool if x not in S_set]

            # 1-out/1-in swap
            for y in non_primes_in_S:
                if time.time() - start_time > limit:
                    break
                S_without = [v for v in S if v != y]
                S_without_set = set(S_without)
                for x in missing:
                    if x in S_without_set:
                        continue
                    idx = bisect.bisect_left(S_without, x)
                    new_S = S_without[:idx] + [x] + S_without[idx:]
                    if is_valid_full(new_S):
                        S = new_S
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        improved = True
                        break
                if improved:
                    break

            # 1-out/2-in move
            if not improved:
                for y in non_primes_in_S:
                    if time.time() - start_time > limit:
                        break
                    S_without = [v for v in S if v != y]
                    S_without_set = set(S_without)
                    missing_now = [x for x in pool if x not in S_without_set]
                    added = 0
                    new_S = S_without[:]
                    for x in missing_now:
                        if added >= 2:
                            break
                        idx = bisect.bisect_left(new_S, x)
                        candidate = new_S[:idx] + [x] + new_S[idx:]
                        if is_valid_full(candidate):
                            new_S = candidate
                            added += 1
                    if added >= 2:
                        if is_valid_full(new_S):
                            S = new_S
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                            improved = True
                            break

            # 2-out/3-in move
            if not improved and len(non_primes_in_S) >= 2:
                for i in range(len(non_primes_in_S)):
                    for j in range(i + 1, len(non_primes_in_S)):
                        if time.time() - start_time > limit:
                            break
                        y1, y2 = non_primes_in_S[i], non_primes_in_S[j]
                        S_without = [v for v in S if v != y1 and v != y2]
                        S_without_set = set(S_without)
                        missing_now = [x for x in pool if x not in S_without_set]
                        added = 0
                        new_S = S_without[:]
                        for x in missing_now:
                            if added >= 3:
                                break
                            idx = bisect.bisect_left(new_S, x)
                            candidate = new_S[:idx] + [x] + new_S[idx:]
                            if is_valid_full(candidate):
                                new_S = candidate
                                added += 1
                        if added >= 3:
                            if is_valid_full(new_S):
                                S = new_S
                                if len(S) > best_len:
                                    best_S = S[:]
                                    best_len = len(S)
                                improved = True
                                break
                    if improved:
                        break

        return S

    # Initial greedy
    S = primes[:]
    S.sort()
    for x in sqfree_comp:
        idx = bisect.bisect_left(S, x)
        new_S = S[:idx] + [x] + S[idx:]
        if is_valid_full(new_S):
            S = new_S

    best_S = S[:]
    best_len = len(S)

    # Local search on initial
    S = try_improve(S, start_time, limit)

    # Multiple random restarts
    for restart in range(40):
        if time.time() - start_time > limit:
            break
        shuffled = sqfree_comp[:]
        random.shuffle(shuffled)
        S = primes[:]
        S.sort()
        for x in shuffled:
            if time.time() - start_time > limit:
                break
            idx = bisect.bisect_left(S, x)
            new_S = S[:idx] + [x] + S[idx:]
            if is_valid_full(new_S):
                S = new_S
        S = try_improve(S, start_time, limit)
        if len(S) > best_len:
            best_S = S[:]
            best_len = len(S)

    # Final polish: try adding any remaining pool elements
    S_set = set(best_S)
    for x in pool:
        if time.time() - start_time > limit:
            break
        if x not in S_set:
            idx = bisect.bisect_left(best_S, x)
            new_S = best_S[:idx] + [x] + best_S[idx:]
            if is_valid_full(new_S):
                best_S = new_S
                S_set = set(best_S)
                best_len = len(best_S)

    # One more local search
    if time.time() - start_time < limit:
        best_S = try_improve(best_S, start_time, limit)
        if len(best_S) > best_len:
            best_len = len(best_S)

    return best_S
