import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    DEADLINE = start + 55.0  # safety margin under 60s

    def now():
        return time.time()

    def check_new_element(a, existing, dp_cache):
        """Return True if adding a keeps the set valid.
        existing: list of ints (sorted not required)
        dp_cache: dict keyed by tuple(existing) -> (dp, total, max_sum_list)
        """
        m = len(existing)
        if m < 2:
            return True

        key = tuple(existing)
        cached = dp_cache.get(key)
        if cached is None:
            total = sum(existing)
            # dp[k] bitset of achievable sums with exactly k elements
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in existing:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << x)
            cached = (dp, total)
            dp_cache[key] = cached
        dp, total = cached

        # Check 1: a is mean of subset of >=2 from existing
        # need k*a achievable with exactly k elements, k>=2
        for k in range(2, m + 1):
            target = k * a
            if target > total:
                break
            if (dp[k] >> target) & 1:
                return False

        # Check 2: some b in existing becomes mean of subset (>=2) of (existing - {b}) + {a}
        # subset must contain a (else already valid). T = {a} U U, U subset of existing\{b}, |U|>=1
        # sum(U) = (|U|+1)*b - a
        for b in existing:
            others = [x for x in existing if x != b]
            n = len(others)
            if n < 1:
                continue
            max_sum = total - b
            # pre-filter: for some j in 1..n, target=(j+1)*b - a must be in [min_possible, max_sum]
            # min sum of j distinct from others >= sum of j smallest; we use loose bound j*1
            found = False
            # quick range/congruence filter over j
            for j in range(1, n + 1):
                t = (j + 1) * b - a
                if t < 0:
                    continue
                if t > max_sum:
                    break
                # need exactly j elements from others summing to t
                # lower bound: j smallest of others
                # (loose) if t < j: continue
                if t < j:
                    continue
                found = True
                break
            if not found:
                continue

            # build dp for others
            dp2 = [0] * (n + 1)
            dp2[0] = 1
            for x in others:
                for k in range(n - 1, -1, -1):
                    if dp2[k]:
                        dp2[k + 1] |= (dp2[k] << x)
            for j in range(1, n + 1):
                t = (j + 1) * b - a
                if t < 0:
                    continue
                if t > max_sum:
                    break
                if (dp2[j] >> t) & 1:
                    return False
        return True

    def full_validate(S):
        """Independent full check of set S. Returns True if valid."""
        S = list(S)
        n = len(S)
        for i, b in enumerate(S):
            others = [x for j, x in enumerate(S) if j != i]
            m = len(others)
            if m < 2:
                continue
            total = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << x)
            for k in range(2, m + 1):
                target = k * b
                if target > total:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def try_greedy(candidates, dp_cache):
        current = []
        for x in candidates:
            if len(current) >= 48:
                break
            if now() > DEADLINE:
                break
            if check_new_element(x, current, dp_cache):
                current.append(x)
        return current

    def conflict_score(x, S):
        """Heuristic: how many existing elements b could have x involved in a mean.
        Higher = more likely to conflict = add later."""
        c = 0
        for b in S:
            # x is in subset for b's mean: need (j+1)*b - x achievable; just count b where 2*b > x roughly
            if x < 2 * b or x > 2 * b + 1000:
                c += 1
        return c

    best_set = []
    best_size = 0
    rng = random.Random(42)

    # Structured passes
    for order in (list(range(512, 0, -1)), list(range(1, 513))):
        if now() > DEADLINE:
            break
        dp_cache = {}
        res = try_greedy(order, dp_cache)
        if full_validate(res) and len(res) > best_size:
            best_size = len(res)
            best_set = res
        if best_size == 48:
            return best_set

    # Random greedy passes with weighted ordering
    for seed in range(2000):
        if now() > DEADLINE:
            break
        r = random.Random(seed)
        cands = list(range(1, 513))
        r.shuffle(cands)
        dp_cache = {}
        res = try_greedy(cands, dp_cache)
        if full_validate(res) and len(res) > best_size:
            best_size = len(res)
            best_set = res
        if best_size == 48:
            return best_set

    # Local search / swaps
    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        dp_cache = {}
        improved = True
        while improved and now() < DEADLINE:
            improved = False
            # add
            for x in available:
                if len(S) >= 48:
                    break
                if now() > DEADLINE:
                    break
                if check_new_element(x, S, dp_cache):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            if len(S) >= 48:
                break
            # swap
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if now() > DEADLINE:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i + 1:]
                    for x in available:
                        if check_new_element(x, S_temp, dp_cache):
                            S = S_temp + [x]
                            S_set.discard(removed)
                            S_set.add(x)
                            available.remove(x)
                            available.append(removed)
                            improved = True
                            break
                    if improved:
                        break
        return S

    if best_set and now() < DEADLINE:
        improved = local_search(best_set)
        if full_validate(improved) and len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == 48:
            return best_set

    # Restarts with perturbation
    if best_set and now() < DEADLINE:
        for _ in range(30):
            if now() > DEADLINE:
                break
            pert = list(best_set)
            r = random.Random(now_ns := int(time.time() * 1e9))
            nr = min(r.randint(1, 3), len(pert))
            for _ in range(nr):
                if pert:
                    pert.pop(r.randrange(len(pert)))
            improved = local_search(pert)
            if full_validate(improved) and len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == 48:
                return best_set

    return best_set
