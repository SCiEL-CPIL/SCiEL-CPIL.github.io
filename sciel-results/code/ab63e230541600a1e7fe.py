import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    import time, random
    from math import gcd
    start = time.time()
    DEADLINE = start + 82.0
    N = 128
    nums = list(range(1, N + 1))

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Precompute bad triples
    bad = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    bad.add((a, b, c))

    # bad_with[x] = set of (a,b) pairs (a<b) such that {a,b,x} is bad
    bad_with = [set() for _ in range(N + 1)]
    for (a, b, c) in bad:
        bad_with[c].add((a, b))
        bad_with[b].add((a, c))
        bad_with[a].add((b, c))

    def can_add(S, x):
        bw = bad_with[x]
        if not bw:
            return True
        for (a, b) in bw:
            if a in S and b in S:
                return False
        return True

    def greedy(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    # Generate diverse orderings
    def num_omega(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    orders = []
    orders.append(nums)
    orders.append(nums[::-1])
    orders.append(sorted(nums, key=lambda v: (v.bit_count(), v)))
    orders.append(sorted(nums, key=lambda v: -v.bit_count()))
    orders.append(sorted(nums, key=lambda v: (num_omega(v), v)))
    orders.append(sorted(nums, key=lambda v: -num_omega(v)))
    random.seed(12345)
    for _ in range(8):
        o = nums[:]
        random.shuffle(o)
        orders.append(o)

    best = set()
    for o in orders:
        S = greedy(o)
        if len(S) > len(best):
            best = S

    # Local search: kick-and-refill
    def local_search(S, max_iter=3000):
        S = set(S)
        best_size = len(S)
        for _ in range(max_iter):
            if time.time() > DEADLINE - 2:
                break
            kick_size = random.randint(1, 4)
            if kick_size > len(S):
                kick_size = max(1, len(S) - 1)
            if kick_size == 0:
                continue
            removed = random.sample(list(S), kick_size)
            remaining = S - set(removed)
            candidates = [x for x in nums if x not in remaining]
            random.shuffle(candidates)
            new_S = set(remaining)
            for x in candidates:
                if can_add(new_S, x):
                    new_S.add(x)
            if len(new_S) > best_size:
                best_size = len(new_S)
                S = new_S
        return S

    # Polish: try additions and small swaps
    def polish(S):
        S = set(S)
        improved = True
        while improved and time.time() < DEADLINE - 3:
            improved = False
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
            for i in list(S):
                S2 = S - {i}
                added = 0
                for x in range(1, N + 1):
                    if x not in S2 and x != i and can_add(S2, x):
                        S2.add(x)
                        added += 1
                        if added == 2:
                            break
                if added == 2:
                    S = S2
                    improved = True
        return S

    best = polish(best)
    best = local_search(best, max_iter=2000)
    best = polish(best)

    # Try local search on other good greedy starts
    for o in orders[:6]:
        if time.time() > DEADLINE - 5:
            break
        S = greedy(o)
        if len(S) >= len(best) - 2:
            S = local_search(S, max_iter=1500)
            S = polish(S)
            if len(S) > len(best):
                best = S

    if not best:
        best = {1}
    return sorted(best)
