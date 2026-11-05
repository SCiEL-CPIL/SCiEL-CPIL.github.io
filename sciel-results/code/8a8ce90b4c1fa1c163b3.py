import time
import random
from math import gcd

def construct():
    deadline = time.time() + 84.0
    n = 128
    nums = list(range(1, n + 1))

    # Precompute GCD table (indexed by value 1..n)
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # ---- Fast conflict check (current program's structural idea) ----
    def can_add_fast(x, S_set):
        """Check if x can be added to set S without creating a bad triple."""
        s_list = list(S_set)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            g_ax = gcd_table[a][x]
            if g_ax == 1:
                for j in range(i + 1, m):
                    b = s_list[j]
                    if gcd_table[a][b] == 1 and gcd_table[b][x] == 1:
                        return False
            else:
                for j in range(i + 1, m):
                    b = s_list[j]
                    if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                        return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add_fast(x, S):
                S.add(x)
        return S

    def addable_any(S):
        for x in range(1, n + 1):
            if x not in S and can_add_fast(x, S):
                return True
        return False

    # ---- Heuristic phase: diverse greedy + local search (current program) ----
    best_size = 0
    best_set = set()

    def make_orderings():
        orderings = []
        for _ in range(3):
            shuffled = nums[:]
            random.shuffle(shuffled)
            orderings.append(shuffled)

        def num_prime_factors(x):
            count = 0
            d = 2
            while d * d <= x:
                while x % d == 0:
                    count += 1
                    x //= d
                d += 1
            if x > 1:
                count += 1
            return count

        orderings.append(sorted(nums, key=num_prime_factors))
        orderings.append(sorted(nums, key=num_prime_factors, reverse=True))
        orderings.append(sorted(nums))
        orderings.append(sorted(nums, reverse=True))
        for _ in range(2):
            shuffled = nums[:]
            random.shuffle(shuffled)
            for _ in range(10):
                i, j = random.randint(0, len(shuffled) - 1), random.randint(0, len(shuffled) - 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            orderings.append(shuffled)
        return orderings

    for order in make_orderings():
        if time.time() > deadline - 6:
            break
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # random restarts
    for i in range(6000):
        if time.time() > deadline - 5:
            break
        shuffled = nums[:]
        random.shuffle(shuffled)
        S = greedy_build(shuffled)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # final add pass
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, n + 1):
            if x not in best_set and can_add_fast(x, best_set):
                best_set.add(x)
                best_size += 1
                improved = True
                break

    # swap-based local search (1-for-1)
    if time.time() < deadline - 5:
        for _ in range(6):
            if time.time() > deadline - 5:
                break
            improved = False
            s_list = list(best_set)
            rem_list = [x for x in range(1, n + 1) if x not in best_set]
            for a in s_list:
                if time.time() > deadline - 5:
                    break
                S_minus_a = best_set - {a}
                for x in rem_list:
                    if can_add_fast(x, S_minus_a):
                        best_set = S_minus_a
                        best_set.add(x)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
            # re-add after a swap
            improved2 = True
            while improved2 and time.time() < deadline - 5:
                improved2 = False
                for x in range(1, n + 1):
                    if x not in best_set and can_add_fast(x, best_set):
                        best_set.add(x)
                        improved2 = True
                        break
            best_size = len(best_set)

    # ---- Exact branch-and-bound on the bad-triple hypergraph (inspiration 2) ----
    # Build bad triples using bitmasks of "coprime to x" for the gcd==1 case.
    coprime_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        m = 0
        for y in range(1, n + 1):
            if gcd_table[x][y] == 1:
                m |= (1 << y)
        coprime_mask[x] = m

    bad_triples = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd_table[i][j]
            if g == 1:
                # need k coprime to both i and j
                cand = coprime_mask[i] & coprime_mask[j]
                cand &= ~((1 << (j + 1)) - 1)  # k > j
                k = cand & (-cand)
                while k:
                    bad_triples.append((i, j, k.bit_length() - 1))
                    k &= k - 1
            else:
                for k in range(j + 1, n + 1):
                    if gcd_table[i][k] == g and gcd_table[j][k] == g:
                        bad_triples.append((i, j, k))

    # conflict adjacency (bitmask per value)
    conf = [0] * (n + 1)
    for (a, b, c) in bad_triples:
        conf[a] |= (1 << b) | (1 << c)
        conf[b] |= (1 << a) | (1 << c)
        conf[c] |= (1 << a) | (1 << b)

    def popcount(x):
        return bin(x).count("1")

    # greedy upper bound: repeatedly take a vertex, remove it and its neighbors
    def greedy_bound(mask):
        cnt = 0
        m = mask
        while m:
            cnt += 1
            v = (m & -m).bit_length() - 1
            m &= ~conf[v]
            m &= ~(1 << v)
        return cnt

    # exact branch-and-bound (max independent set in the conflict graph)
    best_bnb = [best_size]
    best_bnb_set = [set(best_set)]
    bnb_calls = [0]
    memo = {}

    def bnb(mask, size, chosen):
        if time.time() > deadline - 1.0:
            return
        bnb_calls[0] += 1
        if size + popcount(mask) <= best_bnb[0]:
            return
        if mask == 0:
            if size > best_bnb[0]:
                best_bnb[0] = size
                best_bnb_set[0] = set(chosen)
            return
        ub = size + greedy_bound(mask)
        if ub <= best_bnb[0]:
            return
        key = mask
        if key in memo and memo[key] <= best_bnb[0] - size:
            return
        # branch on a vertex with high degree within the remaining set
        m = mask
        best_v = -1
        best_deg = -1
        while m:
            v = (m & -m).bit_length() - 1
            deg = popcount(mask & conf[v])
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m &= m - 1
        v = best_v
        # include v
        bnb(mask & ~(conf[v] | (1 << v)), size + 1, chosen | {v})
        # exclude v
        bnb(mask & ~(1 << v), size, chosen)
        memo[key] = ub - size

    if time.time() < deadline - 2.0 and best_size > 0:
        full_mask = 0
        for x in range(1, n + 1):
            full_mask |= (1 << x)
        bnb(full_mask, 0, set())

    if best_bnb[0] > best_size:
        best_size = best_bnb[0]
        best_set = set(best_bnb_set[0])

    # ---- Final polish: add + 1-for-1 swap + 1-for-2 (inspiration 1) ----
    if time.time() < deadline - 0.5:
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for x in range(1, n + 1):
                if x not in best_set and can_add_fast(x, best_set):
                    best_set.add(x)
                    improved = True
                    break
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            improved = False
            s_list = list(best_set)
            rem_list = [x for x in range(1, n + 1) if x not in best_set]
            for a in s_list:
                if time.time() > deadline - 0.5:
                    break
                S_minus_a = best_set - {a}
                for x in rem_list:
                    if can_add_fast(x, S_minus_a):
                        best_set = S_minus_a
                        best_set.add(x)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        # 1-for-2 swap
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            improved = False
            s_list = list(best_set)
            rem_list = [x for x in range(1, n + 1) if x not in best_set]
            for a in s_list:
                if time.time() > deadline - 0.5:
                    break
                S_minus_a = best_set - {a}
                added = []
                for y in rem_list:
                    if len(added) >= 2:
                        break
                    if can_add_fast(y, S_minus_a | set(added)):
                        added.append(y)
                if len(added) >= 2:
                    best_set = S_minus_a | set(added)
                    improved = True
                    break
            if not improved:
                break
            improved2 = True
            while improved2 and time.time() < deadline - 0.5:
                improved2 = False
                for x in range(1, n + 1):
                    if x not in best_set and can_add_fast(x, best_set):
                        best_set.add(x)
                        improved2 = True
                        break

    # final safety add
    for x in range(1, n + 1):
        if x not in best_set and can_add_fast(x, best_set):
            best_set.add(x)

    if not best_set:
        return [1]
    return list(best_set)
