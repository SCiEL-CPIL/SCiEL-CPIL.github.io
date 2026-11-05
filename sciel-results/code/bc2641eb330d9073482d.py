import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85
    n = 128
    nums = list(range(1, n + 1))

    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_valid(S):
        """Check if set S has no bad triple."""
        s_list = sorted(S)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            for j in range(i + 1, m):
                b = s_list[j]
                g_ab = gcd_table[a][b]
                for k in range(j + 1, m):
                    c = s_list[k]
                    if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                        return False
        return True

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        s_list = list(S)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            g_ax = gcd_table[a][x]
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
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, time_budget=2.0):
        """Improve set S via additions and targeted swaps."""
        S = set(S)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget and time.time() < deadline - 3:
            improved = False
            # Try to add any missing element
            for x in range(1, n + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        break

            if not improved and time.time() - start < time_budget * 0.5:
                # Try targeted swaps
                s_list = list(S)
                random.shuffle(s_list)
                for a in s_list[:min(10, len(s_list))]:
                    if time.time() - start > time_budget or time.time() > deadline - 3:
                        break
                    S_candidate = set(S)
                    S_candidate.remove(a)
                    # Try to add elements that were previously blocked
                    added_b = False
                    for b in range(1, n + 1):
                        if b not in S_candidate:
                            if can_add(b, S_candidate):
                                S_candidate.add(b)
                                added_b = True
                                # Try to add more
                                for c in range(1, n + 1):
                                    if c not in S_candidate:
                                        if can_add(c, S_candidate):
                                            S_candidate.add(c)
                                break
                    if added_b:
                        S = S_candidate
                        improved = True
                        break

        return S

    best_size = 0
    best_set = set()

    # Strategy 1: Greedy by ascending value (include small primes)
    order1 = sorted(nums)
    S = greedy_build(order1)
    S = local_search(S, time_budget=3.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 2: Greedy by descending value (include large numbers)
    order2 = sorted(nums, reverse=True)
    S = greedy_build(order2)
    S = local_search(S, time_budget=3.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 3: Multiple random restarts
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break

        order = nums[:]
        random.shuffle(order)

        S = greedy_build(order)
        S = local_search(S, time_budget=0.3)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Final local search on the best set found
    best_set = local_search(best_set, time_budget=5.0)

    return list(best_set)
