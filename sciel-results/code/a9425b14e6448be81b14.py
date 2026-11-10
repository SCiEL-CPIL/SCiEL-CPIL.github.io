"""Rewritten program: branch-and-bound over {1..22} for max dissociated reciprocal set."""
import time
import random
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L  where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    best = []
    best_len = 0

    def _time_ok():
        return (time.time() - start_time) < deadline

    def _has_collision(cur_sums, n):
        """Check if adding n causes any collision in subset sums."""
        # Collision if (s + n) in cur_sums for any s in cur_sums
        # Equivalent: cur_sums and (cur_sums + n) are not disjoint
        for s in cur_sums:
            if s + n in cur_sums:
                return True
        return False

    def _dfs(order, time_budget=None):
        nonlocal best, best_len
        
        if time_budget is None:
            time_budget = deadline - start_time
        
        # Iterative DFS
        stack = [(0, [], {0})]
        
        while stack:
            pos, cur_list, cur_sums = stack.pop()
            
            if not _time_ok():
                return
                
            # Prune: even if we add all remaining, can't beat best
            remaining = N - pos
            if len(cur_list) + remaining <= best_len:
                continue
                
            if pos == N:
                if len(cur_list) > best_len:
                    best_len = len(cur_list)
                    best = cur_list[:]
                continue

            idx = order[pos]
            n = num[idx]
            v = universe[idx]
            
            # Check collision for including n
            if not _has_collision(cur_sums, n):
                # No collision, we can include
                new_sums = cur_sums | {s + n for s in cur_sums}
                new_list = cur_list + [v]
                # Push exclude first so include is popped first
                stack.append((pos + 1, cur_list, cur_sums))
                stack.append((pos + 1, new_list, new_sums))
            else:
                # Collision, must exclude
                stack.append((pos + 1, cur_list, cur_sums))

    # Precompute conflict degrees for ordering
    # For each element, count how many other elements it directly conflicts with
    # (i.e., 1/a + 1/b = 1/c for some c in universe, or 1/a = 1/b + 1/c)
    conflict_degree = [0] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Check if 1/universe[i] + 1/universe[j] equals 1/universe[k] for some k
            # Or if 1/universe[i] = 1/universe[j] + 1/universe[k] for some k
            # Simplified: check pairwise sum conflicts
            si = num[i]
            sj = num[j]
            for k in range(N):
                if k == i or k == j:
                    continue
                if si + sj == num[k]:
                    conflict_degree[i] += 1
                    break

    # Order 1: Ascending denominator (1, 2, ..., 22)
    order1 = list(range(N))
    _dfs(order1)
    
    # Order 2: Descending denominator (22, 21, ..., 1)
    if _time_ok():
        order2 = list(range(N-1, -1, -1))
        _dfs(order2)

    # Order 3: By conflict degree (low to high - try less-conflicting elements first)
    if _time_ok():
        order3 = sorted(range(N), key=lambda x: conflict_degree[x])
        _dfs(order3)

    # Order 4-10: Randomized with different seeds
    for seed in range(7):
        if not _time_ok():
            break
        rng = random.Random(seed)
        order = list(range(N))
        rng.shuffle(order)
        _dfs(order)

    # Convert best (values) to sorted list
    if best:
        return sorted(best)

    # Fallback: a known dissociated set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
