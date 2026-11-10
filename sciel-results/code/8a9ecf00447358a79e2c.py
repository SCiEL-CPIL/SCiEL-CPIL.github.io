"""Rewritten program: branch-and-bound over {1..22} for max dissociated reciprocal set."""
import time
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L  where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    # Order candidates: Descending value (22, 21, ..., 1)
    # This corresponds to Ascending Numerator (L/22, ..., L)
    # Starting with small reciprocals (light elements) allows packing more elements
    # before the sum set becomes dense and collisions become likely.
    order = list(range(N - 1, -1, -1))  # indices for elements 22, 21, ..., 1
    vals = [universe[i] for i in order]
    nums = [num[i] for i in order]

    best = []
    best_len = 0

    def _time_ok():
        return (time.time() - start_time) < deadline

    # Iterative DFS
    # Stack items: (pos, cur_list, cur_sums)
    # We push 'exclude' then 'include' so that 'include' is processed first (LIFO)
    
    stack = [(0, [], {0})]
    
    while stack:
        pos, cur_list, cur_sums = stack.pop()
        
        if not _time_ok():
            break
            
        # Prune: even if we add all remaining, can't beat best
        remaining = N - pos
        if len(cur_list) + remaining <= best_len:
            continue
            
        if pos == N:
            if len(cur_list) > best_len:
                best_len = len(cur_list)
                best = cur_list[:]
            continue

        n = nums[pos]
        v = vals[pos]
        
        # Check collision for including n
        # Collision if (s + n) in cur_sums for any s in cur_sums
        # Using isdisjoint with a generator is efficient
        if cur_sums.isdisjoint(s + n for s in cur_sums):
            # No collision, we can include
            new_sums = cur_sums | {s + n for s in cur_sums}
            new_list = cur_list + [v]
            # Push exclude first so include is popped first
            stack.append((pos + 1, cur_list, cur_sums))
            stack.append((pos + 1, new_list, new_sums))
        else:
            # Collision, must exclude
            stack.append((pos + 1, cur_list, cur_sums))

    # Convert best (values) to sorted list
    if best:
        return sorted(best)

    # Fallback: a known dissociated set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
