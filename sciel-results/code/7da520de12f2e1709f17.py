"""Maximize reciprocal sum of subset of [2,256] with pairwise lcm > 256."""
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    val = [0] * n
    for i in range(n):
        val[i] = numbers[i]

    # Precompute conflict masks (bit i conflicts with bit j)
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = val[i], val[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    weights = [1.0 / val[i] for i in range(n)]

    def sum_of(mask):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += weights[i]
        return s

    def can_add(i, mask):
        return (conflict[i] & mask) == 0

    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        if conflict[i] & (1 << j):
                            return False
        return True

    # --- Build initial solution: all of [129,256] (mutually compatible) ---
    base_mask = 0
    for i in range(n):
        if val[i] >= 129:
            base_mask |= (1 << i)

    # Try adding small numbers greedily to base
    def greedy_add(mask):
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)) and can_add(i, new_mask):
                new_mask |= (1 << i)
        return new_mask

    candidates = []
    candidates.append(base_mask)
    candidates.append(greedy_add(base_mask))

    # Alternative: start empty, greedy by weight
    def greedy_from_empty(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask

    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    candidates.append(greedy_from_empty(order_w))

    # Greedy by weight but prefer large numbers (to keep [129,256] core)
    order_large = list(range(n))
    order_large.sort(key=lambda i: -val[i])
    candidates.append(greedy_from_empty(order_large))

    # Mixed: large first then small
    order_mixed = order_large + order_w
    candidates.append(greedy_from_empty(order_mixed))

    best_mask = 0
    best_sum = -1.0
    for c in candidates:
        s = sum_of(c)
        if s > best_sum:
            best_sum = s
            best_mask = c

    # --- Exact weighted independent set via branch-and-bound ---
    # Conflict graph: edge (i,j) iff lcm(val[i],val[j]) <= N. We want a
    # maximum-weight independent set (no two selected vertices are adjacent).
    # Equivalently a maximum-weight clique in the complement. We do
    # branch-and-bound directly on the independent-set formulation.

    from fractions import Fraction

    # Exact weights for incumbent comparison; float for the bound.
    w_exact = [Fraction(1, val[i]) for i in range(n)]
    w_float = [1.0 / val[i] for i in range(n)]

    # Precompute, for each vertex, the mask of vertices that are NOT its
    # conflicts (i.e., compatible with it), excluding itself. A vertex j can
    # be added alongside i iff bit j is set in compat[i].
    compat = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        # compatible = everyone except i and except its conflicts
        compat[i] = all_mask & ~(1 << i) & ~conflict[i]

    # Order indices by descending weight so we explore high-value branches first.
    order = sorted(range(n), key=lambda i: -w_float[i])

    best_mask = 0
    best_sum_frac = None
    best_sum_float = -1.0

    def consider(mask):
        nonlocal best_mask, best_sum_frac, best_sum_float
        if mask == 0:
            return
        # Compute exact sum only when needed; cache via float first.
        s_f = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            s_f += w_float[i]
            m ^= lsb
        if best_sum_frac is None or s_f > best_sum_float + 1e-15:
            # exact check to be safe
            s_e = sum(w_exact[i] for i in range(n) if mask & (1 << i))
            if best_sum_frac is None or s_e > best_sum_frac:
                best_sum_frac = s_e
                best_sum_float = s_f
                best_mask = mask

    # Warm-start incumbent from heuristic candidates.
    for c in candidates:
        consider(c)

    # Recursion state
    import sys
    sys.setrecursionlimit(100000)

    # For bound speed, precompute weight of a mask via a dict cache? Instead
    # compute bound incrementally is hard with pivot; we recompute the sum of
    # free weights each node but that is O(popcount). To speed up, precompute
    # weight sums for all masks is infeasible (2^n). Instead, we pass the
    # current sum and compute the free-bound as (total_free_weight). We compute
    # it by summing w_float over free bits; free set shrinks along the path so
    # it's at most n. Acceptable.

    def bound_of(free):
        s = 0.0
        m = free
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            s += w_float[i]
            m ^= lsb
        return s

    def bnb(free, cur_sum, cur_mask):
        nonlocal best_mask, best_sum_frac, best_sum_float
        if elapsed() > DEADLINE:
            return
        # Update incumbent: cur_mask is a feasible independent set.
        if best_sum_frac is None or cur_sum > best_sum_float + 1e-15:
            s_e = sum(w_exact[i] for i in range(n) if cur_mask & (1 << i))
            if best_sum_frac is None or s_e > best_sum_frac:
                best_sum_frac = s_e
                best_sum_float = cur_sum
                best_mask = cur_mask
        if free == 0:
            return
        # Upper bound: cur_sum + sum of all free weights
        ub = cur_sum + bound_of(free)
        if best_sum_frac is not None and ub <= best_sum_float - 1e-15:
            return
        # Pivot: choose free vertex that minimizes the branch set size.
        # Branch set for pivot p = {p} U (free \ (compat[p] U {p}))
        #   = {p} U (free & ~compat[p] & ~(1<<p))
        best_p = -1
        best_cnt = n + 1
        m = free
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            cnt = 1 + (free & ~compat[i] & ~(1 << i)).bit_count()
            if cnt < best_cnt:
                best_cnt = cnt
                best_p = i
            m ^= lsb
        p = best_p
        # Branch set: p and the free vertices incompatible with p.
        branch = (free & ~compat[p]) | (1 << p)
        # Order branch by descending weight.
        b_list = []
        mm = branch
        while mm:
            lsb = mm & (-mm)
            i = lsb.bit_length() - 1
            b_list.append(i)
            mm ^= lsb
        b_list.sort(key=lambda i: -w_float[i])

        for i in b_list:
            if elapsed() > DEADLINE:
                return
            # Include i
            new_free = free & ~compat[i] & ~(1 << i)
            bnb(new_free, cur_sum + w_float[i], cur_mask | (1 << i))
            # Remove i from free for subsequent branches
            free = free & ~(1 << i)
            # Re-check bound for the remaining free (cheap prune)
            if best_sum_frac is not None and cur_sum + bound_of(free) <= best_sum_float - 1e-15:
                return

    # Start: free = all vertices, cur_sum = 0, cur_mask = 0.
    bnb(all_mask, 0.0, 0)

    # If the exact search was cut off by the deadline, best_mask holds the best
    # incumbent found (warm-started from heuristics). If best_mask is empty
    # (shouldn't happen since we warm-start), fall back to a candidate.
    if best_mask == 0:
        best_mask = candidates[0]

    # Final verification
    assert is_valid(best_mask), "Result is invalid!"

    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(val[i])
    result.sort()

    if not result:
        result = [2]

    return result
