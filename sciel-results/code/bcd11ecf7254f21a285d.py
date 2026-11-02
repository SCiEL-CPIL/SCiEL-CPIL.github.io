import time
import math
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        p = n
        d = 3
        while d * d <= p:
            if p % d == 0:
                break
            d += 2
        else:
            return p
        n += 2


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _inv(a, mod):
    # mod is prime
    return pow(a, mod - 2, mod)


def _girth_ok(k, g, S, q, S_neg_map, max_terms):
    # Girth >= g iff no non-backtracking closed walk of length < g exists.
    # Track (last_gen_index, sum) pairs. Vectorized with NumPy.
    target = g - 1
    S_list = list(S)
    nS = len(S_list)
    S_arr = np.array(S_list, dtype=np.int64)
    # neg_idx[j] = index of the generator that is -S_list[j]
    neg_idx = np.zeros(nS, dtype=np.int64)
    for j in range(nS):
        neg_idx[j] = S_neg_map[S_list[j]]

    # At length 1: (j, S_list[j]) for all j
    last_arr = np.arange(nS, dtype=np.int64)
    sum_arr = S_arr.copy()

    for length in range(2, target + 1):
        # Expand: for each (last, s), try all j where S_list[j] != S_list[last]
        # new_sum = (s + S_list[j]) % q, new_last = j
        # Vectorized: outer product
        # last_arr shape (M,), sum_arr shape (M,)
        # For each element, we can go to all j except neg_idx[last]
        # Build full expansion: (M * nS) candidates, then filter
        M = len(last_arr)
        # Expand sums: (M, nS)
        sums_expanded = (sum_arr[:, None] + S_arr[None, :]) % q
        # last expanded: (M, nS)
        last_expanded = np.broadcast_to(np.arange(nS, dtype=np.int64)[None, :], (M, nS))
        # Mask: exclude backtracking (j == neg_idx[last])
        # neg_idx[last] shape (M,), compare with arange(nS) shape (nS,)
        mask = last_expanded != neg_idx[last_arr][:, None]
        # Check for 0 in valid positions
        valid_sums = sums_expanded[mask]
        if len(valid_sums) > 0 and np.any(valid_sums == 0):
            return False
        # Deduplicate (last, sum) pairs
        # Pack into single int for speed: last * q + sum
        last_valid = last_expanded[mask]
        packed = last_valid * q + valid_sums
        unique_packed = np.unique(packed)
        last_arr = (unique_packed // q).astype(np.int64)
        sum_arr = (unique_packed % q).astype(np.int64)
        if len(last_arr) == 0:
            break
    return True


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []

    base_q = _next_prime(max(k, 7))

    candidates = []  # (q, S)
    
    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = 5
        while d * d <= n:
            if n % d == 0 or n % (d + 2) == 0:
                return False
            d += 6
        return True

    # Collect all primes to try
    primes_to_try = []
    for trial in range(200):
        q = base_q + trial * 11
        if q % 2 == 0:
            q += 1
        if _is_prime(q):
            primes_to_try.append(q)

    # Precompute generator pool per prime (cached)
    gen_cache = {}

    def _get_gen_pool(q, time_budget_remaining):
        if q in gen_cache:
            return gen_cache[q]
        pool = [1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
        pool += [q-1, q-2, q-3, q-5, q-7, (q+1)//2, (q+3)//2, (q-1)//2, (q-3)//2]
        # Add random values
        n_random = min(40, max(10, int(time_budget_remaining * 5)))
        for _ in range(n_random):
            pool.append(random.randint(1, q-1))
        pool = [x % q for x in pool if 0 < x % q < q]
        seen_s = set()
        pool_u = []
        for x in pool:
            if x not in seen_s:
                seen_s.add(x)
                pool_u.append(x)
        pool_u = [x for x in pool_u if x != 0]
        # Limit pool size
        if len(pool_u) > 50:
            pool_u = pool_u[:50]
        gen_cache[q] = pool_u
        return pool_u

    for q in primes_to_try:
        if time.time() > deadline - 20.0:
            break

        if k % 2 == 1:
            # k = 2*m + 1: S = {c, a, b, -a, -b}
            m = (k - 1) // 2
        else:
            m = k // 2

        pool = _get_gen_pool(q, deadline - time.time())
        found = False

        for a in pool:
            if found:
                break
            for b in pool:
                if b == a:
                    continue
                if found:
                    break
                for c in pool:
                    if c == a or c == b:
                        continue
                    if k % 2 == 1:
                        S = [c, a, b, (-a) % q, (-b) % q]
                    else:
                        S = [a, b, c, (-a) % q, (-b) % q, (-c) % q]
                    Sset = set(S)
                    if len(Sset) != k or 0 in Sset:
                        continue
                    # Check 2-sidedness
                    if Sset != {(-x) % q for x in Sset}:
                        continue
                    S_neg_map = {x: (-x) % q for x in Sset}
                    if _girth_ok(k, g, S, q, S_neg_map, 0):
                        candidates.append((q, list(S)))
                        found = True
                        break
                    if time.time() > deadline - 20.0:
                        found = True  # stop searching this q
                        break

    # Calculate LUW fallback size to compare
    # For bipartite LUW, girth is 2d+2.
    # We need 2d+2 >= g => d >= (g-2)/2
    d_luw = max(1, math.ceil((g - 2) / 2))
    q_luw = k
    n_luw = 2 * (q_luw ** d_luw)
    
    best_candidate = None
    if candidates:
        # Pick the candidate with smallest q
        candidates.sort(key=lambda t: t[0])
        q, S = candidates[0]
        if q < n_luw:
            best_candidate = (q, S)

    if best_candidate:
        q, S = best_candidate
        n = q
        S_list = list(S)
        edges = []
        for i in range(n):
            for s in S_list:
                j = (i + s) % q
                if j > i:
                    edges.append((i, j))
        return n, edges
    else:
        # Fallback: LUW-style construction (guaranteed valid, larger).
        return _luw_construct(k, g)


def _luw_construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar / Laszlo bipartite graph (valid, larger).
    q = k
    # Optimize d for girth
    # Girth of LUW is 2d+2
    # We need 2d+2 >= g
    d = max(1, math.ceil((g - 2) / 2))
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
