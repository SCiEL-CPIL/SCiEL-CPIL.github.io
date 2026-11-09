"""Solve the packing problem: find 10 distinct positive integers <= 1000000,
each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
minimizing max - min."""

import time
import math
from collections import deque

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000


def _max_matching(adj, n_left, n_right):
    """Find maximum bipartite matching using Hopcroft-Karp.
    adj[i] = list of right-node indices adjacent to left node i.
    Returns (matching_size, assignment) where assignment[i] = right node or -1.
    """
    INF = float('inf')
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found

    def dfs(u):
        for v in adj[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1

    return matching, pair_u


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit

    divisors = DIVISORS[:]
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)

    # Precompute: for each divisor i, the sorted list of positions (0-indexed in period)
    # where a multiple of divisors[i] occurs.
    # pos[i] = sorted list of r in [0, lcm) such that (r+1) % divisors[i] == 0
    # Actually, let's use 1-indexed: value v is a multiple of d iff v % d == 0.
    # In period [1, lcm], multiples of d are d, 2d, 3d, ..., (lcm//d)*d
    pos = []
    for i, d in enumerate(divisors):
        multiples = list(range(d, lcm + 1, d))
        pos.append(multiples)

    best_result = None
    best_width = float('inf')

    # For a window [L, L+W] (1-indexed, inclusive), a value v is in the window
    # iff L <= v <= L+W. In terms of period position p = ((v-1) % lcm) + 1,
    # the window covers certain positions.
    #
    # Key insight: for small W (< lcm), the window [L, L+W] wraps around the period
    # at most once. We can handle this by considering the window in the "unwrapped"
    # line and mapping back.
    #
    # For each W, we need to check all L in [1, lcm] (one full period).
    # For a given L and W, the set of values in [L, L+W] that are multiples of d_i
    # can be determined by checking which elements of pos[i] fall in [L, L+W],
    # handling wrap-around if L+W > lcm.
    #
    # To make this fast, for each divisor i, we precompute pos[i] as a sorted list.
    # For a query interval [a, b] (possibly wrapping), we can use binary search
    # to find how many multiples of d_i are in the interval.
    #
    # But we also need the actual values for the matching. Let's think differently.
    #
    # Alternative: iterate W from 1 upward. For each W, iterate L from 1 to lcm.
    # For each (L, W), build the adjacency: for each divisor i, which values in
    # [L, L+W] are multiples of d_i. Then check if a perfect matching exists.
    #
    # The number of values in [L, L+W] that are multiples of d_i is at most
    # floor((L+W)/d_i) - floor((L-1)/d_i) + 1, which is at most W//d_i + 1.
    # For W < 200 and d_i >= 6, this is at most 34 values per divisor.
    # Total values in window that are multiples of any divisor: at most sum of these.
    # For 10 divisors, at most ~10*34 = 340 values. But many overlap.
    #
    # Actually, let's just enumerate the distinct values in [L, L+W] that are
    # multiples of at least one divisor, build the bipartite graph, and run matching.
    #
    # To speed up: for each W, the set of "candidate values" (multiples of any divisor)
    # in any window of width W is bounded. Let's precompute for each position in the
    # period, which divisors divide that position.

    # Precompute: for each position p in [1, lcm], which divisors divide p.
    # divisors_at[p] = list of i such that p % divisors[i] == 0
    divisors_at = [[] for _ in range(lcm + 1)]
    for i, d in enumerate(divisors):
        for v in range(d, lcm + 1, d):
            divisors_at[v].append(i)

    # Now for a window [L, L+W], the candidate values are those v in [L, L+W]
    # with non-empty divisors_at[v]. For each such v, the adjacent divisors are divisors_at[v].
    #
    # For efficiency, let's precompute a "next multiple" structure or just iterate.
    #
    # Given the constraints (lcm=30030, W up to maybe 300), let's just iterate.
    #
    # For each W from 1 to 300:
    #   For each L from 1 to lcm:
    #     Build list of candidate values in [L, L+W] (handling wrap)
    #     Build bipartite graph
    #     Check perfect matching
    #     If found, record and break

    for W in range(1, 400):
        if time.time() > deadline:
            break

        found = False
        for L in range(1, lcm + 1):
            if time.time() > deadline:
                break

            # Get candidate values in [L, L+W], handling wrap-around
            # Values are 1-indexed in [1, lcm]
            # Window [L, L+W] may wrap if L+W > lcm
            candidates = []  # list of (value, divisors_that_divide_it)
            if L + W <= lcm:
                for v in range(L, L + W + 1):
                    if divisors_at[v]:
                        candidates.append((v, divisors_at[v]))
            else:
                # Wrap: [L, lcm] U [1, L+W-lcm]
                for v in range(L, lcm + 1):
                    if divisors_at[v]:
                        candidates.append((v, divisors_at[v]))
                for v in range(1, L + W - lcm + 1):
                    if divisors_at[v]:
                        candidates.append((v, divisors_at[v]))

            if len(candidates) < N:
                continue

            # Build bipartite graph: left = divisors (0..N-1), right = candidates (0..len-1)
            # adj[i] = list of candidate indices j such that divisor i divides candidates[j]
            n_cand = len(candidates)
            adj = [[] for _ in range(N)]
            for j, (v, divs) in enumerate(candidates):
                for i in divs:
                    adj[i].append(j)

            # Check if every divisor has at least one candidate
            feasible = True
            for i in range(N):
                if not adj[i]:
                    feasible = False
                    break
            if not feasible:
                continue

            # Run Hopcroft-Karp
            matching_size, pair_u = _max_matching(adj, N, n_cand)
            if matching_size == N:
                # Extract assignment
                assignment = []
                for i in range(N):
                    v = candidates[pair_u[i]][0]
                    # Convert to actual value (handle wrap if needed)
                    # The value is already in [1, lcm], which is valid
                    assignment.append(v)

                width = max(assignment) - min(assignment)
                if width < best_width:
                    best_width = width
                    best_result = assignment[:]
                found = True
                break

        if found:
            break

    # Fallback
    if best_result is None:
        best_result = [d for d in divisors]
        best_width = max(best_result) - min(best_result)

    result = [int(x) for x in best_result]
    return result


# Test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Width: {max(result) - min(result)}")
    for i, (d, v) in enumerate(zip(DIVISORS, result)):
        print(f"  Divisor {d}: value {v}, divisible: {v % d == 0}")
    print(f"Distinct: {len(set(result)) == len(result)}")
    print(f"All <= 1000000: {all(v <= 1000000 for v in result)}")
    print(f"All positive: {all(v > 0 for v in result)}")
