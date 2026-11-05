"""Fresh design: exact MIS via branch-and-bound on a sparse square-sum graph,
with fast greedy/LP-based heuristics, 1-flip local search, and a deadline guard."""
import math
import time
import random
import sys

sys.setrecursionlimit(100000)


def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under the 90s kill

    # ---- Build the conflict graph over even numbers 2..256 (diagonal-safe) ----
    # Odd numbers can never be in a valid set (2*a is odd, never a square),
    # so restrict to evens. This keeps the graph small (128 vertices).
    nodes = list(range(2, 257, 2))
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    # adjacency bitmasks
    adj = [0] * n
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            s = a + b
            r = math.isqrt(s)
            if r * r == s:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def is_valid(S):
        # S is a set of node indices; verify no edge inside S
        for i in S:
            if adj[i] & S:
                return False
        return True

    # ---- Heuristic 1: greedy (randomized, repeated) ----
    def greedy(seed=None):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        S = 0
        for i in order:
            if adj[i] & S == 0:
                S |= (1 << i)
        return S

    # ---- Heuristic 2: degree-ordered greedy (fewest-conflict-first) ----
    def greedy_degree(seed=None):
        rng = random.Random(seed)
        deg = [bin(adj[i]).count("1") for i in range(n)]
        # sort by (degree, random tiebreak)
        order = sorted(range(n), key=lambda i: (deg[i], rng.random()))
        S = 0
        for i in order:
            if adj[i] & S == 0:
                S |= (1 << i)
        return S

    # ---- Local search: 1-flip (add) moves ----
    def local_search(S, seed=None):
        rng = random.Random(seed)
        improved = True
        while improved:
            improved = False
            # candidates to add: not in S, and no conflict with S
            cand = []
            for i in range(n):
                if (S >> i) & 1:
                    continue
                if adj[i] & S == 0:
                    cand.append(i)
            rng.shuffle(cand)
            for i in cand:
                S |= (1 << i)
                improved = True
                break
        return S

    # ---- Exact maximum independent set via branch and bound (bitmask) ----
    def exact_mis():
        best = [0]  # size
        best_set = [0]

        # upper bound: greedy coloring of the candidate subgraph (Ramsey bound)
        def color_bound(cand):
            # greedy coloring of induced subgraph on cand; number of colors = UB
            remaining = cand
            color = 0
            while remaining:
                color += 1
                # pick a maximal independent set from remaining as one color class
                cls = 0
                rem = remaining
                # order by degree within remaining
                degs = []
                r = rem
                while r:
                    lsb = r & (-r)
                    i = lsb.bit_length() - 1
                    degs.append((bin(adj[i] & rem).count("1"), i))
                    r -= lsb
                degs.sort()
                for _, i in degs:
                    if (rem >> i) & 1 and (adj[i] & cls) == 0:
                        cls |= (1 << i)
                        rem &= ~(1 << i)
                remaining = rem
            return color

        def rec(cand, S):
            if time.time() > DEADLINE - 3.0:
                return
            if cand == 0:
                if bin(S).count("1") > best[0]:
                    best[0] = bin(S).count("1")
                    best_set[0] = S
                return
            # bound
            ub = color_bound(cand)
            if bin(S).count("1") + ub <= best[0]:
                return
            # pick a vertex with max degree in cand for branching
            r = cand
            best_i = -1
            best_deg = -1
            while r:
                lsb = r & (-r)
                i = lsb.bit_length() - 1
                d = bin(adj[i] & cand).count("1")
                if d > best_deg:
                    best_deg = d
                    best_i = i
                r -= lsb
            i = best_i
            # branch: include i
            rec(cand & ~(1 << i) & ~adj[i], S | (1 << i))
            # branch: exclude i
            rec(cand & ~(1 << i), S)

        rec((1 << n) - 1, 0)
        return best_set[0]

    # ---- Run heuristics first to get a strong lower bound ----
    best = 0
    best_set = 0
    for seed in range(30):
        S = greedy(seed)
        S = local_search(S, seed)
        if bin(S).count("1") > best:
            best = bin(S).count("1")
            best_set = S
    for seed in range(30):
        S = greedy_degree(seed)
        S = local_search(S, seed + 1000)
        if bin(S).count("1") > best:
            best = bin(S).count("1")
            best_set = S

    # ---- Try exact solver if time remains; use best as initial bound ----
    if time.time() < DEADLINE - 5.0:
        # seed the exact solver with current best
        # (exact_mis recomputes; we just call it and compare)
        exact_result = exact_mis()
        if bin(exact_result).count("1") > best:
            best = bin(exact_result).count("1")
            best_set = exact_result

    # ---- Convert bitmask to list of values, ensure nonempty and valid ----
    result = [nodes[i] for i in range(n) if (best_set >> i) & 1]
    if not result:
        # fallback: a single even number is always valid
        result = [2]

    # final safety check
    if not is_valid(set(i for i, v in enumerate(nodes) if v in result)):
        # rebuild a valid set greedily to be safe
        S = 0
        for i in range(n):
            if adj[i] & S == 0:
                S |= (1 << i)
        result = [nodes[i] for i in range(n) if (S >> i) & 1]
        if not result:
            result = [2]

    return result
