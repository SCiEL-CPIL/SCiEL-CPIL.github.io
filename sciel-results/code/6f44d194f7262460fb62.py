import time
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s wall-clock kill

    N = 24
    nums = list(range(1, N + 1))
    n = len(nums)

    # Exact rational representation of 1/k as (num, den) = (1, k).
    # Subset sum of reciprocals is exact via integer arithmetic.

    # For each element i, find all subsets S (as frozenset of indices) whose
    # reciprocal-sum equals 1/nums[i]. We compute these by DP over subset sums.
    # 2^24 = 16M is too big for full subset enumeration in Python, so use
    # meet-in-the-middle: split into two halves, enumerate subset sums of each.

    def subset_sums_half(indices):
        # returns dict: (num, den) -> list of frozenset of indices achieving it
        m = len(indices)
        from fractions import Fraction
        # enumerate all 2^m subsets, accumulate Fraction sum
        sums = {}
        total = 1 << m
        # iterate subsets
        # For speed, do DP: sums over subsets
        # m is 12, 2^12 = 4096, fine.
        vals = [Fraction(1, nums[i]) for i in indices]
        # map from (num,den) to list of bitmasks
        acc = {}
        # use array of size 2^m: sum for each mask
        sums_arr = [Fraction(0)] * total
        masks_by_sum = {}
        for mask in range(1, total):
            # lowest set bit
            lb = mask & (-mask)
            b = lb.bit_length() - 1
            prev = mask ^ lb
            s = sums_arr[prev] + vals[b]
            sums_arr[mask] = s
            key = (s.numerator, s.denominator)
            masks_by_sum.setdefault(key, []).append(mask)
        # convert masks to frozensets of indices
        result = {}
        for key, masks in masks_by_sum.items():
            fs = set()
            for mask in masks:
                idxs = frozenset(indices[b] for b in range(m) if (mask >> b) & 1)
                fs.add(idxs)
            result[key] = fs
        return result

    left = list(range(n // 2))
    right = list(range(n // 2, n))
    left_sums = subset_sums_half(left)
    right_sums = subset_sums_half(right)

    # For each target i, 1/nums[i] = L + R where L in left_sums, R in right_sums.
    # L = 1/nums[i] - R.
    from fractions import Fraction
    conflict = [set() for _ in range(n)]
    for i in range(n):
        target = Fraction(1, nums[i])
        # iterate over smaller dict
        for (rn, rd), rfs in right_sums.items():
            # L = target - 1/(rn/rd) ... wait R is a sum, R = rn/rd
            L = target - Fraction(rn, rd)
            if L.numerator < 0:
                continue
            lkey = (L.numerator, L.denominator)
            lfs = left_sums.get(lkey)
            if not lfs:
                continue
            for lf in lfs:
                for rf in rfs:
                    S = lf | rf
                    # S must not contain i (the "other elements" condition).
                    # If i in S, then 1/nums[i] = sum of reciprocals including itself
                    # is possible only if other sum is 0, impossible for nonempty others.
                    # Actually if i in S and |S|>=1, sum >= 1/nums[i] + ... > 1/nums[i],
                    # so it can't equal target. So i won't be in S for equality. Safe.
                    if i in S:
                        continue
                    for j in S:
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)

    # Build adjacency for conflict graph. Independent set in conflict = no conflict.
    # We want max subset with no internal conflict = max independent set in conflict graph
    # = max clique in complement graph.
    # Complement: edge between i,j if NOT conflicting.
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in conflict[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)

    # Maximum clique via Bron-Kerbosch with pivot + greedy coloring bound.
    # Represent candidate set as frozenset for memo? n=24 small, but branching could be big.
    # Use bitmasks for speed.
    comp_mask = [0] * n
    for i in range(n):
        for j in comp_adj[i]:
            comp_mask[i] |= (1 << j)

    best = [0]
    best_set = [0]

    # greedy coloring of candidate set (list of vertices) returns order and color bounds
    def color_sort(cand_mask):
        # cand_mask: int bitmask of vertices
        verts = [v for v in range(n) if (cand_mask >> v) & 1]
        # greedy color
        color = {}
        order = []
        for v in verts:
            # smallest color not used by already-colored neighbors in order
            used = set()
            for u in order:
                if (comp_mask[v] >> u) & 1:
                    used.add(color[u])
            c = 1
            while c in used:
                c += 1
            color[v] = c
            order.append(v)
        return order, color

    def bk(R_mask, P_mask, X_mask):
        if time.time() > deadline:
            return
        if P_mask == 0:
            if bin(R_mask).count('1') > best[0]:
                best[0] = bin(R_mask).count('1')
                best_set[0] = R_mask
            return
        # bound: if |R| + |P| <= best, prune
        if bin(R_mask).count('1') + bin(P_mask).count('1') <= best[0]:
            return
        order, color = color_sort(P_mask)
        # iterate in reverse color order for bound
        for idx in range(len(order) - 1, -1, -1):
            v = order[idx]
            if not ((P_mask >> v) & 1):
                continue
            if bin(R_mask).count('1') + color[v] <= best[0]:
                # since colors are non-increasing as we go down, can break
                break
            bk(R_mask | (1 << v), P_mask & comp_mask[v], X_mask & comp_mask[v])
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)

    bk(0, (1 << n) - 1, 0)

    if best[0] == 0:
        return list(range(13, 25))

    result = [nums[v] for v in range(n) if (best_set[0] >> v) & 1]
    result.sort()
    return result
# EVOLVE-BLOCK-END
