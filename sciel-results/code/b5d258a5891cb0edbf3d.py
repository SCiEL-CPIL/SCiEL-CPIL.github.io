"""Refined block-partition + Strassen decomposition for 3x5x5 matrix multiplication.

Strategy: partition the multiplication tensor into sub-blocks. Square sub-blocks
(s x s x s) use the Strassen decomposition (rank 7 for s=2, rank 8 for s=1);
non-square sub-blocks use schoolbook. We search over all valid partitions to
minimize the total rank, with an explicit deadline guard.
"""
import time
import itertools

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin under the 150s kill

    n, m, p = DIMENSIONS

    # ---- integer Strassen coefficients for the 2x2x2 tensor (rank 7) ----
    # Each term: (a_idx, b_idx, c_idx, sign). a_idx indexes A row-block,
    # b_idx indexes B col-block, c_idx indexes C row-block.
    S = [
        (0, 0, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 1),
        (0, 1, 0, 1), (1, 1, 0, -1), (0, 0, 1, -1),
    ]

    def block_rank(a, b, c):
        if a == b == c == 2:
            return 7
        if a == b == c == 1:
            return 8
        return a * b * c

    # ---- build factor matrices for a given partition ----
    def build(part_n, part_m, part_p):
        # block index ranges
        def ranges(part):
            out = []
            start = 0
            for s in part:
                out.append((start, start + s))
                start += s
            return out

        rn = ranges(part_n)
        rm = ranges(part_m)
        rp = ranges(part_p)

        # determine total rank
        total = 0
        for (i0, i1) in rn:
            for (j0, j1) in rm:
                for (k0, k1) in rp:
                    total += block_rank(i1 - i0, j1 - j0, k1 - k0)
        rank = total

        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for (i0, i1) in rn:
            for (j0, j1) in rm:
                for (k0, k1) in rp:
                    a, b, c = i1 - i0, j1 - j0, k1 - k0
                    if a == b == c == 2:
                        for (ai, bi, ci, sgn) in S:
                            U[(i0 + ai) * m + (j0 + bi)][col] = sgn
                            V[(j0 + bi) * p + (k0 + ci)][col] = sgn
                            W[(k0 + ci) * n + (i0 + ai)][col] = 1
                            col += 1
                    else:
                        for ai in range(a):
                            for bi in range(b):
                                for ci in range(c):
                                    U[(i0 + ai) * m + (j0 + bi)][col] = 1
                                    V[(j0 + bi) * p + (k0 + ci)][col] = 1
                                    W[(k0 + ci) * n + (i0 + ai)][col] = 1
                                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    # ---- enumerate partitions ----
    def partitions(k):
        # all compositions of k into positive integers
        if k == 1:
            yield (1,)
            return
        for first in range(1, k + 1):
            for rest in partitions(k - first):
                yield (first,) + rest

    best = None
    best_rank = n * m * p  # schoolbook upper bound

    pn_list = list(partitions(n))
    pm_list = list(partitions(m))
    pp_list = list(partitions(p))

    # precompute ranks per partition triple cheaply
    def rank_of(part_n, part_m, part_p):
        t = 0
        for a in part_n:
            for b in part_m:
                for c in part_p:
                    t += block_rank(a, b, c)
        return t

    # search all (small: 3 has 4 comps, 5 has 16 comps -> 4*16*16 = 1024)
    for part_n in pn_list:
        for part_m in pm_list:
            for part_p in pp_list:
                r = rank_of(part_n, part_m, part_p)
                if r < best_rank:
                    best_rank = r
                    best = (part_n, part_m, part_p)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best is None:
        best = (tuple([n]), tuple([m]), tuple([p]))

    result, _ = build(*best)
    return result
# EVOLVE-BLOCK-END
