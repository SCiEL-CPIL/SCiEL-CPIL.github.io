"""Matrix multiplication 3x5x6 with rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_pat, v_pat, w_pat) where patterns are dicts mapping
        local indices to values.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        return terms

    def partitions_of_size(s, max_block=2):
        """
        Generate all ways to partition {0..s-1} into consecutive blocks
        of size 1 or max_block, preserving order.
        Returns list of partitions, each a list of lists.
        """
        results = []
        def rec(remaining, current, acc):
            if not remaining:
                if current:
                    acc.append(current)
                results.append(list(acc))
                return
            # Try block of size 1
            rec(remaining[1:], [], acc + [remaining[0:1]])
            # Try block of size 2 if possible
            if len(remaining) >= 2 and 2 <= max_block:
                rec(remaining[2:], [], acc + [remaining[0:2]])
        rec(list(range(s)), [], [])
        return results

    def build_decomposition(i_groups, j_groups, k_groups, strassen_terms):
        """
        Build a decomposition: Strassen on 2x2x2 blocks, schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1

        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def search_best_partition():
        """
        Search over all partitions of i, j, k into blocks of size 1 or 2.
        Use Strassen on 2x2x2 blocks, schoolbook elsewhere.
        Return the best (lowest rank) valid decomposition.
        """
        strassen_terms = strassen_2x2x2_terms()

        i_parts = partitions_of_size(n, 2)
        j_parts = partitions_of_size(m, 2)
        k_parts = partitions_of_size(p, 2)

        best_result = None
        best_rank = n * m * p + 1

        for i_groups in i_parts:
            if time.time() > deadline:
                break
            for j_groups in j_parts:
                if time.time() > deadline:
                    break
                for k_groups in k_parts:
                    if time.time() > deadline:
                        break
                    result = build_decomposition(i_groups, j_groups, k_groups, strassen_terms)
                    rank = result["rank"]
                    if rank < best_rank:
                        if verify(result["U"], result["V"], result["W"], rank):
                            best_rank = rank
                            best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}

        return best_result, best_rank

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Search for better partition
    improved, improved_rank = search_best_partition()
    if improved is not None and improved_rank < best_rank:
        best_result = improved
        best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
