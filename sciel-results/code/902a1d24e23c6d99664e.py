"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit

    def make_schoolbook(n, m, p):
        """Return (U, V, W) as lists of lists for schoolbook rank n*m*p."""
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
        return U, V, W

    def verify(U, V, W, rank, n, m, p):
        """Verify exact decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def strassen_2x2(n1, m1, p1, offset_i, offset_j, offset_k):
        """
        Decompose a (n1 x m1) @ (m1 x p1) sub-problem using Strassen if n1==m1==p1==2,
        else schoolbook. Returns (U, V, W) as lists with local indexing, and the rank used.
        U has n1*m1 rows, V has m1*p1 rows, W has p1*n1 rows.
        """
        if n1 == 2 and m1 == 2 and p1 == 2:
            # Strassen rank-7 decomposition for 2x2
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            rank = 7
            U = [[0]*rank for _ in range(n1*m1)]  # 4 rows
            V = [[0]*rank for _ in range(m1*p1)]  # 4 rows
            W = [[0]*rank for _ in range(p1*n1)]  # 4 rows

            # A indices: A11=(0,0)->U[0], A12=(0,1)->U[1], A21=(1,0)->U[2], A22=(1,1)->U[3]
            # B indices: B11=(0,0)->V[0], B12=(0,1)->V[1], B21=(1,0)->V[2], B22=(1,1)->V[3]
            # C indices: C11=(0,0)->W[0], C12=(0,1)->W[1], C21=(1,0)->W[2], C22=(1,1)->W[3]
            # W convention: W[k*n+i, r] so C[i,k] is at W[k*2+i]
            # C11: i=0,k=0 -> W[0*2+0]=W[0]
            # C12: i=0,k=1 -> W[1*2+0]=W[2]
            # C21: i=1,k=0 -> W[0*2+1]=W[1]
            # C22: i=1,k=1 -> W[1*2+1]=W[3]

            # M1: (A11+A22)(B11+B22)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1
            # C11 += M1, C12 += M1, C21 += M1, C22 += M1

            # M2: (A21+A22)B11
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[1][1] = 1; W[3][1] = -1
            # C21 += M2, C22 -= M2

            # M3: A11(B21-B22)
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[2][2] = 1; W[3][2] = 1
            # C12 += M3, C22 += M3

            # M4: A22(B12-B11)
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[1][3] = 1
            # C11 += M4, C21 += M4

            # M5: (A11+A12)B22
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[2][4] = 1
            # C11 -= M5, C12 += M5

            # M6: (A21-A11)(B11+B12)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            # C22 += M6

            # M7: (A12-A22)(B21+B22)
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1
            # C11 += M7

            return U, V, W, rank
        else:
            U, V, W = make_schoolbook(n1, m1, p1)
            return U, V, W, n1 * m1 * p1

    def combine_blocks(blocks):
        """
        Combine a list of (U, V, W, rank) blocks into a single decomposition.
        Each block has local indices; we need to map to global.
        blocks is a list of dicts: {'U':..., 'V':..., 'W':..., 'rank':..., 'off_i':..., 'off_j':..., 'off_k':...}
        """
        total_rank = sum(b['rank'] for b in blocks)
        # Global sizes
        gm = m  # middle dimension
        gU_rows = n * m
        gV_rows = m * p
        gW_rows = p * n

        U_global = [[0] * total_rank for _ in range(gU_rows)]
        V_global = [[0] * total_rank for _ in range(gV_rows)]
        W_global = [[0] * total_rank for _ in range(gW_rows)]

        col_offset = 0
        for b in blocks:
            r = b['rank']
            off_i = b['off_i']
            off_j = b['off_j']
            off_k = b['off_k']
            bn, bm, bp = b['bn'], b['bm'], b['bp']

            for li in range(bn):
                for lj in range(bm):
                    gi = off_i + li
                    gj = off_j + lj
                    for c in range(r):
                        U_global[gi * m + gj][col_offset + c] = b['U'][li * bm + lj][c]

            for lj in range(bm):
                for lk in range(bp):
                    gj = off_j + lj
                    gk = off_k + lk
                    for c in range(r):
                        V_global[gj * p + gk][col_offset + c] = b['V'][lj * bp + lk][c]

            for lk in range(bp):
                for li in range(bn):
                    gk = off_k + lk
                    gi = off_i + li
                    for c in range(r):
                        W_global[gk * n + gi][col_offset + c] = b['W'][lk * bn + li][c]

            col_offset += r

        return U_global, V_global, W_global, total_rank

    # Strategy: Try various block partitions
    # n=3, m=5, p=7
    # We can split into Strassen 2x2 blocks + schoolbook strips

    best_U, best_V, best_W = make_schoolbook(n, m, p)
    best_rank = n * m * p

    def try_partition(split_n, split_m, split_p):
        """
        Try to partition the problem into blocks.
        split_n: list of sizes summing to n
        split_m: list of sizes summing to m
        split_p: list of sizes summing to p
        Each block (ni, mj, pk) uses Strassen if 2x2x2 else schoolbook.
        """
        blocks = []
        ci = 0
        for ni in split_n:
            cj = 0
            for mj in split_m:
                ck = 0
                for pk in split_p:
                    U_b, V_b, W_b, r = strassen_2x2(ni, mj, pk, ci, cj, ck)
                    blocks.append({
                        'U': U_b, 'V': V_b, 'W': W_b, 'rank': r,
                        'off_i': ci, 'off_j': cj, 'off_k': ck,
                        'bn': ni, 'bm': mj, 'bp': pk
                    })
                    ck += pk
                cj += mj
            ci += ni

        U_g, V_g, W_g, total_rank = combine_blocks(blocks)

        # Verify
        if verify(U_g, V_g, W_g, total_rank, n, m, p):
            return U_g, V_g, W_g, total_rank
        return None

    # Try various partitions
    # n=3: [3], [1,2], [2,1], [1,1,1]
    # m=5: [5], [1,4], [2,3], [3,2], [4,1], [1,1,3], [1,2,2], [2,1,2], [2,2,1], [3,1,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]
    # p=7: [7], [1,6], [2,5], [3,4], [4,3], [5,2], [6,1], etc.

    # Focus on partitions that maximize 2x2x2 Strassen blocks
    # Best case: use as many 2x2x2 blocks as possible

    # Partition n=3 as [1,2], m=5 as [1,2,2], p=7 as [1,2,2,2]
    # This gives blocks:
    # (1,1,1) sb, (1,1,2) sb, (1,1,2) sb, (1,1,2) sb
    # (2,1,1) sb, (2,1,2) sb, (2,1,2) sb, (2,1,2) sb
    # (2,2,1) sb, (2,2,2) STR, (2,2,2) STR, (2,2,2) STR
    # That's 3 Strassen blocks!

    partitions_to_try = [
        # (split_n, split_m, split_p)
        ([1, 2], [1, 2, 2], [1, 2, 2, 2]),  # 3 Strassen blocks
        ([2, 1], [2, 2, 1], [2, 2, 2, 1]),  # 3 Strassen blocks
        ([1, 2], [2, 2, 1], [2, 2, 2, 1]),  # 2 Strassen blocks
        ([2, 1], [1, 2, 2], [1, 2, 2, 2]),  # 2 Strassen blocks
        ([1, 2], [2, 1, 2], [2, 2, 1, 2]),  # 2 Strassen blocks
        ([2, 1], [2, 1, 2], [2, 2, 1, 2]),  # 2 Strassen blocks
        ([1, 2], [1, 1, 3], [1, 1, 5]),     # 0 Strassen (3 in m)
        ([2, 1], [3, 2], [5, 2]),            # 1 Strassen block
        ([1, 2], [3, 2], [5, 2]),            # 1 Strassen block
        ([2, 1], [2, 3], [2, 5]),            # 1 Strassen block
        ([1, 2], [2, 3], [2, 5]),            # 1 Strassen block
        ([2, 1], [3, 2], [2, 5]),            # 1 Strassen block
        ([1, 2], [3, 1, 1], [3, 2, 2]),      # 1 Strassen block
        ([2, 1], [1, 1, 3], [2, 2, 3]),      # 1 Strassen block
        ([1, 1, 1], [2, 2, 1], [2, 2, 2, 1]), # 3 Strassen blocks (in m=2,2 part)
        ([1, 1, 1], [1, 2, 2], [1, 2, 2, 2]), # 3 Strassen blocks
    ]

    for split_n, split_m, split_p in partitions_to_try:
        if time.time() > deadline - 20:
            break
        result = try_partition(split_n, split_m, split_p)
        if result is not None:
            U_g, V_g, W_g, total_rank = result
            if total_rank < best_rank:
                best_U, best_V, best_W = U_g, V_g, W_g
                best_rank = total_rank

    # Also try pure schoolbook as baseline (already set)
    # Try a few more creative partitions
    # n=3 as [2,1], m=5 as [2,2,1], p=7 as [2,2,2,1]
    extra_partitions = [
        ([2, 1], [2, 2, 1], [2, 2, 2, 1]),
        ([1, 2], [2, 2, 1], [2, 2, 2, 1]),
        ([2, 1], [1, 2, 2], [2, 2, 2, 1]),
        ([1, 2], [1, 2, 2], [2, 2, 2, 1]),
        ([2, 1], [2, 1, 2], [2, 2, 2, 1]),
        ([1, 2], [2, 1, 2], [2, 2, 2, 1]),
    ]

    for split_n, split_m, split_p in extra_partitions:
        if time.time() > deadline - 10:
            break
        result = try_partition(split_n, split_m, split_p)
        if result is not None:
            U_g, V_g, W_g, total_rank = result
            if total_rank < best_rank:
                best_U, best_V, best_W = U_g, V_g, W_g
                best_rank = total_rank

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
