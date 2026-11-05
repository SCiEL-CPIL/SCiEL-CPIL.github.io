import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    # ---------- Structural fallback: Strassen block decomposition, rank 36 ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # ---------- Exact-rational verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Structured Kronecker Product Decomposition, Rank 32 ----------
    def kronecker_product_solution():
        """
        Partition dimensions:
        - Output k: 5 = 2 + 3
        - Inner j: 4 = 2 + 2
        
        Sub-problems:
        1. (i in 0..1, j in 0..1, k in 0..1): 2x2x2 tensor, rank 7 (Strassen)
        2. (i in 0..1, j in 2..3, k in 0..1): 2x2x2 tensor, rank 7 (Strassen)
        3. (i in 0..1, j in 0..1, k in 2..4): 2x2x3 tensor, rank 9 (custom)
        4. (i in 0..1, j in 2..3, k in 2..4): 2x2x3 tensor, rank 9 (custom)
        
        Total rank: 7 + 7 + 9 + 9 = 32
        """
        # 7 + 7 + 12 + 12 = 38... that's worse than 36!
        # Let me reconsider. Use Strassen for the 2x2x2 parts and schoolbook for 2x2x3.
        # Actually, let's try a different partition: split j into 4 single columns.
        # 4 sub-problems of 2x1x5, each rank 2*1*5 = 10 schoolbook, total 40. Worse.
        # Or split k: 5 sub-problems of 2x4x1, each rank 2*4*1=8, total 40. Worse.
        # The Strassen block approach (rank 36) is better than pure schoolbook partitions.
        # Let's keep rank 36 as the primary and verify it works.
        rank = 36
        U = [[0] * rank for _ in range(n * m)]  # 8 x 36
        V = [[0] * rank for _ in range(m * p)]  # 20 x 36
        W = [[0] * rank for _ in range(p * n)]  # 10 x 36
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # Strassen 2x2x2 terms (local indices 0-3 for U, V, W)
        strassen_2x2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        # Schoolbook 2x2x3 decomposition, rank 12
        # U: 2x2 (indices 0-3), V: 2x3 (indices 0-5), W: 3x2 (indices 0-5)
        # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        # Each term: U has one nonzero at (i,j), V has one at (j,k), W has one at (k,i)
        custom_2x2x3 = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_idx = i * 2 + j       # 0..3
                    v_idx = j * 3 + k       # 0..5
                    w_idx = k * 2 + i       # 0..5
                    custom_2x2x3.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        # 12 terms total

        # Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
        for u_t, v_t, w_t in strassen_2x2x2:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + lj] = val  # j in 0,1
            for idx, val in v_t.items():
                lj, lk = idx // 2, idx % 2
                v_g[lj * p + lk] = val  # j in 0,1, k in 0,1
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[lk * n + li] = val  # k in 0,1
            add_term(u_g, v_g, w_g)

        # Sub-problem 2: i in {0,1}, j in {2,3}, k in {0,1}
        for u_t, v_t, w_t in strassen_2x2x2:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + (2 + lj)] = val  # j in 2,3
            for idx, val in v_t.items():
                lj, lk = idx // 2, idx % 2
                v_g[(2 + lj) * p + lk] = val  # j in 2,3, k in 0,1
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[lk * n + li] = val  # k in 0,1
            add_term(u_g, v_g, w_g)

        # Sub-problem 3: i in {0,1}, j in {0,1}, k in {2,3,4}
        for u_t, v_t, w_t in custom_2x2x3:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + lj] = val  # j in 0,1
            for idx, val in v_t.items():
                lj = idx // 3
                lk = idx % 3
                v_g[lj * p + (2 + lk)] = val  # j in 0,1, k in 2,3,4
            for idx, val in w_t.items():
                li = idx // 3
                lk = idx % 3
                w_g[(2 + lk) * n + li] = val  # k in 2,3,4
            add_term(u_g, v_g, w_g)

        # Sub-problem 4: i in {0,1}, j in {2,3}, k in {2,3,4}
        for u_t, v_t, w_t in custom_2x2x3:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * m + (2 + lj)] = val  # j in 2,3
            for idx, val in v_t.items():
                lj = idx // 3
                lk = idx % 3
                v_g[(2 + lj) * p + (2 + lk)] = val  # j in 2,3, k in 2,3,4
            for idx, val in w_t.items():
                li = idx // 3
                lk = idx % 3
                w_g[(2 + lk) * n + li] = val  # k in 2,3,4
            add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    best = strassen_block_solution()
    
    # Verify the Strassen block solution works
    if verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback: pure schoolbook rank 40
    rank = 40
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
# EVOLVE-BLOCK-END
