import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0  # Safety margin before 120s limit

    # ---------- Verify a candidate decomposition ----------
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
                        u_val = U[a_idx][r]
                        v_val = V[b_idx][r]
                        w_val = W[c_idx][r]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            s += u_val * v_val * w_val
                    if s != 1:
                        return False
        return True

    # ---------- Rank-32 decomposition ----------
    # Structure:
    #   j in {0,1} and k in {0,1}: Strassen 2x2x2, rank 7
    #   j in {2,3} and k in {0,1}: Strassen 2x2x2, rank 7
    #   j in {0,1} and k in {2,3,4}: custom 2x2x3, rank 9
    #   j in {2,3} and k in {2,3,4}: custom 2x2x3, rank 9
    #   Total: 7 + 7 + 9 + 9 = 32
    #
    # For 2x2x3 (A: 2x2, B: 2x3, C: 2x3), we use rank 9:
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # We can factor this as:
    #   Let P0 = A[0,0], P1 = A[0,1], P2 = A[1,0], P3 = A[1,1]
    #   Let Q0 = B[0,0], Q1 = B[0,1], Q2 = B[0,2], Q3 = B[1,0], Q4 = B[1,1], Q5 = B[1,2]
    #   C[0,0] = P0*Q0 + P1*Q3
    #   C[0,1] = P0*Q1 + P1*Q4
    #   C[0,2] = P0*Q2 + P1*Q5
    #   C[1,0] = P2*Q0 + P3*Q3
    #   C[1,1] = P2*Q1 + P3*Q4
    #   C[1,2] = P2*Q2 + P3*Q5
    #
    # This is 12 terms schoolbook. To get rank 9, we need a smarter combination.
    # Actually, let's just use schoolbook rank 12 for each 2x2x3 block.
    # Total: 7 + 7 + 12 + 12 = 38. That's worse than 36.
    #
    # Alternative: Use the full Strassen block decomposition (rank 36) as before,
    # but try to find a rank-35 or lower solution.
    #
    # Let me try: split k into {0,1,2,3,4} differently.
    # What if we use Strassen for 2x2x2 blocks covering k={0,1} and k={2,3},
    # and schoolbook for k={4}?
    #   j in {0,1}, k in {0,1}: Strassen, rank 7
    #   j in {2,3}, k in {0,1}: Strassen, rank 7
    #   j in {0,1}, k in {2,3}: Strassen, rank 7
    #   j in {2,3}, k in {2,3}: Strassen, rank 7
    #   j in {0,1}, k in {4}: schoolbook 2x2x1, rank 4
    #   j in {2,3}, k in {4}: schoolbook 2x2x1, rank 4
    #   Total: 7+7+7+7+4+4 = 36. Same as before.
    #
    # What if we merge the two k={4} blocks?
    #   j in {0,1,2,3}, k in {4}: 2x4x1, rank 8 (schoolbook)
    #   Total: 7+7+7+7+8 = 36. Still 36.
    #
    # Let me try a completely different approach: use the 4x4 Strassen-like
    # decomposition for the first 4 columns, then handle the 5th column.
    #
    # Actually, the current best is 36. Let me try to find a rank-35 solution
    # by finding one term that can be shared between sub-problems.
    #
    # Given the complexity, let me implement the rank-36 solution (which we know
    # works) and also attempt a rank-32 solution with a verified 2x2x3 rank-5
    # decomposition if it exists.
    #
    # For now, return the verified rank-36 solution.
    
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

    # ---------- Attempt rank-32 via 2x2x3 rank-9 decomposition ----------
    def rank32_solution():
        """
        Try to achieve rank 32 by using:
        - Two Strassen 2x2x2 blocks (rank 7 each) for k in {0,1}
        - Two 2x2x3 blocks (rank 9 each) for k in {2,3,4}
        Total: 7+7+9+9 = 32
        
        For 2x2x3 with rank 9, we need a decomposition of:
        T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        using 9 rank-1 tensors.
        """
        rank = 32
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

        strassen_2x2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        # 2x2x3 rank-9 decomposition:
        # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
        # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
        # 
        # Using 9 products:
        # P1 = a*e, P2 = b*h, P3 = a*f, P4 = b*i, P5 = a*g, P6 = b*j
        # P7 = c*e, P8 = d*h, P9 = (c+d)*(f+i) ... this doesn't work cleanly.
        #
        # Actually, let's use schoolbook rank 12 for 2x2x3.
        # Total: 7+7+12+12 = 38. Not better than 36.
        #
        # So this approach doesn't help. Let's just use the rank-36 solution.
        return None

    # ---------- Pure schoolbook fallback (rank 40) ----------
    def schoolbook_solution():
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

    # ---------- Try rank-36 solution first ----------
    best = strassen_block_solution()
    if verify(best["U"], best["V"], best["W"]):
        return best

    # ---------- Fallback to schoolbook ----------
    return schoolbook_solution()
# EVOLVE-BLOCK-END
