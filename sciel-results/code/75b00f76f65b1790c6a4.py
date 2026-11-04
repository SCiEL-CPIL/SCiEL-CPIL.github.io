"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 10

    # Strategy: Strassen on 2x2 blocks for the 2x2 part of A (rows 0,1) x even columns of B,
    # plus schoolbook for the 2x1 part of A (row 2) x all columns of B.
    # Additionally, try to merge terms between adjacent Strassen blocks to reduce rank.
    
    # The key insight: in the Strassen decomposition, some intermediate products involve
    # linear combinations that can potentially be shared across adjacent blocks if the
    # A-entries are the same (which they are: a, b, c, d are fixed for all blocks).
    # However, the B-entries differ per block, so direct sharing doesn't work.
    
    # Alternative approach: use a rank-13 decomposition for 2x2 x 2x2 if available,
    # or exploit the specific structure more carefully.
    
    # For now, let's try a different partitioning:
    # Instead of 5 independent Strassen blocks, consider the full 2x10 multiplication
    # as (2x3) x (3x10) and look for cross-term cancellations.
    
    # Let's construct a potentially lower-rank decomposition by combining:
    # 1. Strassen for the 2x2 x 2x10 part (rows 0,1 of A, all 10 columns of B)
    # 2. Schoolbook for the 1x2 x 2x10 part (row 2 of A, all 10 columns of B)
    
    # For the 2x2 x 2x10 part, we can use a rank-7-per-2-columns approach:
    # Group the 10 columns into 5 pairs, apply Strassen to each pair.
    # This gives 5 * 7 = 35 terms for the 2x2 x 2x10 part.
    
    # For the 1x2 x 2x10 part: 2 * 10 = 20 terms.
    # Total: 35 + 20 = 55.
    
    # To beat 55, we need to find cross-block cancellations.
    
    # Let's try a more aggressive approach: use a bilinear algorithm that exploits
    # the full 2x3 x 3x10 structure.
    
    # The tensor M_{2,3,10} has rank at most 2*3*10 = 60 (schoolbook).
    # Known bounds: for M_{m,n,p}, the rank is between max(m,n,p) and m*n*p.
    # For small cases, exact ranks are known.
    
    # Let me try to find a rank-50 decomposition by carefully designing the
    # linear combinations.
    
    # Approach: Use Strassen-like ideas but with overlapping column pairs.
    
    # Actually, let's try a completely different strategy:
    # Treat the 3x10 matrix B as having structure that can be exploited.
    
    # For a 2x3 x 3x10 multiplication, we can view it as:
    # C = A * B where A is 2x3 and B is 3x10.
    
    # One approach: decompose B into a sum of rank-1 matrices (not helpful directly).
    
    # Let's try the following: use a rank-15 decomposition for the 2x2 x 2x3 part
    # (first 2 rows of A, first 3 columns of B... wait, B is 3x10).
    
    # I think the best bet is to try to find a slightly lower rank by careful
    # optimization. Let's start with the 55-rank solution and see if we can
    # reduce it by 1-2 terms through algebraic manipulation.
    
    # For now, return the known 55-rank solution as a baseline, but with
    # a slightly different implementation that might be more efficient.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55

    r = 0

    # Strassen 7 products for 2x2 x 2x2:
    strassen_u = [
        [(0, 1), (4, 1)],      # m1: a+d
        [(3, 1), (4, 1)],      # m2: c+d
        [(0, 1)],              # m3: a
        [(4, 1)],              # m4: d
        [(0, 1), (1, 1)],      # m5: a+b
        [(3, 1), (0, -1)],     # m6: c-a
        [(1, 1), (4, -1)],     # m7: b-d
    ]

    strassen_v = [
        [(0, 1), (11, 1)],     # m1: e+h
        [(0, 1)],              # m2: e
        [(1, 1), (11, -1)],    # m3: f-h
        [(10, 1), (0, -1)],    # m4: g-e
        [(11, 1)],             # m5: h
        [(0, 1), (1, 1)],      # m6: e+f
        [(10, 1), (11, 1)],    # m7: g+h
    ]

    strassen_w = [
        [(0, 1), (3, 1)],      # m1: C00 + C11
        [(1, 1), (3, -1)],     # m2: C10 - C11
        [(2, 1), (3, 1)],      # m3: C01 + C11
        [(0, 1), (1, 1)],      # m4: C00 + C10
        [(0, -1), (2, 1)],     # m5: -C00 + C01
        [(3, 1)],              # m6: C11
        [(0, 1)],              # m7: C00
    ]

    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1

        for s in range(7):
            for (u_off, coeff) in strassen_u[s]:
                U[u_off][r] += coeff

            for (v_off, coeff) in strassen_v[s]:
                if v_off == 0:
                    v_row = col0
                elif v_off == 1:
                    v_row = col1
                elif v_off == 10:
                    v_row = 10 + col0
                elif v_off == 11:
                    v_row = 10 + col1
                else:
                    raise ValueError(f"Bad v_off {v_off}")
                V[v_row][r] += coeff

            for (w_off, coeff) in strassen_w[s]:
                if w_off == 0:
                    w_row = col0 * 2 + 0
                elif w_off == 1:
                    w_row = col0 * 2 + 1
                elif w_off == 2:
                    w_row = col1 * 2 + 0
                elif w_off == 3:
                    w_row = col1 * 2 + 1
                else:
                    raise ValueError(f"Bad w_off {w_off}")
                W[w_row][r] += coeff

            r += 1

    # Schoolbook for A_right (2x1) @ B_bottom (1x10): 2*10 = 20 terms
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1

    # Validate
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank-60
        rank60 = 60
        U2 = [[0] * rank60 for _ in range(n * m)]
        V2 = [[0] * rank60 for _ in range(m * p)]
        W2 = [[0] * rank60 for _ in range(p * n)]
        for rr in range(rank60):
            ii = rr // (m * p)
            rem = rr % (m * p)
            jj = rem // p
            kk = rem % p
            U2[ii * m + jj][rr] = 1
            V2[jj * p + kk][rr] = 1
            W2[kk * n + ii][rr] = 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
