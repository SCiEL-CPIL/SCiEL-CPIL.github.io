DIMENSIONS = [2, 2, 12]

import numpy as np
import time

def construct():
    """
    Rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    Uses Strassen's algorithm on 6 independent 2x2x2 column-pair blocks.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j for A[i][j]
      V[j*p+k, r]: row = j*p+k for B[j][k]
      W[k*n+i, r]: row = k*n+i for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2    # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B11-B01)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    strassen_products = [
        # P1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # P2
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        # P3
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        # P4
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        # P5
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # P6
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        # P7
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],      # P1: C00 += 1, C11 += 1
        [((1,0), 1), ((1,1), -1)],     # P2: C10 += 1, C11 -= 1
        [((0,1), 1), ((1,1), 1)],      # P3: C01 += 1, C11 += 1
        [((0,0), 1), ((1,0), 1)],      # P4: C00 += 1, C10 += 1
        [((0,0), -1), ((0,1), 1)],     # P5: C00 -= 1, C01 += 1
        [((1,1), 1)],                   # P6: C11 += 1
        [((0,0), 1)],                   # P7: C00 += 1
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in strassen_products[s][0]:
                U[ai * m + aj, col] = coeff
            
            for (bj, bk_rel), coeff in strassen_products[s][1]:
                V[bj * p + base_col + bk_rel, col] = coeff
            
            for (ci, ck_rel), coeff in output_contribs[s]:
                W[(base_col + ck_rel) * n + ci, col] = coeff
            
            col += 1
    
    # Verify decomposition correctness
    if time.time() < deadline - 5:
        # Check tensor identity: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Build expected tensor
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i, j, k] = 1.0
        
        # Compute from decomposition
        T_computed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] = np.sum(
                        U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i]
                    )
        
        if not np.allclose(T_computed, T_expected, atol=1e-10):
            # Fallback: return schoolbook rank-48 if Strassen verification fails
            rank_fb = n * m * p  # 48
            U_fb = np.zeros((n * m, rank_fb), dtype=float)
            V_fb = np.zeros((m * p, rank_fb), dtype=float)
            W_fb = np.zeros((p * n, rank_fb), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U_fb[i * m + j, r] = 1.0
                        V_fb[j * p + k, r] = 1.0
                        W_fb[k * n + i, r] = 1.0
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
