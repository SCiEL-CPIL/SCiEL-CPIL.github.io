import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Split the 4 middle indices into two groups of 2.
    # Group 1: j in {0,1} -> A[:, 0:2] (2x2) * B[0:2, :] (2x8) -> C1 (2x8)
    # Group 2: j in {2,3} -> A[:, 2:4] (2x2) * B[2:4, :] (2x8) -> C2 (2x8)
    # C = C1 + C2
    #
    # Each M_{2,2,8} subproblem is decomposed into 4 blocks of M_{2,2,2}.
    # Each M_{2,2,2} block uses Strassen's algorithm with rank 7.
    # Total rank = 2 * 4 * 7 = 56.
    
    U, V, W = _build_rank_56(n, m, p)
    return {"U": U, "V": V, "W": W}

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    8 blocks of 2x2x2, each with rank 7 (Strassen).
    Total rank = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Define Strassen terms locally for 2x2 multiplication
    # Local indices: A[r,c] -> r*2+c, B[r,c] -> r*2+c, C[r,c] -> r*2+c
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs) where each is dict local_idx -> coeff
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})             # M7
    ]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A indices (2x4 matrix, row-major: index = i*m + j = i*4 + j)
            # A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            # B indices (4x8 matrix, row-major: index = j*p + k = j*8 + k)
            # B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b11_idx = (2 * u) * 8 + 2 * v
            b12_idx = (2 * u) * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            # W indices (output convention W[k*n+i] = W[2k+i])
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            c11_idx = (2 * v) * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = (2 * v) * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # Apply Strassen terms
            for s, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_terms):
                col = base_col + s
                
                # Map local A indices to global
                # Local 0->a11, 1->a12, 2->a21, 3->a22
                a_map = {0: a11_idx, 1: a12_idx, 2: a21_idx, 3: a22_idx}
                for local_idx, coeff in a_coeffs.items():
                    U[a_map[local_idx]][col] += coeff
                
                # Map local B indices to global
                # Local 0->b11, 1->b12, 2->b21, 3->b22
                b_map = {0: b11_idx, 1: b12_idx, 2: b21_idx, 3: b22_idx}
                for local_idx, coeff in b_coeffs.items():
                    V[b_map[local_idx]][col] += coeff
                
                # Map local C indices to global
                # Local 0->c11, 1->c12, 2->c21, 3->c22
                c_map = {0: c11_idx, 1: c12_idx, 2: c21_idx, 3: c22_idx}
                for local_idx, coeff in c_coeffs.items():
                    W[c_map[local_idx]][col] += coeff
            
    return U, V, W
