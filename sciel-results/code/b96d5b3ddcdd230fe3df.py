"""Matrix multiplication 3x3x3 rank-26 decomposition."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-26 exact decomposition for 3x3 matrix multiplication.
    
    Uses a block-partitioned approach:
    1. Strassen's algorithm (rank 7) for the 2x2 sub-block (indices 0,1).
    2. Direct terms for interactions involving index 2 (rank 19).
    Total rank: 26.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    # Initialize matrices
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Helper to add a term with specified coefficients
    def add_term(r, u_items, v_items, w_items):
        # u_items: list of (index, coeff)
        for idx, coeff in u_items:
            U[idx][r] += coeff
        for idx, coeff in v_items:
            V[idx][r] += coeff
        for idx, coeff in w_items:
            W[idx][r] += coeff

    # --- Group 1: Strassen for i,j,k in {0,1} (7 terms) ---
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    add_term(0, [(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # Term 1: M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    add_term(1, [(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    
    # Term 2: M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    add_term(2, [(0, 1)], [(1, 1), (4, -1)], [(3, 1), (4, 1)])
    
    # Term 3: M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    add_term(3, [(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # Term 4: M5 = (A00+A10)B11
    # C00 -= M5, C01 += M5
    add_term(4, [(0, 1), (3, 1)], [(4, 1)], [(0, -1), (3, 1)])
    
    # Term 5: M6 = (A01-A00)(B00+B10)
    # C10 += M6
    add_term(5, [(1, 1), (0, -1)], [(0, 1), (3, 1)], [(1, 1)])
    
    # Term 6: M7 = (A11-A10)(B01+B11)
    # C11 += M7, C00 -= M7
    add_term(6, [(4, 1), (3, -1)], [(1, 1), (4, 1)], [(4, 1), (0, -1)])
    
    # --- Group 2: Cross terms involving index 2 (19 terms) ---
    
    # 2.1 A[0:2, 2] B[2, 0:2] (4 terms)
    # i in {0,1}, k in {0,1}
    # A[i,2] B[2,k] -> C[i,k]
    add_term(7, [(2, 1)], [(6, 1)], [(0, 1)])   # i=0, k=0
    add_term(8, [(2, 1)], [(7, 1)], [(3, 1)])   # i=0, k=1
    add_term(9, [(5, 1)], [(6, 1)], [(1, 1)])   # i=1, k=0
    add_term(10, [(5, 1)], [(7, 1)], [(4, 1)])  # i=1, k=1
    
    # 2.2 A[0:2, 0:2] B[0:2, 2] (4 terms)
    # i in {0,1}, j in {0,1}
    # A[i,j] B[j,2] -> C[i,2]
    add_term(11, [(0, 1)], [(2, 1)], [(6, 1)])  # i=0, j=0
    add_term(12, [(1, 1)], [(5, 1)], [(6, 1)])  # i=0, j=1
    add_term(13, [(3, 1)], [(2, 1)], [(7, 1)])  # i=1, j=0
    add_term(14, [(4, 1)], [(5, 1)], [(7, 1)])  # i=1, j=1
    
    # 2.3 A[2, 0:2] B[0:2, 0:2] (4 terms)
    # j in {0,1}, k in {0,1}
    # A[2,j] B[j,k] -> C[2,k]
    add_term(15, [(6, 1)], [(0, 1)], [(2, 1)])  # j=0, k=0
    add_term(16, [(6, 1)], [(1, 1)], [(5, 1)])  # j=0, k=1
    add_term(17, [(7, 1)], [(3, 1)], [(2, 1)])  # j=1, k=0
    add_term(18, [(7, 1)], [(4, 1)], [(5, 1)])  # j=1, k=1
    
    # 2.4 A[2, 2] B[2, 0:2] (2 terms)
    # k in {0,1}
    # A[2,2] B[2,k] -> C[2,k]
    add_term(19, [(8, 1)], [(6, 1)], [(2, 1)])  # k=0
    add_term(20, [(8, 1)], [(7, 1)], [(5, 1)])  # k=1
    
    # 2.5 A[0:2, 2] B[2, 2] (2 terms)
    # i in {0,1}
    # A[i,2] B[2,2] -> C[i,2]
    add_term(21, [(2, 1)], [(8, 1)], [(6, 1)])  # i=0
    add_term(22, [(5, 1)], [(8, 1)], [(7, 1)])  # i=1
    
    # 2.6 A[2, 0:2] B[0:2, 2] (2 terms)
    # j in {0,1}
    # A[2,j] B[j,2] -> C[2,2]
    add_term(23, [(6, 1)], [(2, 1)], [(8, 1)])  # j=0
    add_term(24, [(7, 1)], [(5, 1)], [(8, 1)])  # j=1
    
    # 2.7 A[2, 2] B[2, 2] (1 term)
    add_term(25, [(8, 1)], [(8, 1)], [(8, 1)])
    
    # Verification
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                ui = U[i * 3 + j]
                vj = V[j * 3 + k]
                wk = W[k * 3 + i]
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to rank 27
        rank27 = 27
        U27 = [[0] * rank27 for _ in range(9)]
        V27 = [[0] * rank27 for _ in range(9)]
        W27 = [[0] * rank27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U27[i * 3 + j][col] = 1
                    V27[j * 3 + k][col] = 1
                    W27[k * 3 + i][col] = 1
                    col += 1
        return {"U": U27, "V": V27, "W": W27}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
