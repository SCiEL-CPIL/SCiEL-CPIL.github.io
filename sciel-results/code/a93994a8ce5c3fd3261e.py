import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank
    # 6 blocks of 2x2x2 (Strassen rank 7)
    # 3 blocks of 2x1x2 (Schoolbook rank 4)
    rank = 6 * 7 + 3 * 4
    # rank = 54
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        """
        Add Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        """
        nonlocal col
        # Local indices: i in 0..1, j in 0..1, k in 0..1
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the coefficients for A, B, and C for each term
        # term: (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {(i,j): coeff}
        # B_coeffs: dict {(j,k): coeff}
        # C_coeffs: dict {(i,k): coeff}
        
        terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for A_c, B_c, C_c in terms:
            for (li, lj), val in A_c.items():
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj, col] += val
            
            for (lj, lk), val in B_c.items():
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk, col] += val
            
            for (li, lk), val in C_c.items():
                gi = i_off + li
                gk = k_off + lk
                # W index: k * n + i
                W[gk * n + gi, col] += val
            
            col += 1

    def add_212(i_off, j_off, k_off):
        """
        Add schoolbook decomposition for A[i_off:i_off+2, j_off:j_off+1] * B[j_off:j_off+1, k_off:k_off+2]
        """
        nonlocal col
        # A is 2x1, B is 1x2. C is 2x2.
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        # Local j is always 0.
        
        # Term 1: A0 B0 -> C00
        U[i_off * m + (j_off + 0), col] = 1.0
        V[(j_off + 0) * p + (k_off + 0), col] = 1.0
        W[(k_off + 0) * n + (i_off + 0), col] = 1.0
        col += 1
        
        # Term 2: A0 B1 -> C01
        U[i_off * m + (j_off + 0), col] = 1.0
        V[(j_off + 0) * p + (k_off + 1), col] = 1.0
        W[(k_off + 1) * n + (i_off + 0), col] = 1.0
        col += 1
        
        # Term 3: A1 B0 -> C10
        U[(i_off + 1) * m + (j_off + 0), col] = 1.0
        V[(j_off + 0) * p + (k_off + 0), col] = 1.0
        W[(k_off + 0) * n + (i_off + 1), col] = 1.0
        col += 1
        
        # Term 4: A1 B1 -> C11
        U[(i_off + 1) * m + (j_off + 0), col] = 1.0
        V[(j_off + 0) * p + (k_off + 1), col] = 1.0
        W[(k_off + 1) * n + (i_off + 1), col] = 1.0
        col += 1

    # Block 1: A1 (2x2, i=0,1; j=0,1) with B1 (2x2, j=0,1; k=0,1)
    add_strassen_222(0, 0, 0)
    # Block 2: A1 with B2 (2x2, j=0,1; k=2,3)
    add_strassen_222(0, 0, 2)
    # Block 3: A1 with B3 (2x2, j=0,1; k=4,5)
    add_strassen_222(0, 0, 4)
    
    # Block 4: A2 (2x2, i=0,1; j=2,3) with B4 (2x2, j=2,3; k=0,1)
    add_strassen_222(0, 2, 0)
    # Block 5: A2 with B5 (2x2, j=2,3; k=2,3)
    add_strassen_222(0, 2, 2)
    # Block 6: A2 with B6 (2x2, j=2,3; k=4,5)
    add_strassen_222(0, 2, 4)
    
    # Block 7: A3 (2x1, i=0,1; j=4) with B7 (1x2, j=4; k=0,1)
    add_212(0, 4, 0)
    # Block 8: A3 with B8 (1x2, j=4; k=2,3)
    add_212(0, 4, 2)
    # Block 9: A3 with B9 (1x2, j=4; k=4,5)
    add_212(0, 4, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
