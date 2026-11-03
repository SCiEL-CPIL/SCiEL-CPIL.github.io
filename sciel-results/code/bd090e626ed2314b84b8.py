"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target rank 78
    rank = 78
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a schoolbook block
    # A is n_a x m_a, B is m_a x p_a, C is n_a x p_a
    # Offsets: a0 (row in A), b0 (col in A / row in B), c0 (col in B / row in C)
    def add_schoolbook(n_a, m_a, p_a, a0, b0, c0):
        nonlocal col
        for i in range(n_a):
            for j in range(m_a):
                for k in range(p_a):
                    U[(a0 + i) * m + (b0 + j), col] = 1
                    V[(b0 + j) * p + (c0 + k), col] = 1
                    W[(c0 + k) * n + (a0 + i), col] = 1
                    col += 1

    # Helper to add a Strassen 2x2x2 block
    def add_strassen(a0, b0, c0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[a0, b0], A22 = A[a0+1, b0+1]
        # B11 = B[b0, c0], B22 = B[b0+1, c0+1]
        
        # M1
        U[a0 * m + b0, col] = 1
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[c0 * n + a0, col] = 1       # C11 +=
        W[c0 * n + (a0+1), col] = 1   # C12 += (Wait, C indices are k, i. C[k, i]. C11 is C[c0, a0]. C12 is C[c0, a0+1]? No.
        # C is p x n. C[k, i].
        # Block C is at rows c0..c0+1, cols a0..a0+1.
        # C11 is C[c0, a0]. C12 is C[c0, a0+1]? No, C12 is C[c0, a0+1] is wrong.
        # C[i, k] in math, but W[k*n+i].
        # So C11 corresponds to W[c0, a0].
        # C12 corresponds to W[c0, a0+1]? No.
        # C is 2x2 block.
        # C[0,0] -> W[c0, a0]
        # C[0,1] -> W[c0, a0+1]
        # C[1,0] -> W[c0+1, a0]
        # C[1,1] -> W[c0+1, a0+1]
        
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # C11 += M1
        # C12 += M1
        # C21 += M1
        # C22 += M1
        
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's implement this carefully.
        
        # M1
        U[a0 * m + b0, col] = 1
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[c0 * n + a0, col] = 1
        W[c0 * n + (a0+1), col] = 1
        W[(c0+1) * n + a0, col] = 1
        W[(c0+1) * n + (a0+1), col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(a0+1) * m + b0, col] = 1
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        W[(c0+1) * n + a0, col] = 1
        W[(c0+1) * n + (a0+1), col] = 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[a0 * m + b0, col] = 1
        V[(b0+1) * p + c0, col] = 1
        V[(b0+1) * p + (c0+1), col] = -1
        W[c0 * n + (a0+1), col] = 1
        W[(c0+1) * n + (a0+1), col] = 1
        col += 1
        
        # M4 = A22(B11-B21)
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        V[(b0+1) * p + c0, col] = -1
        W[c0 * n + a0, col] = 1
        W[(c0+1) * n + a0, col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a0 * m + b0, col] = 1
        U[a0 * m + (b0+1), col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[c0 * n + a0, col] = -1
        W[c0 * n + (a0+1), col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(a0+1) * m + b0, col] = 1
        U[a0 * m + b0, col] = -1
        V[b0 * p + c0, col] = 1
        V[b0 * p + (c0+1), col] = 1
        W[(c0+1) * n + (a0+1), col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a0 * m + (b0+1), col] = 1
        U[(a0+1) * m + (b0+1), col] = -1
        V[(b0+1) * p + c0, col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[c0 * n + a0, col] = 1
        W[(c0+1) * n + (a0+1), col] = 1
        col += 1

    # 1. A1 BL: 2x4 * 4x4
    # A1 is rows 0,1 of A. BL is cols 0-3 of B.
    # Split A1 into A11 (cols 0,1) and A12 (cols 2,3).
    # Split BL rows into B_L1 (rows 0,1) and B_L2 (rows 2,3).
    
    # A11 (2x2) * B_L1 (2x4)
    # B_L1 is 2x4. Split into B_L1_L (2x2) and B_L1_R (2x2).
    # A11 * B_L1_L: 2x2 * 2x2 -> Strassen
    add_strassen(0, 0, 0) # a0=0, b0=0, c0=0
    # A11 * B_L1_R: 2x2 * 2x2 -> Strassen
    add_strassen(0, 0, 2) # a0=0, b0=0, c0=2
    
    # A12 (2x2) * B_L2 (2x4)
    # B_L2 is 2x4. Split into B_L2_L (2x2) and B_L2_R (2x2).
    # A12 * B_L2_L: 2x2 * 2x2 -> Strassen
    add_strassen(0, 2, 0) # a0=0, b0=2, c0=0
    # A12 * B_L2_R: 2x2 * 2x2 -> Strassen
    add_strassen(0, 2, 2) # a0=0, b0=2, c0=2
    
    # 2. A1 BR: 2x4 * 4x3
    # A11 (2x2) * B_R1 (2x3)
    # B_R1 is 2x3. Split into B_R1_L (2x2) and B_R1_R (2x1).
    # A11 * B_R1_L: 2x2 * 2x2 -> Strassen
    add_strassen(0, 0, 4) # a0=0, b0=0, c0=4
    # A11 * B_R1_R: 2x2 * 2x1 -> Schoolbook (Rank 4)
    # A is 2x2, B is 2x1, C is 2x1.
    # A rows 0,1. B rows 0,1. C cols 0.
    # B col is 6.
    add_schoolbook(2, 2, 1, 0, 0, 6)
    
    # A12 (2x2) * B_R2 (2x3)
    # B_R2 is 2x3. Split into B_R2_L (2x2) and B_R2_R (2x1).
    # A12 * B_R2_L: 2x2 * 2x2 -> Strassen
    add_strassen(0, 2, 4) # a0=0, b0=2, c0=4
    # A12 * B_R2_R: 2x2 * 2x1 -> Schoolbook (Rank 4)
    add_schoolbook(2, 2, 1, 0, 2, 6)
    
    # 3. A2 BL: 1x4 * 4x4
    # A2 is row 2 of A. BL is cols 0-3 of B.
    # Split A2 into A21 (cols 0,1) and A22 (cols 2,3).
    # Split BL rows into B_L1 (rows 0,1) and B_L2 (rows 2,3).
    
    # A21 (1x2) * B_L1 (2x4) -> Schoolbook Rank 8
    add_schoolbook(1, 2, 4, 2, 0, 0)
    
    # A22 (1x2) * B_L2 (2x4) -> Schoolbook Rank 8
    add_schoolbook(1, 2, 4, 2, 2, 0)
    
    # 4. A2 BR: 1x4 * 4x3
    # A21 (1x2) * B_R1 (2x3) -> Schoolbook Rank 6
    add_schoolbook(1, 2, 3, 2, 0, 4)
    
    # A22 (1x2) * B_R2 (2x3) -> Schoolbook Rank 6
    add_schoolbook(1, 2, 3, 2, 2, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
