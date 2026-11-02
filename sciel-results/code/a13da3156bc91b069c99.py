"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def get_strassen_222():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C11, C12, C21, C22 all use m1? No.
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's define W carefully.
    # W[k*2+i, c]
    # k=0 (row 1 of C), k=1 (row 2 of C)
    # i=0 (col 1 of C), i=1 (col 2 of C)
    
    # m1
    W[0, 0] = 1 # C11
    W[3, 0] = 1 # C22
    
    # m2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    W[3, 1] = -1 # C22
    
    # m3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12
    W[3, 2] = 1 # C22
    
    # m4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C11
    W[2, 3] = 1 # C21
    
    # m5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1 # C12
    
    # m6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # m7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11
    
    return U, V, W

def get_mul_223():
    """Returns U, V, W for 2x2x3 decomposition (rank 11)."""
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Part 1: 2x2x2 (cols 0,1 of output)
    Us, Vs, Ws = get_strassen_222()
    for i in range(4):
        U[i, 0:7] = Us[i, :]
    for j in range(2):
        for k in range(2):
            V[j*3 + k, 0:7] = Vs[j*2 + k, :]
    for k in range(2):
        for i in range(2):
            W[k*2 + i, 0:7] = Ws[k*2 + i, :]
            
    # Part 2: 2x2x1 (col 2 of output)
    # C13 = A11*b1 + A12*b2
    # C23 = A21*b1 + A22*b2
    # Term 7: A11*b1
    U[0, 7] = 1
    V[2, 7] = 1 # j=0, k=2
    W[4, 7] = 1 # k=2, i=0
    
    # Term 8: A12*b2
    U[1, 8] = 1
    V[5, 8] = 1 # j=1, k=2
    W[4, 8] = 1 # k=2, i=0
    
    # Term 9: A21*b1
    U[2, 9] = 1
    V[2, 9] = 1 # j=0, k=2
    W[5, 9] = 1 # k=2, i=1
    
    # Term 10: A22*b2
    U[3, 10] = 1
    V[5, 10] = 1 # j=1, k=2
    W[5, 10] = 1 # k=2, i=1
    
    return U, V, W

def get_mul_233():
    """Returns U, V, W for 2x3x3 decomposition (rank 12)."""
    U = np.zeros((6, 12))
    V = np.zeros((9, 12))
    W = np.zeros((6, 12))
    
    # Part 1: 2x2x3 (using A_left 2x2, B_top 2x3)
    Um, Vm, Wm = get_mul_223()
    for i in range(2):
        for j in range(2):
            U[i*3 + j, 0:11] = Um[i*2 + j, :]
    for j in range(2):
        for k in range(3):
            V[j*3 + k, 0:11] = Vm[j*3 + k, :]
    for k in range(3):
        for i in range(2):
            W[k*2 + i, 0:11] = Wm[k*2 + i, :]
            
    # Part 2: Rank 1 term (a_right 2x1, b_bottom 1x3)
    # C_ik = a_i * b_k
    # Term 11
    U[0*3 + 2, 11] = 1 # i=0, j=2
    U[1*3 + 2, 11] = 1 # i=1, j=2
    V[2*3 + 0, 11] = 1 # j=2, k=0
    V[2*3 + 1, 11] = 1 # j=2, k=1
    V[2*3 + 2, 11] = 1 # j=2, k=2
    for k in range(3):
        for i in range(2):
            W[k*2 + i, 11] = 1
            
    return U, V, W

def construct():
    """
    Find a low-rank tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses block decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Total rank: 4 * 11 + 4 * 12 = 44 + 48 = 92
    total_rank = 92
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_223, V_223, W_223 = get_mul_223()
    U_233, V_233, W_233 = get_mul_233()
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_loc, V_loc, W_loc, i_off, j_off, k_off):
        nonlocal col_offset
        for c in range(U_loc.shape[1]):
            for i in range(U_loc.shape[0]):
                U[i_off*5 + j_off + (i % 2)*5 + (i//2), col_offset + c] = U_loc[i, c] # This mapping is wrong.
        
    # Let's do it manually for clarity and correctness.
    
    # Block 1: A_L1 B_T1 (2x2x3)
    # A: rows 0-1, cols 0-1. B: rows 0-1, cols 0-2. C: rows 0-1, cols 0-2.
    for c in range(11):
        for i in range(2):
            for j in range(2):
                U[i*5 + j, col_offset + c] = U_223[i*2 + j, c]
        for j in range(2):
            for k in range(3):
                V[j*6 + k, col_offset + c] = V_223[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[k*4 + i, col_offset + c] = W_223[k*2 + i, c]
    col_offset += 11
    
    # Block 2: A_L1 B_T2 (2x2x3)
    # A: rows 0-1, cols 0-1. B: rows 0-1, cols 3-5. C: rows 0-1, cols 3-5.
    for c in range(11):
        for i in range(2):
            for j in range(2):
                U[i*5 + j, col_offset + c] = U_223[i*2 + j, c]
        for j in range(2):
            for k in range(3):
                V[j*6 + (k+3), col_offset + c] = V_223[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[(k+3)*4 + i, col_offset + c] = W_223[k*2 + i, c]
    col_offset += 11
    
    # Block 3: A_L2 B_T1 (2x2x3)
    # A: rows 2-3, cols 0-1. B: rows 0-1, cols 0-2. C: rows 2-3, cols 0-2.
    for c in range(11):
        for i in range(2):
            for j in range(2):
                U[(i+2)*5 + j, col_offset + c] = U_223[i*2 + j, c]
        for j in range(2):
            for k in range(3):
                V[j*6 + k, col_offset + c] = V_223[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[k*4 + (i+2), col_offset + c] = W_223[k*2 + i, c]
    col_offset += 11
    
    # Block 4: A_L2 B_T2 (2x2x3)
    # A: rows 2-3, cols 0-1. B: rows 0-1, cols 3-5. C: rows 2-3, cols 3-5.
    for c in range(11):
        for i in range(2):
            for j in range(2):
                U[(i+2)*5 + j, col_offset + c] = U_223[i*2 + j, c]
        for j in range(2):
            for k in range(3):
                V[j*6 + (k+3), col_offset + c] = V_223[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[(k+3)*4 + (i+2), col_offset + c] = W_223[k*2 + i, c]
    col_offset += 11
    
    # Block 5: A_R1 B_B1 (2x3x3)
    # A: rows 0-1, cols 2-4. B: rows 2-4, cols 0-2. C: rows 0-1, cols 0-2.
    for c in range(12):
        for i in range(2):
            for j in range(3):
                U[i*5 + (j+2), col_offset + c] = U_233[i*3 + j, c]
        for j in range(3):
            for k in range(3):
                V[(j+2)*6 + k, col_offset + c] = V_233[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[k*4 + i, col_offset + c] = W_233[k*2 + i, c]
    col_offset += 12
    
    # Block 6: A_R1 B_B2 (2x3x3)
    # A: rows 0-1, cols 2-4. B: rows 2-4, cols 3-5. C: rows 0-1, cols 3-5.
    for c in range(12):
        for i in range(2):
            for j in range(3):
                U[i*5 + (j+2), col_offset + c] = U_233[i*3 + j, c]
        for j in range(3):
            for k in range(3):
                V[(j+2)*6 + (k+3), col_offset + c] = V_233[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[(k+3)*4 + i, col_offset + c] = W_233[k*2 + i, c]
    col_offset += 12
    
    # Block 7: A_R2 B_B1 (2x3x3)
    # A: rows 2-3, cols 2-4. B: rows 2-4, cols 0-2. C: rows 2-3, cols 0-2.
    for c in range(12):
        for i in range(2):
            for j in range(3):
                U[(i+2)*5 + (j+2), col_offset + c] = U_233[i*3 + j, c]
        for j in range(3):
            for k in range(3):
                V[(j+2)*6 + k, col_offset + c] = V_233[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[k*4 + (i+2), col_offset + c] = W_233[k*2 + i, c]
    col_offset += 12
    
    # Block 8: A_R2 B_B2 (2x3x3)
    # A: rows 2-3, cols 2-4. B: rows 2-4, cols 3-5. C: rows 2-3, cols 3-5.
    for c in range(12):
        for i in range(2):
            for j in range(3):
                U[(i+2)*5 + (j+2), col_offset + c] = U_233[i*3 + j, c]
        for j in range(3):
            for k in range(3):
                V[(j+2)*6 + (k+3), col_offset + c] = V_233[j*3 + k, c]
        for k in range(3):
            for i in range(2):
                W[(k+3)*4 + (i+2), col_offset + c] = W_233[k*2 + i, c]
    col_offset += 12
    
    # Convert to lists for the required interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
