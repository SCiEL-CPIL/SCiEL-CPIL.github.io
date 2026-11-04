"""Rewritten program for rank-minimal exact tensor decomposition of 4x4 x 4x5 matmul.

Uses Laderman's rank-49 decomposition for the 4x4x4 part
and schoolbook for the 4x4x1 part (rank 16), total rank 65.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# Deadline guard: kill after 150s wall-clock, use 120s safety margin
_DEADLINE = time.time() + 120

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-65 tensor decomposition for 4x4 x 4x5 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing factor matrices:
        - U: shape (16, 65)
        - V: shape (20, 65)  
        - W: shape (20, 65)
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build Laderman's rank-49 decomposition for 4x4x4
    U_444, V_444, W_444 = _laderman_4x4x4()
    
    # Build schoolbook decomposition for 4x4x1
    U_441, V_441, W_441 = _schoolbook_4x4x1()
    
    # Concatenate
    rank = U_444.shape[1] + U_441.shape[1]  # 49 + 16 = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill in Laderman part (for B columns 0-3)
    U[:, :49] = U_444
    V[:, :49] = V_444
    W[:, :49] = W_444
    
    # Fill in schoolbook part for B column 4
    col_off = 49
    U[:, col_off:] = U_441
    for j in range(4):
        for r in range(16):
            V[j * 5 + 4, col_off + r] = V_441[j, r]
    for i in range(4):
        for r in range(16):
            W[4 * 4 + i, col_off + r] = W_441[i, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _laderman_4x4x4():
    """Laderman's rank-49 decomposition for 4x4x4 matrix multiplication.
    
    Returns:
        U: shape (16, 49)
        V: shape (16, 49)
        W: shape (16, 49)
    """
    # Laderman's algorithm: 49 scalar multiplications for 4x4x4.
    # This is based on Strassen's algorithm applied to 2x2 blocks,
    # where each 2x2x2 uses Strassen's 7-multiplication algorithm.
    
    # 2x2 Strassen factors:
    # s1 = (a00+a11)*(b00+b11)
    # s2 = (a10+a11)*b00
    # s3 = a00*(b01-b11)
    # s4 = a11*(b10-b00)
    # s5 = (a00+a01)*b11
    # s6 = (a10-a00)*(b00+b01)
    # s7 = (a01-a11)*(b10+b11)
    # c00 = s1+s4-s7-s5
    # c01 = s3+s7
    # c10 = s2+s6
    # c11 = s1-s2+s3+s5
    
    s_a = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    s_b = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    s_c = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 0, 1],
        [0, 0, 1, 0],
        [-1, 1, 0, 0],
    ]
    
    # Top-level Strassen for 4x4:
    # T1 = (A11+A22)*(B11+B22)
    # T2 = (A21+A22)*B11
    # T3 = A11*(B12-B22)
    # T4 = A22*(B21-B11)
    # T5 = (A11+A12)*B22
    # T6 = (A21-A11)*(B11+B12)
    # T7 = (A12-A22)*(B21+B22)
    # C11 = T1+T4-T7-T5
    # C12 = T3+T7
    # C21 = T2+T6
    # C22 = T1-T2+T3+T5
    
    # Define block indices:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
    # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    col = 0
    for t in range(7):
        for b in range(7):
            # Determine A-side coefficients
            # Top-level t specifies which blocks of A are used and with what signs
            # Bottom-level b specifies which entries within the 2x2 block
            
            # For each top-level product, define the A block combination:
            # t=0 (T1): A11+A22 -> blocks [(0,0,1), (0,1,1), (1,0,1), (1,1,1)] where (bi,bj,coeff)
            # This means A_mat[i,j] = A_block1[i,j] + A_block2[i,j]
            
            # Let me define the top-level A and B block specifications:
            # Each entry is (block_row, block_col, coeff)
            
            top_A = [
                [(0, 0, 1), (1, 1, 1)],  # T1: A11+A22
                [(1, 0, 1), (1, 1, 1)],  # T2: A21+A22
                [(0, 0, 1)],             # T3: A11
                [(1, 1, 1)],             # T4: A22
                [(0, 0, 1), (0, 1, 1)],  # T5: A11+A12
                [(1, 0, 1), (0, 0, -1)], # T6: A21-A11
                [(0, 1, 1), (1, 1, -1)], # T7: A12-A22
            ]
            top_B = [
                [(0, 0, 1), (1, 1, 1)],  # T1: B11+B22
                [(0, 0, 1)],             # T2: B11
                [(0, 1, 1), (1, 1, -1)], # T3: B12-B22
                [(1, 0, 1), (0, 0, -1)], # T4: B21-B11
                [(1, 1, 1)],             # T5: B22
                [(0, 0, 1), (0, 1, 1)],  # T6: B11+B12
                [(1, 0, 1), (1, 1, 1)],  # T7: B21+B22
            ]
            top_C = [
                [(0, 0, 1), (1, 1, 1)],  # T1: C11+C22 (but with signs from output combo)
                [(1, 0, 1), (1, 1, -1)], # T2: C21-C22
                [(0, 1, 1), (1, 1, 1)],  # T3: C12+C22
                [(0, 0, 1)],             # T4: C11
                [(0, 0, -1), (1, 1, 1)], # T5: -C11+C22
                [(1, 0, 1)],             # T6: C21
                [(0, 0, -1), (0, 1, 1)], # T7: -C11+C12
            ]
            
            # Now, for this (t,b) pair, compute the scalar coefficients
            # The 2x2 A_mat has entries from combining the specified A blocks
            # The 2x2 B_mat has entries from combining the specified B blocks
            # The 2x2 C_mat has entries from combining the specified C blocks
            
            # For bottom-level Strassen product b:
            # a_coeff[i,j] from s_a[b]
            # b_coeff[i,j] from s_b[b]
            # c_coeff[i,j] from s_c[b]
            
            # The actual A entries are linear combinations of the 4x4 A entries
            # A_mat[i,j] = sum over specified blocks: coeff * A[2*block_row+i, 2*block_col+j]
            
            for ai in range(2):
                for aj in range(2):
                    a_val = 0.0
                    for (br, bc, coeff) in top_A[t]:
                        a_idx = (2 * br + ai) * 4 + (2 * bc + aj)
                        a_val += coeff * s_a[b][ai * 2 + aj]
                    if a_val != 0:
                        U[a_idx, col] = a_val
            
            for bi in range(2):
                for bj in range(2):
                    b_val = 0.0
                    for (br, bc, coeff) in top_B[t]:
                        b_idx = (2 * br + bi) * 4 + (2 * bc + bj)
                        b_val += coeff * s_b[b][bi * 2 + bj]
                    if b_val != 0:
                        V[b_idx, col] = b_val
            
            for ci in range(2):
                for cj in range(2):
                    c_val = 0.0
                    for (br, bc, coeff) in top_C[t]:
                        c_idx = (2 * br + ci) * 4 + (2 * bc + cj)
                        c_val += coeff * s_c[b][ci * 2 + cj]
                    if c_val != 0:
                        W[c_idx, col] = c_val
            
            col += 1
    
    return U, V, W


def _schoolbook_4x4x1():
    """Schoolbook decomposition for 4x4 x 4x1 matrix-vector product.
    
    Returns:
        U: shape (16, 16)
        V: shape (4, 16)
        W: shape (4, 16)
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    
    return U, V, W
# EVOLVE-BLOCK-END
