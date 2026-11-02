"""
Matrix multiplication 2x4 * 4x4 tensor decomposition.
Uses a direct rank-28 decomposition based on Strassen's algorithm applied to 2x2 blocks.
"""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize factor matrices as NumPy arrays of zeros
    # U: (n*m, rank) -> (8, 28)
    # V: (m*p, rank) -> (16, 28)
    # W: (p*n, rank) -> (8, 28)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def idx_a(i, j):
        return i * m + j
        
    def idx_b(i, j):
        return i * p + j
        
    def idx_c(i, j):
        # Trace convention: W[k*n+i, r]
        # i is row of C, j is col of C
        return j * n + i

    def add_strassen_block(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """
        Adds a Strassen decomposition for a 2x2 block multiplication.
        A sub-block: A[a_r0:a_r0+2, a_c0:a_c0+2]
        B sub-block: B[b_r0:b_r0+2, b_c0:b_c0+2]
        C sub-block: C[c_r0:c_r0+2, c_c0:c_c0+2]
        """
        # U indices for A
        u_00 = idx_a(a_r0, a_c0)
        u_01 = idx_a(a_r0, a_c0 + 1)
        u_10 = idx_a(a_r0 + 1, a_c0)
        u_11 = idx_a(a_r0 + 1, a_c0 + 1)
        
        # V indices for B
        v_00 = idx_b(b_r0, b_c0)
        v_01 = idx_b(b_r0, b_c0 + 1)
        v_10 = idx_b(b_r0 + 1, b_c0)
        v_11 = idx_b(b_r0 + 1, b_c0 + 1)
        
        # W indices for C
        w_00 = idx_c(c_r0, c_c0)
        w_01 = idx_c(c_r0, c_c0 + 1)
        w_10 = idx_c(c_r0 + 1, c_c0)
        w_11 = idx_c(c_r0 + 1, c_c0 + 1)
        
        # Strassen M1..M7
        # M1 = (a00+a11)(b00+b11)
        U[u_00, offset + 0] += 1
        U[u_11, offset + 0] += 1
        V[v_00, offset + 0] += 1
        V[v_11, offset + 0] += 1
        W[w_00, offset + 0] += 1
        W[w_11, offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[u_10, offset + 1] += 1
        U[u_11, offset + 1] += 1
        V[v_00, offset + 1] += 1
        W[w_10, offset + 1] += 1
        W[w_11, offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[u_00, offset + 2] += 1
        V[v_01, offset + 2] += 1
        V[v_11, offset + 2] -= 1
        W[w_01, offset + 2] += 1
        W[w_11, offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[u_11, offset + 3] += 1
        V[v_10, offset + 3] += 1
        V[v_00, offset + 3] -= 1
        W[w_00, offset + 3] += 1
        W[w_10, offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[u_00, offset + 4] += 1
        U[u_01, offset + 4] += 1
        V[v_11, offset + 4] += 1
        W[w_00, offset + 4] -= 1
        W[w_01, offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u_10, offset + 5] += 1
        U[u_00, offset + 5] -= 1
        V[v_00, offset + 5] += 1
        V[v_01, offset + 5] += 1
        W[w_11, offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u_01, offset + 6] += 1
        U[u_11, offset + 6] -= 1
        V[v_10, offset + 6] += 1
        V[v_11, offset + 6] += 1
        W[w_00, offset + 6] += 1

    # Decompose the 2x4 * 4x4 into four 2x2 * 2x2 blocks
    # A is 2x4. Split into A_left (2x2) and A_right (2x2).
    # B is 4x4. Split into 2x2 blocks: Top-Left, Top-Right, Bottom-Left, Bottom-Right.
    # C is 2x4. Split into C_left (2x2) and C_right (2x2).
    
    # Block 1: A_left * B_top_left -> C_left
    # A rows 0-1, cols 0-1. B rows 0-1, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right * B_bottom_left -> C_left
    # A rows 0-1, cols 2-3. B rows 2-3, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left * B_top_right -> C_right
    # A rows 0-1, cols 0-1. B rows 0-1, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_right * B_bottom_right -> C_right
    # A rows 0-1, cols 2-3. B rows 2-3, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Convert to lists of lists to match expected interface format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
