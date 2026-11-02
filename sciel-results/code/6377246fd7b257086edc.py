"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = 140.0

    # Strategy:
    # Split A (3x5) into A_top (2x5) and A_bot (1x5).
    # Split B (5x5) into B_left (5x2) and B_right (5x3).
    # C = [A_top B_left + A_top B_right; A_bot B_left + A_bot B_right]
    # C is 3x5.
    # C_top_left (2x2) = A_top (2x5) * B_left (5x2)
    # C_top_right (2x3) = A_top (2x5) * B_right (5x3)
    # C_bot_left (1x2) = A_bot (1x5) * B_left (5x2)
    # C_bot_right (1x3) = A_bot (1x5) * B_right (5x3)
    
    # We will build the decomposition term by term.
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (15, rank)
    
    # Indices:
    # A[i, j] -> U[i*m + j, r]
    # B[j, k] -> V[j*p + k, r]
    # C[i, k] -> W[k*n + i, r]
    
    # We maintain lists of terms.
    # Each term is a dict or tuple specifying U, V, W coefficients.
    # Since we are doing exact integer arithmetic, we store coefficients.
    
    terms = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict { (i, j): coeff }
        v_coeffs: dict { (j, k): coeff }
        w_coeffs: dict { (i, k): coeff }
        """
        terms.append((u_coeffs, v_coeffs, w_coeffs))

    # --- Sub-problem 1: C_top_left (2x2) = A_top (2x5) * B_left (5x2) ---
    # A_top rows: 0, 1. Cols: 0..4.
    # B_left rows: 0..4. Cols: 0, 1.
    # C_top_left rows: 0, 1. Cols: 0, 1.
    
    # Split inner dim (5) into 2, 2, 1.
    # Blocks:
    # 1. A_top[:, 0:2] (2x2) * B_left[0:2, :] (2x2) -> Strassen (Rank 7)
    # 2. A_top[:, 2:4] (2x2) * B_left[2:4, :] (2x2) -> Strassen (Rank 7)
    # 3. A_top[:, 4:5] (2x1) * B_left[4:5, :] (1x2) -> Schoolbook (Rank 4)
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # We need to map these to the global indices.
    # For block 1:
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: i in {0,1}, k in {0,1}
    
    def strassen_2x2(a_i_base, a_j_base, b_j_base, b_k_base, c_i_base, c_k_base):
        """
        Adds 7 terms for a 2x2 Strassen multiplication.
        a_i_base, a_j_base: offset for A
        b_j_base, b_k_base: offset for B
        c_i_base, c_k_base: offset for C
        """
        # M1
        u = {(a_i_base, a_j_base): 1, (a_i_base+1, a_j_base+1): 1}
        v = {(b_j_base, b_k_base): 1, (b_j_base+1, b_k_base+1): 1}
        w = {(c_i_base, c_k_base): 1, (c_i_base+1, c_k_base+1): 1}
        add_term(u, v, w)
        
        # M2
        u = {(a_i_base+1, a_j_base): 1, (a_i_base+1, a_j_base+1): 1}
        v = {(b_j_base, b_k_base): 1}
        w = {(c_i_base+1, c_k_base): 1, (c_i_base+1, c_k_base+1): -1}
        add_term(u, v, w)
        
        # M3
        u = {(a_i_base, a_j_base): 1}
        v = {(b_j_base+1, b_k_base): 1, (b_j_base+1, b_k_base+1): -1}
        w = {(c_i_base, c_k_base+1): 1, (c_i_base+1, c_k_base+1): 1}
        add_term(u, v, w)
        
        # M4
        u = {(a_i_base+1, a_j_base+1): 1}
        v = {(b_j_base, b_k_base+1): 1, (b_j_base, b_k_base): -1}
        w = {(c_i_base, c_k_base): 1, (c_i_base+1, c_k_base): 1}
        add_term(u, v, w)
        
        # M5
        u = {(a_i_base, a_j_base): 1, (a_i_base, a_j_base+1): 1}
        v = {(b_j_base+1, b_k_base+1): 1}
        w = {(c_i_base, c_k_base): -1, (c_i_base, c_k_base+1): 1}
        add_term(u, v, w)
        
        # M6
        u = {(a_i_base+1, a_j_base): 1, (a_i_base, a_j_base): -1}
        v = {(b_j_base, b_k_base): 1, (b_j_base+1, b_k_base): 1}
        w = {(c_i_base+1, c_k_base): 1}
        add_term(u, v, w)
        
        # M7
        u = {(a_i_base, a_j_base+1): 1, (a_i_base+1, a_j_base+1): -1}
        v = {(b_j_base, b_k_base+1): 1, (b_j_base+1, b_k_base+1): 1}
        w = {(c_i_base, c_k_base): -1, (c_i_base+1, c_k_base+1): 1}
        add_term(u, v, w)
        
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # Block 3: A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]
    # Schoolbook for 2x1 * 1x2
    # C[i,k] = A[i,4] * B[4,k]
    # Terms:
    # i=0, k=0: A[0,4]B[4,0]
    add_term({(0, 4): 1}, {(4, 0): 1}, {(0, 0): 1})
    # i=0, k=1: A[0,4]B[4,1]
    add_term({(0, 4): 1}, {(4, 1): 1}, {(0, 1): 1})
    # i=1, k=0: A[1,4]B[4,0]
    add_term({(1, 4): 1}, {(4, 0): 1}, {(1, 0): 1})
    # i=1, k=1: A[1,4]B[4,1]
    add_term({(1, 4): 1}, {(4, 1): 1}, {(1, 1): 1})

    # --- Sub-problem 2: C_top_right (2x3) = A_top (2x5) * B_right (5x3) ---
    # A_top rows: 0, 1. Cols: 0..4.
    # B_right rows: 0..4. Cols: 2, 3, 4.
    # C_top_right rows: 0, 1. Cols: 2, 3, 4.
    
    # Split inner dim (5) into 2, 2, 1.
    # Block 1: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # 2x2 * 2x3. Schoolbook. Rank 12.
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term({(i, j): 1}, {(j, k+2): 1}, {(i, k+2): 1})
                
    # Block 2: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    # 2x2 * 2x3. Schoolbook. Rank 12.
    for i in range(2):
        for j in range(2, 4):
            for k in range(3):
                add_term({(i, j): 1}, {(j, k+2): 1}, {(i, k+2): 1})
                
    # Block 3: A[0:2, 4:5] * B[4:5, 2:5] -> C[0:2, 2:5]
    # 2x1 * 1x3. Schoolbook. Rank 6.
    for i in range(2):
        for k in range(3):
            add_term({(i, 4): 1}, {(4, k+2): 1}, {(i, k+2): 1})

    # --- Sub-problem 3: C_bot_left (1x2) = A_bot (1x5) * B_left (5x2) ---
    # A_bot row: 2. Cols: 0..4.
    # B_left rows: 0..4. Cols: 0, 1.
    # C_bot_left row: 2. Cols: 0, 1.
    
    # Schoolbook. Rank 10.
    for j in range(5):
        for k in range(2):
            add_term({(2, j): 1}, {(j, k): 1}, {(2, k): 1})

    # --- Sub-problem 4: C_bot_right (1x3) = A_bot (1x5) * B_right (5x3) ---
    # A_bot row: 2. Cols: 0..4.
    # B_right rows: 0..4. Cols: 2, 3, 4.
    # C_bot_right row: 2. Cols: 2, 3, 4.
    
    # Schoolbook. Rank 15.
    for j in range(5):
        for k in range(3):
            add_term({(2, j): 1}, {(j, k+2): 1}, {(2, k+2): 1})

    # Now, construct U, V, W matrices.
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
        for (i, j), coeff in u_coeffs.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_coeffs.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_coeffs.items():
            W[k * n + i][r] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
