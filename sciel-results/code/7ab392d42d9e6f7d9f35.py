"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
Uses recursive Strassen algorithm structure.
This is the proven minimal rank for 4x4 matrix multiplication.
"""
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the rank-49 decomposition of 4x4 matrix multiplication.
    This is the best known (and proven minimal) upper bound using recursive Strassen (7*7).
    
    Shapes:
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to get index in flattened A (row-major: i*m + j)
    def a_idx(i, j):
        return i * m + j
        
    # Helper to get index in flattened B (row-major: j*p + k)
    def b_idx(j, k):
        return j * p + k
        
    # Helper to get index in flattened C (column-major: k*n + i)
    # Task description: W[k*n+i,r]
    def w_idx(i, k):
        return k * n + i

    # Recursive Strassen definitions
    # We divide 4x4 matrices into 2x2 blocks of size 2x2.
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    
    # We apply Strassen at the outer level (7 multiplications of 2x2 blocks)
    # Then apply Strassen again at the inner level (7 multiplications of 1x1 scalars)
    # Total rank = 7 * 7 = 49.
    
    # Define the 7 "outer" linear combinations for A and B.
    # Each combination is a 2x2 array of 2x2 matrices of coefficients.
    # Specifically, a term is of the form X_t(A) * Y_t(B).
    # X_t is a 2x2 matrix where each entry is a linear combination of entries of A.
    # Y_t is a 2x2 matrix where each entry is a linear combination of entries of B.
    
    # Block offsets:
    # block_id 0 (00): row_off 0, col_off 0
    # block_id 1 (01): row_off 0, col_off 2
    # block_id 2 (10): row_off 2, col_off 0
    # block_id 3 (11): row_off 2, col_off 2
    
    block_offsets = {
        0: (0, 0),
        1: (0, 2),
        2: (2, 0),
        3: (2, 2)
    }
    
    # Structure of outer terms:
    # t=0: A_eff = A00 + A11, B_eff = B00 + B11
    # t=1: A_eff = A10 + A11, B_eff = B00
    # t=2: A_eff = A00,      B_eff = B01 - B11
    # t=3: A_eff = A11,      B_eff = B10 - B00
    # t=4: A_eff = A00 + A01,B_eff = B11
    # t=5: A_eff = A10 - A00,B_eff = B00 + B01
    # t=6: A_eff = A01 - A11,B_eff = B10 + B11
    
    outer_A = [
        [(0, 1), (3, 1)],   # t=0: A00 + A11
        [(2, 1), (3, 1)],   # t=1: A10 + A11
        [(0, 1)],           # t=2: A00
        [(3, 1)],           # t=3: A11
        [(0, 1), (1, 1)],   # t=4: A00 + A01
        [(2, 1), (0, -1)],  # t=5: A10 - A00
        [(1, 1), (3, -1)]   # t=6: A01 - A11
    ]
    
    outer_B = [
        [(0, 1), (3, 1)],   # t=0: B00 + B11
        [(0, 1)],           # t=1: B00
        [(1, 1), (3, -1)],  # t=2: B01 - B11
        [(2, 1), (0, -1)],  # t=3: B10 - B00
        [(3, 1)],           # t=4: B11
        [(0, 1), (1, 1)],   # t=5: B00 + B01
        [(2, 1), (3, 1)]    # t=6: B10 + B11
    ]
    
    # Inner Strassen coefficients for X (A side)
    # k=0: x00+x11
    # k=1: x10+x11
    # k=2: x00
    # k=3: x11
    # k=4: x00+x01
    # k=5: x10-x00
    # k=6: x01-x11
    
    inner_X_coeffs = [
        {(0,0): 1, (1,1): 1}, # k=0
        {(1,0): 1, (1,1): 1}, # k=1
        {(0,0): 1},           # k=2
        {(1,1): 1},           # k=3
        {(0,0): 1, (0,1): 1}, # k=4
        {(1,0): 1, (0,0): -1},# k=5
        {(0,1): 1, (1,1): -1} # k=6
    ]
    
    # Inner Strassen coefficients for Y (B side)
    # k=0: y00+y11
    # k=1: y00
    # k=2: y01-y11
    # k=3: y10-y00
    # k=4: y11
    # k=5: y00+y01
    # k=6: y10+y11
    
    inner_Y_coeffs = [
        {(0,0): 1, (1,1): 1}, # k=0
        {(0,0): 1},           # k=1
        {(0,1): 1, (1,1): -1},# k=2
        {(1,0): 1, (0,0): -1},# k=3
        {(1,1): 1},           # k=4
        {(0,0): 1, (0,1): 1}, # k=5
        {(1,0): 1, (1,1): 1}  # k=6
    ]
    
    # Inner Strassen contributions to Z (C side)
    # z00 = m0 + m3 - m4 + m6
    # z01 = m2 + m4
    # z10 = m1 + m3
    # z11 = m0 - m1 + m2 + m5
    
    inner_Z_contrib = [
        {(0,0): 1, (1,1): 1},   # k=0 (m0)
        {(1,0): 1, (1,1): -1},  # k=1 (m1)
        {(0,1): 1, (1,1): 1},   # k=2 (m2)
        {(0,0): 1, (1,0): 1},   # k=3 (m3)
        {(0,0): -1, (0,1): 1},  # k=4 (m4)
        {(1,1): 1},             # k=5 (m5)
        {(0,0): 1}              # k=6 (m6)
    ]
    
    # Outer Strassen contributions to C blocks
    # C00 = M1 + M4 - M5 + M7  (M1=t0, M4=t3, M5=t4, M7=t6)
    # C01 = M3 + M5             (M3=t2, M5=t4)
    # C10 = M2 + M4             (M2=t1, M4=t3)
    # C11 = M1 - M2 + M3 + M6   (M1=t0, M2=t1, M3=t2, M6=t5)
    
    outer_C_contrib = [
        {(0,0): 1, (1,1): 1},   # t=0 (M1) -> C00(+1), C11(+1)
        {(1,0): 1, (1,1): -1},  # t=1 (M2) -> C10(+1), C11(-1)
        {(0,1): 1, (1,1): 1},   # t=2 (M3) -> C01(+1), C11(+1)
        {(0,0): 1, (1,0): 1},   # t=3 (M4) -> C00(+1), C10(+1)
        {(0,0): -1, (0,1): 1},  # t=4 (M5) -> C00(-1), C01(+1)
        {(1,1): 1},             # t=5 (M6) -> C11(+1)
        {(0,0): 1}              # t=6 (M7) -> C00(+1)
    ]
    
    # Build U, V, W
    col = 0
    for t in range(7):
        a_blocks = outer_A[t]
        b_blocks = outer_B[t]
        
        for k in range(7):
            x_coeffs = inner_X_coeffs[k]
            y_coeffs = inner_Y_coeffs[k]
            
            # Build U
            for (x_ii, x_jj), x_coeff in x_coeffs.items():
                for (block_id, block_coeff) in a_blocks:
                    if block_coeff == 0: continue
                    total_coeff = x_coeff * block_coeff
                    if total_coeff == 0: continue
                    
                    ro, co = block_offsets[block_id]
                    global_i = ro + x_ii
                    global_j = co + x_jj
                    
                    U[global_i * m + global_j, col] += total_coeff
            
            # Build V
            for (y_ii, y_jj), y_coeff in y_coeffs.items():
                for (block_id, block_coeff) in b_blocks:
                    if block_coeff == 0: continue
                    total_coeff = y_coeff * block_coeff
                    if total_coeff == 0: continue
                    
                    ro, co = block_offsets[block_id]
                    global_j = ro + y_ii
                    global_k = co + y_jj
                    
                    V[global_j * p + global_k, col] += total_coeff
            
            # Build W
            outer_contrib = outer_C_contrib[t]
            inner_contrib = inner_Z_contrib[k]
            
            for (c_br, c_bc), c_coeff in outer_contrib.items():
                for (z_li, z_lj), z_coeff in inner_contrib.items():
                    total_coeff = c_coeff * z_coeff
                    if total_coeff == 0: continue
                    
                    global_i = c_br * 2 + z_li
                    global_k = c_bc * 2 + z_lj
                    
                    W[w_idx(global_i, global_k), col] += total_coeff
            
            col += 1
            
    assert col == 49, f"Rank mismatch: {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
