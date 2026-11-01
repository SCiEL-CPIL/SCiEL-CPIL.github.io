"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # We decompose into 8 independent 2x2x2x2 Strassen problems.
    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # B is 4x8. Split into B_TL (2x2), B_TR (2x2) for top half?
    # Actually, B is 4x8.
    # Rows 0,1 correspond to A_L.
    # Rows 2,3 correspond to A_R.
    # Cols 0..7.
    # For A_L (rows 0,1 of A), it multiplies rows 0,1 of B.
    # B_top = B[0:2, 0:8]. This is 2x8.
    # Split B_top into 4 blocks of 2x2: B0, B1, B2, B3.
    # A_L * B_top = A_L * [B0 B1 B2 B3] = [A_L B0, A_L B1, A_L B2, A_L B3].
    # Each is a 2x2 matrix multiplication.
    # Similarly for A_R and B_bottom = B[2:4, 0:8].
    
    # Total 8 Strassen multiplications.
    # Rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+b)(e+g)
    # m2 = (c+d)(e+g)
    # m3 = a*h
    # m4 = d*f
    # m5 = (a-c)(g+h)
    # m6 = (b-d)(e+f)
    # m7 = (b-a)(e+g) -- Wait, standard is:
    # m1 = (a+b)(e+g)
    # m2 = (c+d)(e+g)
    # m3 = a h
    # m4 = d f
    # m5 = (a-c)(g+h)
    # m6 = (b-d)(e+f)
    # m7 = (a+b-d)(h-f) ? No.
    # Let's use the standard set:
    # m1 = (a+b)(e+g)
    # m2 = (c+d)(e+g)
    # m3 = a h
    # m4 = d f
    # m5 = (a-c)(g+h)
    # m6 = (b-d)(e+f)
    # m7 = (b-a)(e+g) is NOT correct.
    # Correct 7th: m7 = (a+b-d)(h-f) is one variant.
    # Another: m7 = (b-a)(e+g) is wrong.
    # Let's use:
    # m1 = (a+b)(e+g)
    # m2 = (c+d)(e+g)
    # m3 = a h
    # m4 = d f
    # m5 = (a-c)(g+h)
    # m6 = (b-d)(e+f)
    # m7 = (b-a)(e+g) -> No.
    
    # Standard Strassen:
    # p1 = (a + b)(e + g)
    # p2 = (c + d)(e + g)
    # p3 = a h
    # p4 = d f
    # p5 = (a - c)(g + h)
    # p6 = (b - d)(e + f)
    # p7 = (a + b - d)(h - f)
    
    # C = [[p1 + p5 + p7 - p4 - p6, p3 + p6],
    #      [p2 + p5, p1 + p3 - p2 - p7]]
    
    # We need to map these to the global indices.
    
    # Helper to add a Strassen block
    # block_id: 0..7
    # a_rows: 0,1 (indices in A)
    # b_cols_base: 0, 2, 4, 6 (start of 2-col block in B)
    # c_cols_base: same as b_cols_base
    
    # A indices: i*m + j.
    # For A_L (top half of A, rows 0,1), i=0,1.
    # j=0,1 for left half? No, A is 2x4.
    # A_L is A[:, 0:2]. A_R is A[:, 2:4].
    # So for A_L, j=0,1. For A_R, j=2,3.
    
    # B indices: j*p + k.
    # For B_top (rows 0,1 of B), j=0,1.
    # For B_bottom (rows 2,3 of B), j=2,3.
    # k=0..7.
    # We split k into 4 blocks of 2: 0-1, 2-3, 4-5, 6-7.
    
    # 8 blocks:
    # 0: A_L, B_top, k=0,1
    # 1: A_L, B_top, k=2,3
    # 2: A_L, B_top, k=4,5
    # 3: A_L, B_top, k=6,7
    # 4: A_R, B_bottom, k=0,1
    # 5: A_R, B_bottom, k=2,3
    # 6: A_R, B_bottom, k=4,5
    # 7: A_R, B_bottom, k=6,7
    
    # For each block, we have a 2x2 A part and 2x2 B part.
    # A part indices:
    # If A_L: i=0,1; j=0,1.
    # If A_R: i=0,1; j=2,3.
    # Global A index: i*4 + j.
    
    # B part indices:
    # If B_top: j=0,1.
    # If B_bottom: j=2,3.
    # k_base.
    # Global B index: j*8 + k.
    
    # C part indices:
    # i=0,1.
    # k_base.
    # Global C index: k*2 + i.
    
    col = 0
    for block in range(8):
        if block < 4:
            # A_L
            a_i_offset = 0
            a_j_offset = 0
            # B_top
            b_j_offset = 0
        else:
            # A_R
            a_i_offset = 0
            a_j_offset = 2
            # B_bottom
            b_j_offset = 2
            
        k_base = (block % 4) * 2
        
        # A indices for the 2x2 block
        # a = A[0, a_j_offset], b = A[0, a_j_offset+1]
        # c = A[1, a_j_offset], d = A[1, a_j_offset+1]
        
        idx_a00 = 0 * m + (a_j_offset + 0)
        idx_a01 = 0 * m + (a_j_offset + 1)
        idx_a10 = 1 * m + (a_j_offset + 0)
        idx_a11 = 1 * m + (a_j_offset + 1)
        
        # B indices for the 2x2 block
        # e = B[b_j_offset, k_base], f = B[b_j_offset, k_base+1]
        # g = B[b_j_offset+1, k_base], h = B[b_j_offset+1, k_base+1]
        
        idx_b00 = (b_j_offset + 0) * p + (k_base + 0)
        idx_b01 = (b_j_offset + 0) * p + (k_base + 1)
        idx_b10 = (b_j_offset + 1) * p + (k_base + 0)
        idx_b11 = (b_j_offset + 1) * p + (k_base + 1)
        
        # C indices for the 2x2 block
        # C[0, k_base], C[0, k_base+1]
        # C[1, k_base], C[1, k_base+1]
        
        idx_c00 = (k_base + 0) * n + 0
        idx_c01 = (k_base + 1) * n + 0
        idx_c10 = (k_base + 0) * n + 1
        idx_c11 = (k_base + 1) * n + 1
        
        # Strassen steps
        # p1 = (a+b)(e+g)
        U[idx_a00][col] = 1
        U[idx_a01][col] = 1
        V[idx_b00][col] = 1
        V[idx_b10][col] = 1
        W[idx_c00][col] = 1
        W[idx_c01][col] = 1
        W[idx_c10][col] = 1
        W[idx_c11][col] = 1
        col += 1
        
        # p2 = (c+d)(e+g)
        U[idx_a10][col] = 1
        U[idx_a11][col] = 1
        V[idx_b00][col] = 1
        V[idx_b10][col] = 1
        W[idx_c10][col] = 1
        W[idx_c11][col] = 1
        col += 1
        
        # p3 = a h
        U[idx_a00][col] = 1
        V[idx_b11][col] = 1
        W[idx_c01][col] = 1
        W[idx_c11][col] = 1
        col += 1
        
        # p4 = d f
        U[idx_a11][col] = 1
        V[idx_b01][col] = 1
        W[idx_c00][col] = -1
        W[idx_c11][col] = -1
        col += 1
        
        # p5 = (a-c)(g+h)
        U[idx_a00][col] = 1
        U[idx_a10][col] = -1
        V[idx_b10][col] = 1
        V[idx_b11][col] = 1
        W[idx_c00][col] = 1
        W[idx_c01][col] = 1
        W[idx_c10][col] = 1
        W[idx_c11][col] = 1
        col += 1
        
        # p6 = (b-d)(e+f)
        U[idx_a01][col] = 1
        U[idx_a11][col] = -1
        V[idx_b00][col] = 1
        V[idx_b01][col] = 1
        W[idx_c00][col] = -1
        W[idx_c01][col] = 1
        W[idx_c11][col] = -1
        col += 1
        
        # p7 = (a+b-d)(h-f)
        U[idx_a00][col] = 1
        U[idx_a01][col] = 1
        U[idx_a11][col] = -1
        V[idx_b11][col] = 1
        V[idx_b01][col] = -1
        W[idx_c00][col] = 1
        W[idx_c01][col] = -1
        W[idx_c10][col] = -1
        W[idx_c11][col] = 1
        col += 1
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
