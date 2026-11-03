"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 64 decomposition
    # Part 1: M_4,4,4 (Rank 28)
    # Part 2: M_4,1,4 (Rank 16)
    # Part 3: M_4,5,1 (Rank 20)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx):
        U_parts.append([0] * 1) # Placeholder, we build lists of rows
        # Actually, let's build the full matrices directly or lists of columns
        pass

    # Let's build U, V, W as lists of rows (lists of floats)
    # We know the total rank is 64.
    rank = 64
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: M_4,4,4 using 4 blocks of M_2,2,2 (Strassen) ---
    # Strassen coefficients for M_2,2,2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the global indices.
    # For a block at (br, bc) in A and (br, bc) in B, result at (br, bc) in C.
    # A indices: i in [br*2, br*2+1], j in [bc*2, bc*2+1]
    # B indices: j in [br*2, br*2+1], k in [bc*2, bc*2+1]
    # C indices: i in [br*2, br*2+1], k in [bc*2, bc*2+1]
    
    def add_strassen_block(br, bc):
        nonlocal col
        # A block rows: 2*br, 2*br+1. Cols: 2*bc, 2*bc+1
        a00_i, a00_j = 2*br, 2*bc
        a01_i, a01_j = 2*br, 2*bc+1
        a10_i, a10_j = 2*br+1, 2*bc
        a11_i, a11_j = 2*br+1, 2*bc+1
        
        # B block rows: 2*br, 2*br+1. Cols: 2*bc, 2*bc+1
        b00_j, b00_k = 2*br, 2*bc
        b01_j, b01_k = 2*br, 2*bc+1
        b10_j, b10_k = 2*br+1, 2*bc
        b11_j, b11_k = 2*br+1, 2*bc+1
        
        # C block rows: 2*br, 2*br+1. Cols: 2*bc, 2*bc+1
        c00_i, c00_k = 2*br, 2*bc
        c01_i, c01_k = 2*br, 2*bc+1
        c10_i, c10_k = 2*br+1, 2*bc
        c11_i, c11_k = 2*br+1, 2*bc+1
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 terms for this block.
        
        # Term 1: M1
        U[a00_i*m + a00_j][col] = 1
        U[a11_i*m + a11_j][col] = 1
        V[b00_j*p + b00_k][col] = 1
        V[b11_j*p + b11_k][col] = 1
        W[c00_i*n + c00_k][col] = 1 # C00
        W[c01_i*n + c01_k][col] = 1 # C01? No, C01 is (0,1) in block.
        # Wait, the W index is k*n + i.
        # For C00 (local 0,0): i=2*br, k=2*bc. W[2*bc*n + 2*br]
        # For C01 (local 0,1): i=2*br, k=2*bc+1. W[(2*bc+1)*n + 2*br]
        # For C10 (local 1,0): i=2*br+1, k=2*bc. W[2*bc*n + 2*br+1]
        # For C11 (local 1,1): i=2*br+1, k=2*bc+1. W[(2*bc+1)*n + 2*br+1]
        
        # Correction:
        # M1 contributes to C00, C01, C10, C11?
        # No, M1 is a scalar.
        # C00 += M1
        # C01 += M1? No.
        # Let's re-verify Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00 and C11.
        W[c00_i*n + c00_k][col] = 1
        W[c11_i*n + c11_k][col] = 1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[a10_i*m + a10_j][col] = 1
        U[a11_i*m + a11_j][col] = 1
        V[b00_j*p + b00_k][col] = 1
        W[c10_i*n + c10_k][col] = 1
        W[c11_i*n + c11_k][col] = -1
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[a00_i*m + a00_j][col] = 1
        V[b01_j*p + b01_k][col] = 1
        V[b11_j*p + b11_k][col] = -1
        W[c01_i*n + c01_k][col] = 1
        W[c11_i*n + c11_k][col] = 1
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[a11_i*m + a11_j][col] = 1
        V[b10_j*p + b10_k][col] = 1
        V[b00_j*p + b00_k][col] = -1
        W[c00_i*n + c00_k][col] = 1
        W[c10_i*n + c10_k][col] = 1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[a00_i*m + a00_j][col] = 1
        U[a01_i*m + a01_j][col] = 1
        V[b11_j*p + b11_k][col] = 1
        W[c00_i*n + c00_k][col] = -1
        W[c01_i*n + c01_k][col] = 1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[a10_i*m + a10_j][col] = 1
        U[a00_i*m + a00_j][col] = -1
        V[b00_j*p + b00_k][col] = 1
        V[b01_j*p + b01_k][col] = 1
        W[c11_i*n + c11_k][col] = 1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[a01_i*m + a01_j][col] = 1
        U[a11_i*m + a11_j][col] = -1
        V[b10_j*p + b10_k][col] = 1
        V[b11_j*p + b11_k][col] = 1
        W[c00_i*n + c00_k][col] = 1
        col += 1

    for br in range(2):
        for bc in range(2):
            add_strassen_block(br, bc)
            
    # --- Part 2: M_4,1,4 (Rank 16) ---
    # A[:, 4] * B[4, :] -> C[:, 0:4]
    for i in range(4):
        for k in range(4):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1

    # --- Part 3: M_4,5,1 (Rank 20) ---
    # A[:, 0:5] * B[:, 4] -> C[:, 4]
    for i in range(4):
        for j in range(5):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
