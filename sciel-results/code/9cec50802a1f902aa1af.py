import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # Helper for 2x2x2 Strassen
    # i_off: row offset in A (0 or 1 relative to block)
    # j_off: col offset in A (0 or 1 relative to block)
    # j_off_B: row offset in B (0 or 1 relative to block)
    # k_off: col offset in B (0 or 1 relative to block)
    def add_strassen(i_off, j_off, j_off_B, k_off):
        nonlocal col
        # A indices: (i_off + 0, j_off + 0), (i_off + 0, j_off + 1), (i_off + 1, j_off + 0), (i_off + 1, j_off + 1)
        # B indices: (j_off_B + 0, k_off + 0), (j_off_B + 0, k_off + 1), (j_off_B + 1, k_off + 0), (j_off_B + 1, k_off + 1)
        
        # M1 = (A00+A11)(B00+B11)
        u = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off_B, k_off): 1, (j_off_B+1, k_off+1): 1}
        w = {(k_off, i_off): 1, (k_off+1, i_off+1): 1} # C00, C11
        # Map to flat indices
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)
        
        # M2 = (A10+A11)B00
        u = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off_B, k_off): 1}
        w = {(k_off, i_off+1): 1} # C10
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)
        
        # M3 = A00(B10-B00)
        u = {(i_off, j_off): 1}
        v = {(j_off_B+1, k_off): 1, (j_off_B, k_off): -1}
        w = {(k_off, i_off): 1} # C00? No, C01?
        # Strassen: C01 = M3 + M5. C00 = M1 + M4 - M5 + M7.
        # Let's check standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M3 contributes to C01.
        # C01 -> k=1, i=0.
        w = {(k_off+1, i_off): 1}
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)
        
        # M4 = A11(B01-B11)
        u = {(i_off+1, j_off+1): 1}
        v = {(j_off_B, k_off+1): 1, (j_off_B+1, k_off+1): -1}
        w = {(k_off, i_off): 1, (k_off, i_off+1): 1} # C00, C10
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)
        
        # M5 = (A00+A01)B11
        u = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        v = {(j_off_B+1, k_off+1): 1}
        w = {(k_off, i_off): -1, (k_off+1, i_off): 1} # C00 (-M5), C01 (+M5)
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)
        
        # M6 = (A10-A00)(B00+B01)
        u = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        v = {(j_off_B, k_off): 1, (j_off_B, k_off+1): 1}
        w = {(k_off+1, i_off+1): 1} # C11
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)
        
        # M7 = (A01-A11)(B10+B11)
        u = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        v = {(j_off_B+1, k_off): 1, (j_off_B+1, k_off+1): 1}
        w = {(k_off, i_off): 1} # C00
        u_flat = {i*m + j: val for (i, j), val in u.items()}
        v_flat = {j*p + k: val for (j, k), val in v.items()}
        w_flat = {k*n + i: val for (k, i), val in w.items()}
        add_term(u_flat, v_flat, w_flat)

    # Helper for 2x2x1 (A 2x2, B 2x1)
    # i_off, j_off for A
    # j_off_B, k_off for B
    def add_2x2x1(i_off, j_off, j_off_B, k_off):
        nonlocal col
        # A00*B00, A01*B10, A10*B00, A11*B10
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # T1: A00*B00 -> C00
        u = {(i_off, j_off): 1}
        v = {(j_off_B, k_off): 1}
        w = {(k_off, i_off): 1}
        add_term({i*m+j: val for i,j,val in u.items()}, {j*p+k: val for j,k,val in v.items()}, {k*n+i: val for k,i,val in w.items()})
        
        # T2: A01*B10 -> C00
        u = {(i_off, j_off+1): 1}
        v = {(j_off_B+1, k_off): 1}
        w = {(k_off, i_off): 1}
        add_term({i*m+j: val for i,j,val in u.items()}, {j*p+k: val for j,k,val in v.items()}, {k*n+i: val for k,i,val in w.items()})
        
        # T3: A10*B00 -> C10
        u = {(i_off+1, j_off): 1}
        v = {(j_off_B, k_off): 1}
        w = {(k_off, i_off+1): 1}
        add_term({i*m+j: val for i,j,val in u.items()}, {j*p+k: val for j,k,val in v.items()}, {k*n+i: val for k,i,val in w.items()})
        
        # T4: A11*B10 -> C10
        u = {(i_off+1, j_off+1): 1}
        v = {(j_off_B+1, k_off): 1}
        w = {(k_off, i_off+1): 1}
        add_term({i*m+j: val for i,j,val in u.items()}, {j*p+k: val for j,k,val in v.items()}, {k*n+i: val for k,i,val in w.items()})

    # Decomposition:
    # A = [A1, A2] (cols 0,1 and 2,3)
    # B = [B1; B2] (rows 0,1 and 2,3)
    
    # Block 1: A1 (i=0,1, j=0,1) * B1 (j=0,1, k=0..4)
    # Split B1 into B1_L (k=0,1) and B1_R (k=2,3,4)
    # B1_L: 2x2. Strassen.
    add_strassen(0, 0, 0, 0) # i_off=0, j_off=0, j_off_B=0, k_off=0
    
    # B1_R: 2x3. Split into B1_R_L (k=2,3) and B1_R_R (k=4)
    # B1_R_L: 2x2. Strassen.
    add_strassen(0, 0, 0, 2) # i_off=0, j_off=0, j_off_B=0, k_off=2
    
    # B1_R_R: 2x1. 2x2x1.
    add_2x2x1(0, 0, 0, 4) # i_off=0, j_off=0, j_off_B=0, k_off=4

    # Block 2: A2 (i=0,1, j=2,3) * B2 (j=2,3, k=0..4)
    # Split B2 into B2_L (k=0,1) and B2_R (k=2,3,4)
    # B2_L: 2x2. Strassen.
    add_strassen(0, 2, 2, 0) # i_off=0, j_off=2, j_off_B=2, k_off=0
    
    # B2_R: 2x3. Split into B2_R_L (k=2,3) and B2_R_R (k=4)
    # B2_R_L: 2x2. Strassen.
    add_strassen(0, 2, 2, 2) # i_off=0, j_off=2, j_off_B=2, k_off=2
    
    # B2_R_R: 2x1. 2x2x1.
    add_2x2x1(0, 2, 2, 4) # i_off=0, j_off=2, j_off_B=2, k_off=4

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
