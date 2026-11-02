"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The problem is to multiply A (2x2) by B (2x12).
    # We view B as 6 blocks of 2x2 matrices.
    # The tensor is the Kronecker product of the 2x2x2 multiplication tensor and the 1x6 identity tensor.
    # Rank = Rank(2x2x2) * Rank(1x6) = 7 * 6 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    # U_forms[i][t] is the coefficient of A[i] in the t-th form.
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ]
    
    # For each of the 6 blocks u (corresponding to 2x2 sub-matrices of B)
    # and each of the 7 Strassen forms t
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U: The A-forms are the same for all blocks
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V: B-forms
            # B indices for block u:
            # b00 -> index 0*p + 2*u
            # b01 -> index 0*p + 2*u + 1
            # b10 -> index 1*p + 2*u
            # b11 -> index 1*p + 2*u + 1
            
            # Strassen B-forms (standard):
            # T0: b00 + b11
            # T1: b00
            # T2: b01 - b11
            # T3: b10 - b00
            # T4: b11
            # T5: b00 + b01
            # T6: b10 + b11
            
            b00_idx = 0 * p + 2*u
            b01_idx = 0 * p + 2*u + 1
            b10_idx = 1 * p + 2*u
            b11_idx = 1 * p + 2*u + 1
            
            if t == 0:
                V[b00_idx][c] += 1
                V[b11_idx][c] += 1
            elif t == 1:
                V[b00_idx][c] += 1
            elif t == 2:
                V[b01_idx][c] += 1
                V[b11_idx][c] -= 1
            elif t == 3:
                V[b10_idx][c] += 1
                V[b00_idx][c] -= 1
            elif t == 4:
                V[b11_idx][c] += 1
            elif t == 5:
                V[b00_idx][c] += 1
                V[b01_idx][c] += 1
            elif t == 6:
                V[b10_idx][c] += 1
                V[b11_idx][c] += 1
                
            # Fill W: C-forms
            # C indices for block u:
            # c00 -> index (2*u)*n + 0
            # c01 -> index (2*u+1)*n + 0  <-- Wait, W[k*n+i]
            # The output C is 2x12.
            # C[i, j] where i in {0,1}, j in {0..11}.
            # W index is k*n + i.
            # Here k is the column index of C (0..11), i is the row index of C (0..1).
            # So C[i, j] corresponds to W[j*n + i].
            
            c00_idx = (2*u) * n + 0
            c01_idx = (2*u+1) * n + 0
            c10_idx = (2*u) * n + 1
            c11_idx = (2*u+1) * n + 1
            
            # Strassen C-forms (standard):
            # C00 = M0 + M3 - M4 + M6
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M2 + M5
            
            if t == 0: # M0
                W[c00_idx][c] += 1
                W[c11_idx][c] += 1
            elif t == 1: # M1
                W[c10_idx][c] += 1
                W[c11_idx][c] -= 1
            elif t == 2: # M2
                W[c01_idx][c] += 1
                W[c11_idx][c] += 1
            elif t == 3: # M3
                W[c00_idx][c] += 1
                W[c10_idx][c] += 1
            elif t == 4: # M4
                W[c00_idx][c] -= 1
                W[c01_idx][c] += 1
            elif t == 5: # M5
                W[c11_idx][c] += 1
            elif t == 6: # M6
                W[c00_idx][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
