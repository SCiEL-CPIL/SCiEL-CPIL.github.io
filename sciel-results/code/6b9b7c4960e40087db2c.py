"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 blocks of Strassen (rank 7 each)
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients (A parts)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6 (A10-A00)
        [0, 1, 0, -1]  # M7 (A01-A11)
    ]
    
    # V coefficients (B parts)
    # For a block p, B indices are:
    # B00 -> j=0, k=2p -> idx 8*0 + 2p
    # B01 -> j=0, k=2p+1 -> idx 8*0 + 2p+1
    # B10 -> j=1, k=2p -> idx 8*1 + 2p
    # B11 -> j=1, k=2p+1 -> idx 8*1 + 2p+1
    
    # W coefficients (C parts)
    # C indices: W[k*n + i]
    # Block p outputs:
    # C00 -> i=0, k=2p -> idx 2*2p + 0
    # C01 -> i=0, k=2p+1 -> idx 2*(2p+1) + 0
    # C10 -> i=1, k=2p -> idx 2*2p + 1
    # C11 -> i=1, k=2p+1 -> idx 2*(2p+1) + 1
    
    # W contributions for each M_l
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 += -1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += -1, C01 += 1
    # M5: C00 += 1
    # M6: C11 += 1
    # M7: C00 += -1, C10 += 1
    
    W_coeffs = [
        # M1
        [(0, 1), (3, 1)], # (i, k_offset_in_block) -> no, let's use explicit indices later
        # M2
        [(2, 1), (3, -1)],
        # M3
        [(1, 1), (3, 1)],
        # M4
        [(0, -1), (1, 1)],
        # M5
        [(0, 1)],
        # M6
        [(3, 1)],
        # M7
        [(0, -1), (2, 1)]
    ]
    
    # V coeffs structure: list of (j, k_offset, coeff)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)], # M1: B00, B11
        [(0, 0, 1)],            # M2: B00
        [(1, 0, 1), (1, 1, -1)],# M3: B10, B11
        [(0, 1, 1), (0, 0, -1)],# M4: B01, B00
        [(1, 1, 1)],            # M5: B11
        [(0, 0, 1), (0, 1, 1)], # M6: B00, B01
        [(1, 0, 1), (1, 1, 1)]  # M7: B10, B11
    ]
    
    for p_block in range(4):
        for l in range(7):
            col = p_block * 7 + l
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = U_coeffs[l][i*m + j]
            
            # Fill V
            for j, k_off, coeff in V_coeffs[l]:
                k_idx = 2 * p_block + k_off
                V[j*p + k_idx][col] = coeff
            
            # Fill W
            for i, coeff in W_coeffs[l]:
                # i is row of C (0 or 1)
                # We need to map M_l to C indices.
                # The W_coeffs above were encoded as (i, coeff) but I need to be careful with the mapping.
                # Let's re-verify the W mapping logic.
                pass
                
    # Re-implementing W filling more explicitly to avoid confusion
    W = [[0] * rank for _ in range(p * n)]
    for p_block in range(4):
        for l in range(7):
            col = p_block * 7 + l
            # M1
            if l == 0:
                W[2*(2*p_block) + 0][col] = 1     # C00
                W[2*(2*p_block+1) + 1][col] = 1   # C11
            elif l == 1: # M2
                W[2*(2*p_block) + 1][col] = 1     # C10
                W[2*(2*p_block+1) + 1][col] = -1  # C11
            elif l == 2: # M3
                W[2*(2*p_block+1) + 0][col] = 1   # C01
                W[2*(2*p_block+1) + 1][col] = 1   # C11
            elif l == 3: # M4
                W[2*(2*p_block) + 0][col] = -1    # C00
                W[2*(2*p_block+1) + 0][col] = 1   # C01
            elif l == 4: # M5
                W[2*(2*p_block) + 0][col] = 1     # C00
            elif l == 5: # M6
                W[2*(2*p_block+1) + 1][col] = 1   # C11
            elif l == 6: # M7
                W[2*(2*p_block) + 0][col] = -1    # C00
                W[2*(2*p_block) + 1][col] = 1     # C10

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
