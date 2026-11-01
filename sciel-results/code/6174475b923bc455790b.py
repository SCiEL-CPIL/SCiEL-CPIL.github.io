"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen-like decomposition for 2x2 blocks
    # Rank = 7 * ceil(p/2) = 7 * 7 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for A (2x2)
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    
    A_forms = [
        [1, 0, 0, 1],  # S1
        [0, 0, 1, 1],  # S2
        [1, 0, 0, 0],  # S3
        [0, 0, 0, 1],  # S4
        [1, 1, 0, 0],  # S5
        [-1, 0, 1, 0], # S6
        [0, 1, 0, -1]  # S7
    ]
    
    # For each pair of columns (k, k+1)
    # B indices for pair (k, k+1):
    # B00: (0, k), B01: (0, k+1)
    # B10: (1, k), B11: (1, k+1)
    
    # Strassen products for B pair:
    # P1 = S1 * (B00 + B11)
    # P2 = S2 * B00
    # P3 = S3 * (B01 - B11)
    # P4 = S4 * (B10 - B00)
    # P5 = S5 * B11
    # P6 = S6 * (B00 + B01)
    # P7 = S7 * (B10 + B11)
    
    # Contributions to C (2x2 block for cols k, k+1):
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P7
    # C11 = P1 - P2 + P3 + P6
    
    # C indices for pair (k, k+1):
    # C00: (k, 0) -> index k*2 + 0
    # C01: (k+1, 0) -> index (k+1)*2 + 0
    # C10: (k, 1) -> index k*2 + 1
    # C11: (k+1, 1) -> index (k+1)*2 + 1
    
    for pair_idx in range(7):
        k = pair_idx * 2
        # B indices
        b00 = 0 * p + k
        b01 = 0 * p + (k + 1)
        b10 = 1 * p + k
        b11 = 1 * p + (k + 1)
        
        # C indices
        c00 = k * n + 0
        c01 = (k + 1) * n + 0
        c10 = k * n + 1
        c11 = (k + 1) * n + 1
        
        # Define B forms for this pair
        # P1: B00 + B11
        # P2: B00
        # P3: B01 - B11
        # P4: B10 - B00
        # P5: B11
        # P6: B00 + B01
        # P7: B10 + B11
        
        B_forms = [
            {b00: 1, b11: 1},
            {b00: 1},
            {b01: 1, b11: -1},
            {b10: 1, b00: -1},
            {b11: 1},
            {b00: 1, b01: 1},
            {b10: 1, b11: 1}
        ]
        
        # C forms for each P
        # P1 -> C00 +1, C11 +1
        # P2 -> C10 +1, C11 -1
        # P3 -> C01 +1, C11 +1
        # P4 -> C00 +1
        # P5 -> C01 +1, C00 -1
        # P6 -> C11 +1
        # P7 -> C10 +1, C00 -1
        
        C_forms = [
            {c00: 1, c11: 1},
            {c10: 1, c11: -1},
            {c01: 1, c11: 1},
            {c00: 1},
            {c01: 1, c00: -1},
            {c11: 1},
            {c10: 1, c00: -1}
        ]
        
        for s in range(7):
            # Fill U
            for i, val in enumerate(A_forms[s]):
                if val != 0:
                    U[i][col] = val
            
            # Fill V
            for idx, val in B_forms[s].items():
                V[idx][col] = val
            
            # Fill W
            for idx, val in C_forms[s].items():
                W[idx][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
