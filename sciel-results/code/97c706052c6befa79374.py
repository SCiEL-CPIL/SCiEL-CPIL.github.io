"""Matrix multiplication 2x2 by 2x6 - Strassen-based block decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of the output.
    # B is 2x6, so we split it into three 2x2 blocks: B0, B1, B2.
    # A is 2x2.
    # C is 2x6, split into three 2x2 blocks: C0, C1, C2.
    # C_s = A * B_s for s=0,1,2.
    # Strassen rank for 2x2 is 7.
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen intermediates for A (same for all blocks)
    # A_1 = A00 + A11
    # A_2 = A10 + A11
    # A_3 = A00
    # A_4 = A11
    # A_5 = A00 + A01
    # A_6 = A10 - A00
    # A_7 = A01 - A11
    
    # Indices for U (A): i*m + j
    # 0:00, 1:01, 2:10, 3:11
    
    # Strassen intermediates for B_s
    # B_s is 2x2 with entries B[0, 2s], B[0, 2s+1], B[1, 2s], B[1, 2s+1]
    # B_1 = B00 + B11 (relative to block)
    # B_2 = B00
    # B_3 = B10 - B11
    # B_4 = B01 - B00
    # B_5 = B11
    # B_6 = B00 + B01
    # B_7 = B10 + B11
    
    # Indices for V (B): j*p + k
    # Block s uses k=2s, 2s+1.
    # j=0: 2s, 2s+1
    # j=1: 6+2s, 6+2s+1
    
    # Reconstruction of C_s
    # C00 (i=0, k=2s) = P1 + P4 - P5 + P7
    # C01 (i=0, k=2s+1) = P3 + P5
    # C10 (i=1, k=2s) = P2 + P4
    # C11 (i=1, k=2s+1) = P1 - P2 + P3 + P6
    
    # Indices for W (C): k*n + i
    # Block s uses k=2s, 2s+1.
    # i=0: 4s, 4s+2
    # i=1: 4s+1, 4s+3
    
    for s in range(3):
        for r in range(7):
            col = s * 7 + r
            
            # Set U (depends only on r)
            if r == 0: # A1
                U[0][col] = 1
                U[3][col] = 1
            elif r == 1: # A2
                U[2][col] = 1
                U[3][col] = 1
            elif r == 2: # A3
                U[0][col] = 1
            elif r == 3: # A4
                U[3][col] = 1
            elif r == 4: # A5
                U[0][col] = 1
                U[1][col] = 1
            elif r == 5: # A6
                U[2][col] = 1
                U[0][col] = -1
            elif r == 6: # A7
                U[1][col] = 1
                U[3][col] = -1
            
            # Set V (depends on s and r)
            # B indices:
            # b00 = V[2s], b01 = V[2s+1]
            # b10 = V[6+2s], b11 = V[7+2s]
            
            if r == 0: # B1 = b00 + b11
                V[2*s][col] = 1
                V[7+2*s][col] = 1
            elif r == 1: # B2 = b00
                V[2*s][col] = 1
            elif r == 2: # B3 = b10 - b11
                V[6+2*s][col] = 1
                V[7+2*s][col] = -1
            elif r == 3: # B4 = b01 - b00
                V[2*s+1][col] = 1
                V[2*s][col] = -1
            elif r == 4: # B5 = b11
                V[7+2*s][col] = 1
            elif r == 5: # B6 = b00 + b01
                V[2*s][col] = 1
                V[2*s+1][col] = 1
            elif r == 6: # B7 = b10 + b11
                V[6+2*s][col] = 1
                V[7+2*s][col] = 1
            
            # Set W (depends on s and r)
            # C indices:
            # c00 = W[4s], c01 = W[4s+2]
            # c10 = W[4s+1], c11 = W[4s+3]
            
            if r == 0: # P1 contributes to C00 (+1) and C11 (+1)
                W[4*s][col] += 1
                W[4*s+3][col] += 1
            elif r == 1: # P2 contributes to C10 (+1) and C11 (-1)
                W[4*s+1][col] += 1
                W[4*s+3][col] += -1
            elif r == 2: # P3 contributes to C01 (+1) and C11 (+1)
                W[4*s+2][col] += 1
                W[4*s+3][col] += 1
            elif r == 3: # P4 contributes to C00 (+1) and C10 (+1)
                W[4*s][col] += 1
                W[4*s+1][col] += 1
            elif r == 4: # P5 contributes to C00 (-1) and C01 (+1)
                W[4*s][col] += -1
                W[4*s+2][col] += 1
            elif r == 5: # P6 contributes to C11 (+1)
                W[4*s+3][col] += 1
            elif r == 6: # P7 contributes to C00 (+1)
                W[4*s][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
