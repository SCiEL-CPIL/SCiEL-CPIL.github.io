"""Rewritten program using Kronecker product factorization for 2x2 x 2x12 matrix multiplication."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We model this as a Kronecker product: (2x2 mult) ⊗ (1x6 scaling)
    # Using rank-9 decomposition for 2x2x2 and rank-6 for 1x6
    # Total rank = 9 * 6 = 54... but we can do better with shared structure
    
    # Actually, let's use the known optimal approach:
    # For 2x2 * 2x12, we can treat it as 6 independent 2x2 * 2x2 blocks
    # Each 2x2 * 2x2 needs rank 7 (Strassen)
    # So total rank = 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],   # M0: a00 + a11
        [0, 0, 1, 1],   # M1: a10 + a11
        [1, 0, 0, 0],   # M2: a00
        [0, 0, 0, 1],   # M3: a11
        [1, 1, 0, 0],   # M4: a00 + a01
        [-1, 0, 1, 0],  # M5: a10 - a00
        [0, 1, 0, -1]   # M6: a01 - a11
    ]
    
    # For each block u (0..5) and each Strassen term t (0..6)
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U: same for all blocks
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V: B is 2x12, so B indices are 0..11
            # Block u corresponds to columns 2u and 2u+1
            # B row 0, col 2u: index 2u
            # B row 0, col 2u+1: index 2u+1
            # B row 1, col 2u: index p + 2u = 12 + 2u
            # B row 1, col 2u+1: index p + 2u + 1 = 12 + 2u + 1
            
            # Strassen terms for B (2x2 block):
            # M0: b00 + b11 -> V[2u][c] += 1, V[12+2u+1][c] += 1
            # M1: b00 -> V[2u][c] += 1
            # M2: b01 - b11 -> V[2u+1][c] += 1, V[12+2u+1][c] -= 1
            # M3: b10 - b00 -> V[12+2u][c] += 1, V[2u][c] -= 1
            # M4: b11 -> V[12+2u+1][c] += 1
            # M5: b00 + b01 -> V[2u][c] += 1, V[2u+1][c] += 1
            # M6: b10 + b11 -> V[12+2u][c] += 1, V[12+2u+1][c] += 1
            
            if t == 0:
                V[2*u][c] += 1
                V[12 + 2*u + 1][c] += 1
            elif t == 1:
                V[2*u][c] += 1
            elif t == 2:
                V[2*u + 1][c] += 1
                V[12 + 2*u + 1][c] -= 1
            elif t == 3:
                V[12 + 2*u][c] += 1
                V[2*u][c] -= 1
            elif t == 4:
                V[12 + 2*u + 1][c] += 1
            elif t == 5:
                V[2*u][c] += 1
                V[2*u + 1][c] += 1
            elif t == 6:
                V[12 + 2*u][c] += 1
                V[12 + 2*u + 1][c] += 1
                
            # Fill W: Output C is 2x12, indices 0..23
            # C[i,k] where i is row (0,1) and k is column (0..11)
            # Index in W: k*n + i = k*2 + i
            # For block u, columns 2u and 2u+1:
            # C[0,2u]: index 2u*2 + 0 = 4u
            # C[0,2u+1]: index (2u+1)*2 + 0 = 4u+2
            # C[1,2u]: index 2u*2 + 1 = 4u+1
            # C[1,2u+1]: index (2u+1)*2 + 1 = 4u+3
            
            # Strassen output combinations:
            # C00 = M0 + M3 - M4 + M6
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M2 + M5
            
            if t == 0:  # M0
                W[4*u][c] += 1       # C00
                W[4*u + 3][c] += 1   # C11
            elif t == 1:  # M1
                W[4*u + 1][c] += 1   # C10
                W[4*u + 3][c] -= 1   # C11
            elif t == 2:  # M2
                W[4*u + 2][c] += 1   # C01
                W[4*u + 3][c] += 1   # C11
            elif t == 3:  # M3
                W[4*u][c] += 1       # C00
                W[4*u + 1][c] += 1   # C10
            elif t == 4:  # M4
                W[4*u][c] -= 1       # C00
                W[4*u + 2][c] += 1   # C01
            elif t == 5:  # M5
                W[4*u + 3][c] += 1   # C11
            elif t == 6:  # M6
                W[4*u][c] += 1       # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
