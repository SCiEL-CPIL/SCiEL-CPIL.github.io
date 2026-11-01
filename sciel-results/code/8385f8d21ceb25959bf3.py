"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-7 decomposition for each 2x2x2 block.
    # The 2x2 x 2x10 multiplication tensor is a direct sum of five 2x2x2 blocks.
    # Each block uses the standard Strassen decomposition of rank 7.
    # Total rank = 5 * 7 = 35.
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2
    # M1 = (a00 - a10)(b10 + b11)
    # M2 = (a00 + a01)(b00)
    # M3 = (a10 + a11)(b11)
    # M4 = (a01 - a00)(b00 + b10)
    # M5 = (a00 + a10)(b01 - b11)
    # M6 = (a10 - a11)(b01 + b11)  -- wait, let me use the standard form
    # Standard Strassen:
    # M1 = (a00 - a10) * (b10 + b11)
    # M2 = (a00 + a01) * b00
    # M3 = (a10 + a11) * b11
    # M4 = (a01 - a00) * (b00 + b10)
    # M5 = (a00 + a10) * (b01 - b11)
    # M6 = (a10 - a11) * (b01 + b11)
    # M7 = (a00 + a01 - a10 - a11) * (b00 + b11)
    # c00 = M5 + M4 + M1 - M7
    # c01 = M2 + M7
    # c10 = M3 + M5
    # c11 = M1 - M4 - M3 + M6
    
    # Let me verify with the standard Strassen formulas:
    # c00 = M1 - M4 + M5 - M7
    # c01 = M2 + M7
    # c10 = M3 + M5
    # c11 = M1 - M3 - M4 + M6
    
    # U matrix: shape (n*m, rank) = (4, 35)
    # Index: i*m + j where i in 0..1, j in 0..1
    # V matrix: shape (m*p, rank) = (20, 35)
    # Index: j*p + k where j in 0..1, k in 0..9
    # W matrix: shape (p*n, rank) = (20, 35)
    # Index: k*n + i where k in 0..9, i in 0..1
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        for r in range(rank_per_block):
            col = b * rank_per_block + r
            
            # Set U based on term r (same for all blocks)
            if r == 0:
                # M1 = (a00 - a10)(b10 + b11)
                U[0][col] = 1   # a00
                U[2][col] = -1  # a10
            elif r == 1:
                # M2 = (a00 + a01)(b00)
                U[0][col] = 1   # a00
                U[1][col] = 1   # a01
            elif r == 2:
                # M3 = (a10 + a11)(b11)
                U[2][col] = 1   # a10
                U[3][col] = 1   # a11
            elif r == 3:
                # M4 = (a01 - a00)(b00 + b10)
                U[0][col] = -1  # a00
                U[1][col] = 1   # a01
            elif r == 4:
                # M5 = (a00 + a10)(b01 - b11)
                U[0][col] = 1   # a00
                U[2][col] = 1   # a10
            elif r == 5:
                # M6 = (a10 - a11)(b01 + b11)
                U[2][col] = 1   # a10
                U[3][col] = -1  # a11
            elif r == 6:
                # M7 = (a00 + a01 - a10 - a11)(b00 + b11)
                U[0][col] = 1   # a00
                U[1][col] = 1   # a01
                U[2][col] = -1  # a10
                U[3][col] = -1  # a11
            
            # Set V based on term r and block b
            # Block b uses columns 2b and 2b+1 of B
            # V index: j*p + k
            if r == 0:
                # (b10 + b11): j=1, k=2b and j=1, k=2b+1
                V[1 * p + 2 * b][col] = 1
                V[1 * p + 2 * b + 1][col] = 1
            elif r == 1:
                # b00: j=0, k=2b
                V[0 * p + 2 * b][col] = 1
            elif r == 2:
                # b11: j=1, k=2b+1
                V[1 * p + 2 * b + 1][col] = 1
            elif r == 3:
                # (b00 + b10): j=0, k=2b and j=1, k=2b
                V[0 * p + 2 * b][col] = 1
                V[1 * p + 2 * b][col] = 1
            elif r == 4:
                # (b01 - b11): j=0, k=2b+1 and j=1, k=2b+1
                V[0 * p + 2 * b + 1][col] = 1
                V[1 * p + 2 * b + 1][col] = -1
            elif r == 5:
                # (b01 + b11): j=0, k=2b+1 and j=1, k=2b+1
                V[0 * p + 2 * b + 1][col] = 1
                V[1 * p + 2 * b + 1][col] = 1
            elif r == 6:
                # (b00 + b11): j=0, k=2b and j=1, k=2b+1
                V[0 * p + 2 * b][col] = 1
                V[1 * p + 2 * b + 1][col] = 1
            
            # Set W based on term r and block b
            # W index: k*n + i
            # Block b produces output columns 2b and 2b+1
            if r == 0:
                # M1: c00 += M1, c11 -= M1
                W[(2 * b) * n + 0][col] = 1    # c00
                W[(2 * b + 1) * n + 1][col] = -1  # c11
            elif r == 1:
                # M2: c01 += M2
                W[(2 * b + 1) * n + 0][col] = 1  # c01
            elif r == 2:
                # M3: c10 += M3, c11 -= M3
                W[(2 * b) * n + 1][col] = 1    # c10
                W[(2 * b + 1) * n + 1][col] = -1  # c11
            elif r == 3:
                # M4: c00 -= M4, c11 += M4
                W[(2 * b) * n + 0][col] = -1   # c00
                W[(2 * b + 1) * n + 1][col] = 1  # c11
            elif r == 4:
                # M5: c00 += M5, c10 += M5
                W[(2 * b) * n + 0][col] = 1    # c00
                W[(2 * b) * n + 1][col] = 1    # c10
            elif r == 5:
                # M6: c11 += M6
                W[(2 * b + 1) * n + 1][col] = 1  # c11
            elif r == 6:
                # M7: c00 -= M7, c01 += M7
                W[(2 * b) * n + 0][col] = -1   # c00
                W[(2 * b + 1) * n + 0][col] = 1  # c01
    
    # Convert to integer entries where possible
    for i in range(n * m):
        for c in range(total_rank):
            if U[i][c] == int(U[i][c]):
                U[i][c] = int(U[i][c])
    for i in range(m * p):
        for c in range(total_rank):
            if V[i][c] == int(V[i][c]):
                V[i][c] = int(V[i][c])
    for i in range(p * n):
        for c in range(total_rank):
            if W[i][c] == int(W[i][c]):
                W[i][c] = int(W[i][c])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
