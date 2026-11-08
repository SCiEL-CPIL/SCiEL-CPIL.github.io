"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition using 3x3 blocks of 2x2 matrices.
    # Outer level: Schoolbook 3x3 (27 terms).
    # Inner level: Strassen 2x2 (7 terms).
    # Total rank: 27 * 7 = 189.
    n, m, p = DIMENSIONS
    # We partition into 3x3 blocks of size 2x2.
    # Block indices i, j, k in 0..2.
    # Block size b = 2.
    b = 2
    num_blocks = n // b # 3
    
    rank = (num_blocks ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                # Indices for A_ij (rows 2i..2i+1, cols 2j..2j+1)
                # a00, a01, a10, a11
                a00_idx = (2 * i) * m + (2 * j)
                a01_idx = (2 * i) * m + (2 * j + 1)
                a10_idx = (2 * i + 1) * m + (2 * j)
                a11_idx = (2 * i + 1) * m + (2 * j + 1)
                
                # Indices for B_jk (rows 2j..2j+1, cols 2k..2k+1)
                # b00, b01, b10, b11
                b00_idx = (2 * j) * p + (2 * k)
                b01_idx = (2 * j) * p + (2 * k + 1)
                b10_idx = (2 * j + 1) * p + (2 * k)
                b11_idx = (2 * j + 1) * p + (2 * k + 1)
                
                # Indices for C_ik (rows 2i..2i+1, cols 2k..2k+1)
                # c00, c01, c10, c11
                c00_idx = (2 * i) * n + (2 * k)
                c01_idx = (2 * i) * n + (2 * k + 1)
                c10_idx = (2 * i + 1) * n + (2 * k)
                c11_idx = (2 * i + 1) * n + (2 * k + 1)
                
                # Strassen 7 products for 2x2
                # P1 = (a00 + a11)(b00 + b11)
                U[a00_idx][col] = 1
                U[a11_idx][col] = 1
                V[b00_idx][col] = 1
                V[b11_idx][col] = 1
                W[c00_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
                
                # P2 = (a10 + a11)b00
                U[a10_idx][col] = 1
                U[a11_idx][col] = 1
                V[b00_idx][col] = 1
                W[c10_idx][col] = 1
                W[c11_idx][col] = -1
                col += 1
                
                # P3 = a00(b01 - b11)
                U[a00_idx][col] = 1
                V[b01_idx][col] = 1
                V[b11_idx][col] = -1
                W[c01_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
                
                # P4 = a11(b10 - b00)
                U[a11_idx][col] = 1
                V[b10_idx][col] = 1
                V[b00_idx][col] = -1
                W[c00_idx][col] = 1
                W[c10_idx][col] = 1
                col += 1
                
                # P5 = (a00 + a01)b11
                U[a00_idx][col] = 1
                U[a01_idx][col] = 1
                V[b11_idx][col] = 1
                W[c00_idx][col] = -1
                W[c01_idx][col] = 1
                col += 1
                
                # P6 = (a10 - a00)(b00 + b01)
                U[a10_idx][col] = 1
                U[a00_idx][col] = -1
                V[b00_idx][col] = 1
                V[b01_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
                
                # P7 = (a01 - a11)(b10 + b11)
                U[a01_idx][col] = 1
                U[a11_idx][col] = -1
                V[b10_idx][col] = 1
                V[b11_idx][col] = 1
                W[c00_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
