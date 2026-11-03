"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2x2 Strassen decomposition
    U2 = [[1, 0, 1, 0, 1, -1, 0],
          [0, 0, 0, 0, 1, 0, -1],
          [0, 1, 0, 0, 0, 1, 0],
          [1, 1, 0, 1, 0, 0, 1]]
    V2 = [[1, 1, 0, -1, 0, 1, 0],
          [0, 0, -1, 0, 0, 1, 0],
          [0, 0, 0, 1, 0, 0, 1],
          [1, 0, 1, 0, 1, 0, 1]]
    W2 = [[1, 0, 0, 1, -1, 0, 1],
          [0, 0, 1, 0, 1, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [1, -1, 1, 0, 0, 1, 0]]
    
    # Build 4x4x4 decomposition via tensor product
    rank444 = 7 * 7  # 49
    U444 = [[0] * rank444 for _ in range(16)]
    V444 = [[0] * rank444 for _ in range(16)]
    W444 = [[0] * rank444 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            # Index in 4x4x4
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            
                            # Tensor product index
                            col = (i1 * 2 + i2) * 7 + (j1 * 2 + j2)  # This is wrong
                            
                            # Correct tensor product:
                            # The 4x4 matrix A has indices (i1,i2) where i1 is block row, i2 is row within block
                            # U444[i1*4 + i2, col] = U2[i1, col1] * U2[i2, col2]
                            # where col = col1 * 7 + col2
                            pass
    
    # Let me redo this properly
    U444 = [[0] * 49 for _ in range(16)]
    V444 = [[0] * 49 for _ in range(16)]
    W444 = [[0] * 49 for _ in range(16)]
    
    for a1 in range(2):
        for a2 in range(2):
            for b1 in range(2):
                for b2 in range(2):
                    for c1 in range(2):
                        for c2 in range(2):
                            # A index: a1*2 + a2 (row), b1*2 + b2 (col)
                            # B index: b1*2 + b2 (row), c1*2 + c2 (col)
                            # C index: a1*2 + a2 (row), c1*2 + c2 (col)
                            
                            a_idx = a1 * 2 + a2
                            b_idx = b1 * 2 + b2
                            c_idx = c1 * 2 + c2
                            
                            # Tensor product column: (a1*2+b1) * 7 + (a2*2+b2) ... no
                            # The tensor product of two 2x2x2 tensors:
                            # Column index = col1 * 7 + col2 where col1 is from first 2x2x2, col2 from second
                            # U444[a_idx*4 + b_idx, col] = U2[a1, col1] * U2[a2, col2] ... this is getting confusing
                            
                            pass
    
    # I'm overcomplicating this. Let me use a simpler approach.
    # For now, just use schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
