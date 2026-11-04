"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # We'll partition A (3x4) into A1 (2x4) and A2 (1x4)
    # Partition B (4x8) into B1 (4x3) and B2 (4x5)
    #
    # Block 1: (2x4)*(4x3) -> 2x3x4  (m1=2, p1=3, k=4)
    # Block 2: (2x4)*(4x5) -> 2x5x4  (m2=2, p2=5, k=4)
    # Block 3: (1x4)*(4x3) -> 1x3x4  (m3=1, p3=3, k=4)
    # Block 4: (1x4)*(4x5) -> 1x5x4  (m4=1, p4=5, k=4)
    
    # For each block, we use schoolbook decomposition
    # Block with dimensions (rows x cols) * (cols x depth)
    # A_block is (rows x cols), B_block is (cols x depth)
    # Output C_block is (rows x depth)
    # Tensor entries: U[i*cols + col, c], V[col*depth + d, c], W[d*rows + i, c]
    
    # Block 1: 2x4 * 4x3
    r1, k1, d1 = 2, 4, 3  # rows=2, shared=4, depth=3
    rank1 = r1 * k1 * d1
    # U block: (r1*k1) x rank1
    U1 = [[0]*rank1 for _ in range(r1*k1)]
    V1 = [[0]*rank1 for _ in range(k1*d1)]
    W1 = [[0]*rank1 for _ in range(d1*r1)]
    col = 0
    for i in range(r1):
        for a in range(k1):
            for b in range(d1):
                U1[i*k1+a][col] = 1
                V1[a*d1+b][col] = 1
                W1[b*r1+i][col] = 1
                col += 1
    
    # Block 2: 2x4 * 4x5
    r2, k2, d2 = 2, 4, 5
    rank2 = r2 * k2 * d2
    U2 = [[0]*rank2 for _ in range(r2*k2)]
    V2 = [[0]*rank2 for _ in range(k2*d2)]
    W2 = [[0]*rank2 for _ in range(d2*r2)]
    col = 0
    for i in range(r2):
        for a in range(k2):
            for b in range(d2):
                U2[i*k2+a][col] = 1
                V2[a*d2+b][col] = 1
                W2[b*r2+i][col] = 1
                col += 1
    
    # Block 3: 1x4 * 4x3
    r3, k3, d3 = 1, 4, 3
    rank3 = r3 * k3 * d3
    U3 = [[0]*rank3 for _ in range(r3*k3)]
    V3 = [[0]*rank3 for _ in range(k3*d3)]
    W3 = [[0]*rank3 for _ in range(d3*r3)]
    col = 0
    for i in range(r3):
        for a in range(k3):
            for b in range(d3):
                U3[i*k3+a][col] = 1
                V3[a*d3+b][col] = 1
                W3[b*r3+i][col] = 1
                col += 1
    
    # Block 4: 1x4 * 4x5
    r4, k4, d4 = 1, 4, 5
    rank4 = r4 * k4 * d4
    U4 = [[0]*rank4 for _ in range(r4*k4)]
    V4 = [[0]*rank4 for _ in range(k4*d4)]
    W4 = [[0]*rank4 for _ in range(d4*r4)]
    col = 0
    for i in range(r4):
        for a in range(k4):
            for b in range(d4):
                U4[i*k4+a][col] = 1
                V4[a*d4+b][col] = 1
                W4[b*r4+i][col] = 1
                col += 1
    
    total_rank = rank1 + rank2 + rank3 + rank4
    U = [[0]*total_rank for _ in range(n*m)]  # n*m = 12
    V = [[0]*total_rank for _ in range(m*p)]  # m*p = 32
    W = [[0]*total_rank for _ in range(p*n)]  # p*n = 24
    
    # Map block indices to global indices
    # A is 3x4, split into rows [0,1] (block1,2) and row [2] (block3,4)
    # B is 4x8, split into cols [0,1,2] (blocks 1,3) and cols [3,4,5,6,7] (blocks 2,4)
    # C is 3x8, output[i,j] where i in 0..2, j in 0..7
    # W index: k*n + i where k is output col index (0..7), i is output row index (0..2)
    
    # Block 1: A rows 0,1; B cols 0,1,2; Output rows 0,1; Output cols 0,1,2
    # U1 local (i,a) -> global U[(i)*4 + a]  where i in 0,1; a in 0,1,2,3
    # V1 local (a,b) -> global V[a*8 + b]   where a in 0,1,2,3; b in 0,1,2
    # W1 local (b,i) -> global W[b*3 + i]   where b in 0,1,2; i in 0,1
    
    # Block 2: A rows 0,1; B cols 3,4,5,6,7; Output rows 0,1; Output cols 3,4,5,6,7
    # U2 local (i,a) -> global U[i*4 + a]
    # V2 local (a,b) -> global V[a*8 + (3+b)]  where b in 0..4 maps to cols 3..7
    # W2 local (b,i) -> global W[(3+b)*3 + i]
    
    # Block 3: A row 2; B cols 0,1,2; Output row 2; Output cols 0,1,2
    # U3 local (i,a) -> global U[2*4 + a] = U[8+a]  where i=0 only, a in 0..3
    # V3 local (a,b) -> global V[a*8 + b]
    # W3 local (b,i) -> global W[b*3 + 2]  where i=0 only
    
    # Block 4: A row 2; B cols 3,4,5,6,7; Output row 2; Output cols 3,4,5,6,7
    # U4 local (i,a) -> global U[8+a]
    # V4 local (a,b) -> global V[a*8 + (3+b)]
    # W4 local (b,i) -> global W[(3+b)*3 + 2]
    
    col_offset = 0
    # Copy block 1
    for i in range(r1):
        for a in range(k1):
            for c in range(rank1):
                if U1[i*k1+a][c] != 0:
                    U[i*m + a][col_offset + c] += U1[i*k1+a][c]
    for a in range(k1):
        for b in range(d1):
            for c in range(rank1):
                if V1[a*d1+b][c] != 0:
                    V[a*p + b][col_offset + c] += V1[a*d1+b][c]
    for b in range(d1):
        for i in range(r1):
            for c in range(rank1):
                if W1[b*r1+i][c] != 0:
                    W[b*n + i][col_offset + c] += W1[b*r1+i][c]
    col_offset += rank1
    
    # Copy block 2
    for i in range(r2):
        for a in range(k2):
            for c in range(rank2):
                if U2[i*k2+a][c] != 0:
                    U[i*m + a][col_offset + c] += U2[i*k2+a][c]
    for a in range(k2):
        for b in range(d2):
            for c in range(rank2):
                if V2[a*d2+b][c] != 0:
                    V[a*p + (3+b)][col_offset + c] += V2[a*d2+b][c]
    for b in range(d2):
        for i in range(r2):
            for c in range(rank2):
                if W2[b*r2+i][c] != 0:
                    W[(3+b)*n + i][col_offset + c] += W2[b*r2+i][c]
    col_offset += rank2
    
    # Copy block 3
    for i in range(r3):
        for a in range(k3):
            for c in range(rank3):
                if U3[i*k3+a][c] != 0:
                    U[(r1+i)*m + a][col_offset + c] += U3[i*k3+a][c]
    for a in range(k3):
        for b in range(d3):
            for c in range(rank3):
                if V3[a*d3+b][c] != 0:
                    V[a*p + b][col_offset + c] += V3[a*d3+b][c]
    for b in range(d3):
        for i in range(r3):
            for c in range(rank3):
                if W3[b*r3+i][c] != 0:
                    W[b*n + (r1+i)][col_offset + c] += W3[b*r3+i][c]
    col_offset += rank3
    
    # Copy block 4
    for i in range(r4):
        for a in range(k4):
            for c in range(rank4):
                if U4[i*k4+a][c] != 0:
                    U[(r1+i)*m + a][col_offset + c] += U4[i*k4+a][c]
    for a in range(k4):
        for b in range(d4):
            for c in range(rank4):
                if V4[a*d4+b][c] != 0:
                    V[a*p + (3+b)][col_offset + c] += V4[a*d4+b][c]
    for b in range(d4):
        for i in range(r4):
            for c in range(rank4):
                if W4[b*r4+i][c] != 0:
                    W[(3+b)*n + (r1+i)][col_offset + c] += W4[b*r4+i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
