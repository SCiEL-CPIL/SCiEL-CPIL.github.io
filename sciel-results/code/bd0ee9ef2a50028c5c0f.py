"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 7 blocks of 2x2x2 (Strassen, rank 7)
    # 1 block of 2x2x1 (Schoolbook, rank 4)
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen components for 2x2x2
    # u, v, w are lists of length 4, 4, 4 respectively
    # Indices:
    # u: i*m+j
    # v: j*p+k (local p=2)
    # w: k*n+i (local n=2)
    
    strassen_u = [
        [1, 0, 0, 1], # M1: a00+a11
        [0, 0, 1, 1], # M2: a10+a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00+a01
        [-1, 0, 1, 0],# M6: a10-a00
        [0, 1, 0, -1] # M7: a01-a11
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # M1: b00+b11
        [1, 0, 0, 0], # M2: b00
        [0, 1, 0, -1],# M3: b01-b11
        [-1, 0, 1, 0],# M4: b10-b00
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00+b01
        [0, 0, 1, 1]  # M7: b10+b11
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1: c00+c11 (w0, w3)
        [0, 0, 1, -1],# M2: c10-c11 (w2, w3)
        [0, 0, 1, 1], # M3: c01+c11 (w2, w3) -- Wait, c01 is i=0,k=1 -> w2. c11 is i=1,k=1 -> w3.
        [1, 0, 1, 0], # M4: c00+c10 (w0, w2)
        [-1, 0, 1, 0],# M5: -c00+c01 (w0, w2)
        [0, 0, 0, 1], # M6: c11 (w3)
        [1, 0, 0, 0]  # M7: c00 (w0)
    ]
    
    # Verify Strassen W indices
    # c00: i=0, k=0 -> w[0]
    # c01: i=0, k=1 -> w[2]
    # c10: i=1, k=0 -> w[1] -- WAIT.
    # W index is k*n + i.
    # n=2.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    
    # Let's re-evaluate strassen_w based on c indices.
    # M1: c00(+1), c11(+1) -> w[0]=1, w[3]=1.
    # M2: c10(+1), c11(-1) -> w[1]=1, w[3]=-1.
    # M3: c01(+1), c11(+1) -> w[2]=1, w[3]=1.
    # M4: c00(+1), c10(+1) -> w[0]=1, w[1]=1.
    # M5: c00(-1), c01(+1) -> w[0]=-1, w[2]=1.
    # M6: c11(+1) -> w[3]=1.
    # M7: c00(+1) -> w[0]=1.
    
    strassen_w = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # 7 blocks of 2x2x2
    for block_idx in range(7):
        # k offsets
        k_offset = block_idx * 2
        
        for r in range(7):
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_val = strassen_u[r][i * 2 + j]
                    U[i * m + j][col] = u_val
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    v_val = strassen_v[r][j * 2 + k_local]
                    # Global k = k_offset + k_local
                    V[j * p + (k_offset + k_local)][col] = v_val
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    w_val = strassen_w[r][k_local * 2 + i]
                    # Global k = k_offset + k_local
                    W[(k_offset + k_local) * n + i][col] = w_val
            
            col += 1

    # 1 block of 2x2x1
    k_offset = 14
    
    # Schoolbook components
    # 1. a00 b00 -> c00
    U[0][col] = 1
    V[0 * p + k_offset][col] = 1
    W[k_offset * n + 0][col] = 1
    col += 1
    
    # 2. a01 b10 -> c00
    U[1][col] = 1
    V[1 * p + k_offset][col] = 1
    W[k_offset * n + 0][col] = 1
    col += 1
    
    # 3. a10 b00 -> c10
    U[2][col] = 1
    V[0 * p + k_offset][col] = 1
    W[k_offset * n + 1][col] = 1
    col += 1
    
    # 4. a11 b10 -> c10
    U[3][col] = 1
    V[1 * p + k_offset][col] = 1
    W[k_offset * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
