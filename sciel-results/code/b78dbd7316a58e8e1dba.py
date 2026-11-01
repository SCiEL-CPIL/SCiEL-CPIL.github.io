"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of 2x2 * 2x14 as 7 independent 2x2 multiplications
    # Each 2x2 multiplication uses Strassen's rank-7 algorithm
    # Total rank = 7 * 7 = 49 (vs schoolbook 56)
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # Strassen's 7 products for a 2x2 * 2x2 multiplication
    # U, V are 4x7 matrices for each block, W is 4x7
    # Row indices: U: i*2+j (i=0,1; j=0,1), V: j*2+k (j=0,1; k=0,1)
    # W: k*2+i (k=0,1; i=0,1)
    
    # Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For each Strassen product r in 0..6:
    # U_r[4], V_r[4], W_r[4]
    
    # Product 0: M1 = (a+d)(e+h)
    # U: a at idx0, d at idx3 -> [1,0,0,1]
    # V: e at idx0, h at idx3 -> [1,0,0,1]
    # C00 += M1, C11 += M1
    # W: C00 at idx0, C11 at idx3 -> [1,0,0,1]
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 1, 1, 1]   # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+=1, C11+=1
        [0, 1, 0, -1], # M2: C10+=1, C11+=-1
        [-1, 0, 1, 0], # M3: C00+=-1, C01+=1
        [1, 1, 0, 0],  # M4: C00+=1, C10+=1
        [-1, 0, 1, 0], # M5: C00+=-1, C01+=1
        [0, 0, 0, 1],  # M6: C11+=1
        [1, 0, 0, 1]   # M7: C00+=1, C11+=1
    ]
    
    # Build global matrices
    num_blocks = p // 2  # 14 // 2 = 7
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            # U: same for all blocks (acts on the 2x2 input matrix A)
            for idx in range(4):
                U[idx][col] = strassen_U[r][idx]
            
            # V: acts on block 'block' of the 2x14 matrix B
            # V row index: j*2 + k, where k is offset by block*2
            for idx in range(4):
                v_row = idx + block * 2  # j*2+k where k = idx%2 + block*2
                V[v_row][col] = strassen_V[r][idx]
            
            # W: acts on block 'block' of the output 2x14 matrix C
            # W row index: k*n + i, where k is offset by block*2
            for idx in range(4):
                w_row = idx + block * 2  # k*2+i where k = idx%2 + block*2
                W[w_row][col] = strassen_W[r][idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
