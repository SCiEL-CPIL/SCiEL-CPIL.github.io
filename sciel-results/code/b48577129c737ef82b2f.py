"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=12, split into 6 blocks of 2
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2x2 block
    # U_s[i,j], V_s[j,k], W_s[k,i]
    # s from 0 to 6
    
    # M1: (A00+A11)(B00+B11), W: C00+=1, C11+=1
    U_0 = [[1, 0], [0, 1]] # A00=1, A11=1
    V_0 = [[1, 0], [0, 1]] # B00=1, B11=1
    W_0 = [[1, 0], [0, 1]] # C00=1, C11=1 (W[k,i]) -> W[0,0]=1, W[1,1]=1

    # M2: (A10+A11)B00, W: C10+=1, C11+=-1
    U_1 = [[0, 0], [1, 1]] # A10=1, A11=1
    V_1 = [[1, 0], [0, 0]] # B00=1
    W_1 = [[0, 0], [1, -1]] # C10=1 (W[1,0]), C11=-1 (W[1,1])

    # M3: A00(B01-B11), W: C01+=1, C10+=1
    U_2 = [[1, 0], [0, 0]] # A00=1
    V_2 = [[0, 1], [-1, 0]] # B01=1, B11=-1
    W_2 = [[0, 1], [1, 0]] # C01=1 (W[1,0]? No C01 is k=1,i=0 -> W[1,0]=1), C10=1 (W[0,1]=1)
    # Check indices: C[i,k]. W[k,i].
    # C01: i=0, k=1. W[1,0].
    # C10: i=1, k=0. W[0,1].
    # So W_2[1,0]=1, W_2[0,1]=1.
    # Matrix W_2:
    # k=0: [W[0,0], W[0,1]] = [0, 1]
    # k=1: [W[1,0], W[1,1]] = [1, 0]
    W_2 = [[0, 1], [1, 0]]

    # M4: A11(B10-B00), W: C00+=-1, C01+=1
    U_3 = [[0, 0], [0, 1]] # A11=1
    V_3 = [[-1, 0], [1, 0]] # B00=-1, B10=1
    W_3 = [[-1, 0], [1, 0]] # C00=-1 (W[0,0]), C01=1 (W[1,0])
    # k=0: [-1, 0]
    # k=1: [1, 0]

    # M5: (A00+A01)B11, W: C00+=1, C10+=-1
    U_4 = [[1, 1], [0, 0]] # A00=1, A01=1
    V_4 = [[0, 0], [0, 1]] # B11=1
    W_4 = [[1, 0], [0, -1]] # C00=1 (W[0,0]), C10=-1 (W[0,1])
    # k=0: [1, 0]
    # k=1: [0, 0] -> Wait. C10 is i=1, k=0. W[0,1].
    # W_4[0,0]=1, W_4[0,1]=-1.
    # W_4[1,0]=0, W_4[1,1]=0.
    W_4 = [[1, -1], [0, 0]]

    # M6: (A10-A00)(B00+B01), W: C10+=-1, C11+=1
    U_5 = [[-1, 0], [1, 0]] # A00=-1, A10=1
    V_5 = [[1, 1], [0, 0]] # B00=1, B01=1
    W_5 = [[0, -1], [0, 1]] # C10=-1 (W[0,1]), C11=1 (W[1,1])
    # W[0,0]=0, W[0,1]=-1
    # W[1,0]=0, W[1,1]=1

    # M7: (A01-A11)(B10+B11), W: C00+=-1, C01+=1
    U_6 = [[0, 1], [0, -1]] # A01=1, A11=-1
    V_6 = [[0, 1], [1, 1]] # B10=1, B11=1
    W_6 = [[-1, 0], [1, 0]] # C00=-1 (W[0,0]), C01=1 (W[1,0])
    # W[0,0]=-1, W[0,1]=0
    # W[1,0]=1, W[1,1]=0

    # Store in lists for easy access
    U_blocks = [U_0, U_1, U_2, U_3, U_4, U_5, U_6]
    V_blocks = [V_0, V_1, V_2, V_3, V_4, V_5, V_6]
    W_blocks = [W_0, W_1, W_2, W_3, W_4, W_5, W_6]

    # Fill global matrices
    for g in range(num_blocks):
        for s in range(7):
            c = g * 7 + s
            # U is independent of block
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = U_blocks[s][i][j]
            
            # V and W depend on block g
            for j in range(m):
                for k_loc in range(2):
                    k_global = g * 2 + k_loc
                    V[j * p + k_global][c] = V_blocks[s][j][k_loc]
                    
            for i in range(n):
                for k_loc in range(2):
                    k_global = g * 2 + k_loc
                    W[k_global * n + i][c] = W_blocks[s][k_loc][i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
