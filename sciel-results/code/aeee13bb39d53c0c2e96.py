import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U: (n*m, R) -> (6, R)
    # V: (m*p, R) -> (24, R)
    # W: (p*n, R) -> (16, R)
    
    rank = 29
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # Indices for 2x2 matrix: 00, 01, 10, 11
    # u, v, w are vectors of length 4
    
    strassen_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11
        [0, 1, 0, 1],  # M3 -> C01, C11
        [1, 0, 1, 0],  # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0]   # M7 -> C00
    ]
    
    # Rank 0: Handle j=2 part (A_R B_B)
    # U rows for j=2: i=0 -> idx 2, i=1 -> idx 5
    # V rows for j=2: k=0..7 -> idx 16..23
    # W rows for k=0..7, i=0..1: idx 4k .. 4k+1
    
    for i in range(n):
        U[i * m + 2][0] = 1.0
        
    for k in range(p):
        V[2 * p + k][0] = 1.0
        
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1.0
            
    # Ranks 1..28: Handle j=0,1 part (A_L B_T) using Strassen on 4 blocks
    # A_L is 2x2 (i in 0..1, j in 0..1)
    # B_T is 2x8 (j in 0..1, k in 0..7)
    # C_1 is 2x8 (i in 0..1, k in 0..7)
    
    for block in range(4):
        # k_local 0,1 corresponds to k_global 2*block, 2*block+1
        for s in range(7):
            r = block * 7 + s + 1
            
            # Fill U
            # u_s is [00, 01, 10, 11]
            # U indices for j=0,1:
            # i=0, j=0 -> 0
            # i=0, j=1 -> 1
            # i=1, j=0 -> 3
            # i=1, j=1 -> 4
            U[0][r] = strassen_u[s][0]
            U[1][r] = strassen_u[s][1]
            U[3][r] = strassen_u[s][2]
            U[4][r] = strassen_u[s][3]
            
            # Fill V
            # v_s is [00, 01, 10, 11] for B
            # V indices for j=0,1 and k in block:
            # j=0, k=2b -> 0
            # j=0, k=2b+1 -> 1
            # j=1, k=2b -> 8
            # j=1, k=2b+1 -> 9
            # Note: V row index is j*p + k
            k0 = block * 2
            k1 = block * 2 + 1
            
            V[0 * p + k0][r] = strassen_v[s][0]
            V[0 * p + k1][r] = strassen_v[s][1]
            V[1 * p + k0][r] = strassen_v[s][2]
            V[1 * p + k1][r] = strassen_v[s][3]
            
            # Fill W
            # w_s is [00, 01, 10, 11] for C
            # W indices for k in block, i=0,1:
            # k=2b, i=0 -> 4b
            # k=2b, i=1 -> 4b+1
            # k=2b+1, i=0 -> 4b+2
            # k=2b+1, i=1 -> 4b+3
            # Note: W row index is k*n + i
            
            W[k0 * n + 0][r] = strassen_w[s][0]
            W[k0 * n + 1][r] = strassen_w[s][1]
            W[k1 * n + 0][r] = strassen_w[s][2]
            W[k1 * n + 1][r] = strassen_w[s][3]

    return {"U": U, "V": V, "W": W}
