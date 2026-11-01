"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen 2x2x2 decomposition
    # U_s[i*m+j], V_s[j*2+k], W_s[k*n+i] for 2x2 case
    strassen_U = [
        [1, 0, 0, 1],   # s=0: a+d
        [0, 0, 1, 1],   # s=1: c+d
        [1, 0, 0, 0],   # s=2: a
        [0, 0, 0, 1],   # s=3: d
        [1, 1, 0, 0],   # s=4: a+b
        [-1, 0, 1, 0],  # s=5: c-a
        [0, 1, 0, -1],  # s=6: b-d
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # s=0: e+h
        [1, 0, 0, 0],   # s=1: e
        [0, 1, -1, 0],  # s=2: f-h
        [-1, 0, 1, 0],  # s=3: g-e
        [0, 0, 0, 1],   # s=4: h
        [1, 1, 0, 0],   # s=5: e+f
        [0, 0, 1, 1],   # s=6: g+h
    ]
    
    strassen_W = [
        [1, 0, 0, 1],   # s=0: W[0,0]=1, W[1,1]=1
        [0, 1, 0, -1],  # s=1: W[0,1]=1, W[1,1]=-1
        [0, 0, 1, 1],   # s=2: W[1,0]=1, W[1,1]=1
        [1, 1, 0, 0],   # s=3: W[0,0]=1, W[0,1]=1
        [-1, 0, 1, 0],  # s=4: W[0,0]=-1, W[1,0]=1
        [0, 0, 0, 1],   # s=5: W[1,1]=1
        [1, 0, 0, 0],   # s=6: W[0,0]=1
    ]
    
    rank = num_blocks * 7  # 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        for s in range(7):
            r = b * 7 + s
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = strassen_U[s][i * m + j]
            for j in range(m):
                for kk in range(2):
                    col = 2 * b + kk
                    V[j * p + col][r] = strassen_V[s][j * 2 + kk]
            for i in range(n):
                for kk in range(2):
                    col = 2 * b + kk
                    W[col * n + i][r] = strassen_W[s][kk * n + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
