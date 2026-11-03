import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 189 (27 block multiplications * 7 Strassen multiplications)
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U[r][u][v], V[r][v][w], W[r][u][w]
    # r in 0..6
    # u, v, w in 0..1
    
    U_strassen = [
        [[[0,0],[0,0]], [[0,0],[0,1]]],  # r=0: M1 (0,0)=1, (1,1)=1
        [[[0,0],[0,0]], [[0,1],[0,1]]],  # r=1: M2 (1,0)=1, (1,1)=1
        [[[1,0],[0,0]], [[0,0],[0,0]]],  # r=2: M3 (0,0)=1
        [[[0,0],[0,0]], [[0,0],[0,1]]],  # r=3: M4 (1,1)=1
        [[[1,1],[0,0]], [[0,0],[0,0]]],  # r=4: M5 (0,0)=1, (0,1)=1
        [[[-1,0],[0,0]], [[0,1],[0,0]]], # r=5: M6 (1,0)=1, (0,0)=-1
        [[[0,1],[0,0]], [[0,0],[0,-1]]]  # r=6: M7 (0,1)=1, (1,1)=-1
    ]
    # Correction: U_strassen[r][u][v]
    # r=0: (0,0)=1, (1,1)=1
    # r=1: (1,0)=1, (1,1)=1
    # r=2: (0,0)=1
    # r=3: (1,1)=1
    # r=4: (0,0)=1, (0,1)=1
    # r=5: (1,0)=1, (0,0)=-1
    # r=6: (0,1)=1, (1,1)=-1
    
    U_s = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]]
    ]
    
    V_strassen = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,0],[1,-1]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ]
    # Correction: V_strassen[r][v][w]
    # r=0: (0,0)=1, (1,1)=1
    # r=1: (0,0)=1
    # r=2: (1,0)=1, (1,1)=-1
    # r=3: (0,1)=1, (0,0)=-1
    # r=4: (1,1)=1
    # r=5: (0,0)=1, (0,1)=1
    # r=6: (1,0)=1, (1,1)=1
    
    V_s = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,0],[1,-1]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ]
    
    W_strassen = [
        [1,0,0,1], # r=0: Z00=1, Z11=1
        [0,1,0,-1], # r=1: Z10=1, Z11=-1
        [0,0,1,1], # r=2: Z01=1, Z11=1
        [1,1,0,0], # r=3: Z00=1, Z10=1
        [-1,0,1,0], # r=4: Z00=-1, Z01=1
        [0,0,0,1], # r=5: Z11=1
        [1,0,0,0] # r=6: Z00=1
    ]
    # Correction: W_strassen[r][k*2+i]
    # r=0: Z00(idx0)=1, Z11(idx3)=1
    # r=1: Z10(idx1)=1, Z11(idx3)=-1
    # r=2: Z01(idx2)=1, Z11(idx3)=1
    # r=3: Z00(idx0)=1, Z10(idx1)=1
    # r=4: Z00(idx0)=-1, Z01(idx2)=1
    # r=5: Z11(idx3)=1
    # r=6: Z00(idx0)=1
    
    W_s = [
        [1,0,0,1],
        [0,1,0,-1],
        [0,0,1,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,0]
    ]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for r in range(7):
                    # U part
                    for u in range(2):
                        for v in range(2):
                            val = U_s[r][u][v]
                            if val != 0:
                                idx_U = (I * 2 + u) * 6 + (J * 2 + v)
                                U[idx_U][col] += val
                    # V part
                    for v in range(2):
                        for w in range(2):
                            val = V_s[r][v][w]
                            if val != 0:
                                idx_V = (J * 2 + v) * 6 + (K * 2 + w)
                                V[idx_V][col] += val
                    # W part
                    for u in range(2):
                        for w in range(2):
                            # W_s index is k*2+i, here k=w, i=u
                            # So index is w*2 + u
                            val = W_s[r][w * 2 + u]
                            if val != 0:
                                idx_W = (K * 2 + w) * 6 + (I * 2 + u)
                                W[idx_W][col] += val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
