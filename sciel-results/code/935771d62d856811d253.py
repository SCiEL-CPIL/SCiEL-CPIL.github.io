import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # Strassen vectors for 2x2x2
    # Indices: 00, 01, 10, 11
    u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    num_blocks = p // 2 # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(num_blocks):
        for r in range(7):
            c = j * 7 + r
            
            # Fill U
            for i in range(4):
                U[i][c] = u[r][i]
                
            # Fill V
            # B block j occupies columns 2j, 2j+1
            # V index is k * p + l
            # k=0, l=2j -> 2j
            # k=0, l=2j+1 -> 2j+1
            # k=1, l=2j -> 16 + 2j
            # k=1, l=2j+1 -> 16 + 2j + 1
            V[2*j][c] = v[r][0]
            V[2*j+1][c] = v[r][1]
            V[16+2*j][c] = v[r][2]
            V[16+2*j+1][c] = v[r][3]
            
            # Fill W
            # C block j occupies columns 2j, 2j+1
            # W index is l * n + i
            # l=2j, i=0 -> 4j
            # l=2j, i=1 -> 4j+1
            # l=2j+1, i=0 -> 4j+2
            # l=2j+1, i=1 -> 4j+3
            W[4*j][c] = w[r][0]
            W[4*j+1][c] = w[r][1]
            W[4*j+2][c] = w[r][2]
            W[4*j+3][c] = w[r][3]
            
    return {"U": U, "V": V, "W": W}
