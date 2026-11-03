import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We split p=8 into two blocks of size 4.
    # Each block is a 2x3x4 multiplication.
    # Schoolbook rank for 2x3x4 is 12.
    # Total rank = 12 + 12 = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)] # 6 x 24
    V = [[0] * rank for _ in range(m * p)] # 24 x 24
    W = [[0] * rank for _ in range(p * n)] # 16 x 24
    
    # Block 1: k in 0..3
    # Block 2: k in 4..7
    
    col = 0
    for block in range(2):
        k_offset = block * 4
        for i in range(n):
            for j in range(m):
                for k in range(4):
                    # U index: i*m + j
                    # V index: j*p + (k_offset + k)
                    # W index: (k_offset + k)*n + i
                    
                    U[i * m + j][col] = 1
                    V[j * p + k_offset + k][col] = 1
                    W[(k_offset + k) * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
