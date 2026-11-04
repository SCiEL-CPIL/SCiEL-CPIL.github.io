import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i*2+j][c] += val
        for (j, k), val in v_dict.items():
            V[j*2+k][c] += val
        for (k, i), val in w_dict.items():
            W[k*2+i][c] += val
    
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U1, V1, W1 = _strassen_2x2()
    
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U1[i*2+j][c] * V1[j*2+k][c] * W1[k*2+i][c] for c in range(7))
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        return _schoolbook_fallback()
    
    U444 = [[0]*49 for _ in range(16)]
    V444 = [[0]*49 for _ in range(16)]
    W444 = [[0]*49 for _ in range(16)]
    
    for i in range(2):
        for j in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    I, J = 2*i+i2, 2*j+j2
                    for c1 in range(7):
                        for c2 in range(7):
                            U444[I*4+J][c1*7+c2] += U1[i*2+j][c1] * U1[i2*2+j2][c2]
    
    for j in range(2):
        for k in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    J, K = 2*j+j2, 2*k+k2
                    for c1 in range(7):
                        for c2 in range(7):
                            V444[J*4+K][c1*7+c2] += V1[j*2+k][c1] * V1[j2*2+k2][c2]
    
    for k in range(2):
        for i in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    K, I = 2*k+k2, 2*i+i2
                    for c1 in range(7):
                        for c2 in range(7):
                            W444[K*4+I][c1*7+c2] += W1[k*2+i][c1] * W1[k2*2+i2][c2]
    
    valid444 = True
    for I in range(4):
        for J in range(4):
            for K in range(4):
                s = sum(U444[I*4+J][c] * V444[J*4+K][c] * W444[K*4+I][c] for c in range(49))
                if s != 1:
                    valid444 = False
                    break
            if not valid444: break
        if not valid444: break
    
    if not valid444:
        return _schoolbook_fallback()
    
    total_rank = 114
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+K][c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][c] = W444[K*4+I][c]
    
    offset2 = 49
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][offset2+c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+(K+4)][offset2+c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[(K+4)*4+I][offset2+c] = W444[K*4+I][c]
    
    offset3 = 98
    for I in range(4):
        for J in range(4):
            idx = I*4+J
            U[I*4+J][offset3+idx] = 1
            V[J*9+8][offset3+idx] = 1
            W[8*4+I][offset3+idx] = 1
    
    for I in range(4):
        for J in range(4):
            for K in range(9):
                s = sum(U[I*4+J][c] * V[J*9+K][c] * W[K*4+I][c] for c in range(total_rank))
                if s != 1:
                    return _schoolbook_fallback()
    
    return {"U": U, "V": V, "W": W}
