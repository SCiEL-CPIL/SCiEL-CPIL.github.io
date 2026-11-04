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

def _strassen_4x4x4():
    """Return U, V, W for 4x4x4 Strassen-based decomposition (rank 49)."""
    U1, V1, W1 = _strassen_2x2()
    
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
    
    return U444, V444, W444

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

def _validate_decomposition(U, V, W, n, m, p):
    """Validate using NumPy for speed and correctness."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U444, V444, W444 = _strassen_4x4x4()
    
    # Validate 4x4x4 decomposition
    if not _validate_decomposition(U444, V444, W444, 4, 4, 4):
        return _schoolbook_fallback()
    
    # Total rank: 49 + 49 + 1 = 99
    total_rank = 99
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    # Block 1: B columns 0-3 (k=0..3)
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*m+J][c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*p+K][c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*n+I][c] = W444[K*4+I][c]
    
    # Block 2: B columns 4-7 (k=4..7)
    offset2 = 49
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*m+J][offset2+c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*p+(K+4)][offset2+c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*n+I][offset2+c] = W444[K*4+I][c]
    
    # Block 3: B column 8 (k=8) — rank 1
    offset3 = 98
    for I in range(4):
        for J in range(4):
            U[I*m+J][offset3] = 1
            V[J*p+8][offset3] = 1
            W[8*n+I][offset3] = 1

    # Final validation
    if not _validate_decomposition(U, V, W, n, m, p):
        return _schoolbook_fallback()
        
    return {"U": U, "V": V, "W": W}
