import numpy as np
import time

def construct():
    """
    Returns a rank-35 tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses Strassen's algorithm for each of the 5 independent 2x2 column blocks of B.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 2, 10
    rank = 35

    # Strassen's 7 terms for 2x2 matrix multiplication
    # u_vecs: factors for A (2x2), indexed [A00, A01, A10, A11]
    # v_vecs: factors for B (2x2), indexed [B00, B01, B10, B11]
    # w_vecs: factors for C (2x2), indexed [C00, C10, C01, C11]
    # (C convention: C[i,j] stored as C row i, col j; W index = col*n + row)
    
    u_vecs = [
        [1, 0, 0, 1],    # M1: (a+d)
        [0, 0, 1, 1],    # M2: (c+d)
        [1, 0, 0, 0],    # M3: a
        [0, 0, 0, 1],    # M4: d
        [1, 1, 0, 0],    # M5: (a+b)
        [-1, 0, 1, 0],   # M6: (c-a)
        [0, 1, 0, -1]    # M7: (d-b)
    ]
    
    v_vecs = [
        [1, 0, 0, 1],    # M1: (e+h)
        [1, 0, 0, 0],    # M2: e
        [0, 1, 0, -1],   # M3: (f-h)
        [-1, 0, 1, 0],   # M4: (g-e)
        [0, 0, 0, 1],    # M5: h
        [1, 1, 0, 0],    # M6: (e+f)
        [0, 0, 1, 1]     # M7: (g+h)
    ]
    
    w_vecs = [
        [1, 0, 0, 1],    # M1: contributes to C00, C11
        [0, 1, 0, -1],   # M2: contributes to C10, C11
        [0, 0, 1, 1],    # M3: contributes to C01, C11
        [1, 1, 0, 0],    # M4: contributes to C00, C10
        [-1, 0, 1, 0],   # M5: contributes to C00, C01
        [0, 0, 0, 1],    # M6: contributes to C11
        [1, 0, 0, 0]     # M7: contributes to C00
    ]
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    num_blocks = p // 2  # 5 blocks of 2x2
    
    for r in range(num_blocks):
        for k in range(7):
            if time.time() > deadline:
                # Return partial result if time runs out, though this shouldn't happen
                return {"U": U, "V": V, "W": W}
            
            # U: same for all blocks (A is 2x2)
            for i in range(n * m):
                U[i][col] = u_vecs[k][i]
            
            # V: B is 2x10, block r covers columns 2r, 2r+1
            # B[0, 2r]   -> V index 0*p + 2r = 2r
            # B[0, 2r+1] -> V index 0*p + 2r+1 = 2r+1
            # B[1, 2r]   -> V index 1*p + 2r = 10+2r
            # B[1, 2r+1] -> V index 1*p + 2r+1 = 10+2r+1
            V[2 * r][col] = v_vecs[k][0]
            V[2 * r + 1][col] = v_vecs[k][1]
            V[p + 2 * r][col] = v_vecs[k][2]
            V[p + 2 * r + 1][col] = v_vecs[k][3]
            
            # W: C is 2x10, convention W[k*n+i] where k=col, i=row
            # C[0, 2r]   -> W[(2r)*2 + 0] = W[4r]
            # C[1, 2r]   -> W[(2r)*2 + 1] = W[4r+1]
            # C[0, 2r+1] -> W[(2r+1)*2 + 0] = W[4r+2]
            # C[1, 2r+1] -> W[(2r+1)*2 + 1] = W[4r+3]
            W[4 * r][col] = w_vecs[k][0]
            W[4 * r + 1][col] = w_vecs[k][1]
            W[4 * r + 2][col] = w_vecs[k][2]
            W[4 * r + 3][col] = w_vecs[k][3]
            
            col += 1
        
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
