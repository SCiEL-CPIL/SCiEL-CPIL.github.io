import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 6, 6, 6
    dim = n  # 6

    # Block structure: 6x6 = 2x2 blocks of 3x3
    # Outer: 2x2 matrix multiplication (Strassen rank 7)
    # Inner: 3x3 matrix multiplication (Naive rank 27)
    # Total Rank: 7 * 27 = 189
    
    block_size = 3
    total_rank = 7 * 27  # 189
    
    # Initialize matrices
    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2 multiplication
    # Each step is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # Coeffs are lists of (row, col, coeff)
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    bs = block_size
    
    # Iterate over Strassen steps
    for x_list, y_list, c_list in strassen_2x2:
        # Iterate over inner 3x3 block indices
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    # Update U
                    for (ib, jb, x_coeff) in x_list:
                        u_idx = (ib * bs + r) * dim + (jb * bs + s)
                        U[u_idx, col] += x_coeff
                    
                    # Update V
                    for (jb, kb, y_coeff) in y_list:
                        v_idx = (jb * bs + s) * dim + (kb * bs + c)
                        V[v_idx, col] += y_coeff
                    
                    # Update W
                    for (ib, kb, c_coeff) in c_list:
                        # Trace convention: W[k*n+i, r] where k=col_C, i=row_C
                        w_idx = (kb * bs + c) * dim + (ib * bs + r)
                        W[w_idx, col] += c_coeff
                    
                    col += 1
                    
                    if time.time() > deadline:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Quick verification
    U_arr = U
    V_arr = V
    W_arr = W
    
    import random
    random.seed(42)
    valid = True
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
        if abs(val - 1.0) > 1e-10:
            valid = False
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback))
        V_f = np.zeros((m * p, rank_fallback))
        W_f = np.zeros((p * n, rank_fallback))
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col_f] = 1
                    V_f[j * p + k, col_f] = 1
                    W_f[k * n + i, col_f] = 1
                    col_f += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
