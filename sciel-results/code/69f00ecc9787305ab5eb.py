import numpy as np

def construct():
    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build 4x4 factors from 2x2 Strassen factors using Kronecker product logic
    # Rank is 7 * 7 = 49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    # Vectorized construction for speed
    # Indices for U4: (i, j) -> i*4+j. c1, c2 -> c1*7+c2
    # U4[i*4+j, c1*7+c2] = U2[i_hi*2+j_hi, c1] * U2[i_lo*2+j_lo, c2]
    
    # Create index grids
    i_range = np.arange(4)
    j_range = np.arange(4)
    k_range = np.arange(4)
    c1_range = np.arange(7)
    c2_range = np.arange(7)
    
    # U4
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            u_hi = U2[i_hi*2+j_hi, :]
            u_lo = U2[i_lo*2+j_lo, :]
            # Outer product of u_hi and u_lo, reshaped to (7, 7) then flattened
            U4[i*4+j, :] = np.outer(u_hi, u_lo).flatten()
            
    # V4
    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            v_hi = V2[j_hi*2+k_hi, :]
            v_lo = V2[j_lo*2+k_lo, :]
            V4[j*4+k, :] = np.outer(v_hi, v_lo).flatten()
            
    # W4
    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            w_hi = W2[k_hi*2+i_hi, :]
            w_lo = W2[k_lo*2+i_lo, :]
            W4[k*4+i, :] = np.outer(w_hi, w_lo).flatten()

    # Verify 4x4 correctness
    # Check sum(U4[i,j] * V4[j,k] * W4[k,i]) == 1 for all i,j,k
    ok4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :]))
                if s != 1:
                    ok4 = False
                    break
            if not ok4:
                break
        if not ok4:
            break

    if not ok4:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
