import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (rows indexed by (i,j) of A, i.e., row*2+col)
    # V2: 4x7 (rows indexed by (j,k) of B, i.e., row*2+col)
    # W2: 4x7 (rows indexed by (k,i) of C, i.e., row*2+col in trace convention)
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
    
    # Kronecker products: 16x49
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Build permutations: map standard row-major index S to Kronecker index K
    # For U (A matrix, 4x4):
    #   S = r*4 + c
    #   r = 2*r_out + r_in, c = 2*c_out + c_in
    #   Kronecker: outer = r_out*2 + c_out, inner = r_in*2 + c_in
    #   K = outer*4 + inner
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out, r_in = r // 2, r % 2
        c_out, c_in = c // 2, c % 2
        K = (r_out * 2 + c_out) * 4 + (r_in * 2 + c_in)
        perm_U[S] = K
    
    # For V (B matrix, 4x4):
    #   S = j*4 + k
    #   j = 2*j_out + j_in, k = 2*k_out + k_in
    #   Kronecker: outer = j_out*2 + k_out, inner = j_in*2 + k_in
    #   K = outer*4 + inner
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        K = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        perm_V[S] = K
    
    # For W (C matrix, output, trace convention W[k*4+r]):
    #   S = k*4 + r  (k is col of B / row of C in trace, r is row of A / col of C in trace)
    #   k = 2*k_out + k_in, r = 2*r_out + r_in
    #   In W2, row index = k*2 + r (first index is k from B, second is r from A)
    #   Kronecker: outer = k_out*2 + r_out, inner = k_in*2 + r_in
    #   K = outer*4 + inner
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        r = S % 4
        k_out, k_in = k // 2, k % 2
        r_out, r_in = r // 2, r % 2
        K = (k_out * 2 + r_out) * 4 + (k_in * 2 + r_in)
        perm_W[S] = K
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify tensor identity: sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] == 1
    # for all (i, j, k) in 0..3
    def verify(U, V, W, n=4, m=4, p=4):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(rank):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, info = verify(U4, V4, W4)
    
    if not valid:
        # Fallback to schoolbook if Kronecker fails verification
        if time.time() - start_time > 100:
            n, m, p = 4, 4, 4
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
        
        # If not yet at timeout, still fall back to schoolbook (rank 64)
        # since the Kronecker decomposition is invalid
        n, m, p = 4, 4, 4
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
    
    # Safe deadline check
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
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
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
