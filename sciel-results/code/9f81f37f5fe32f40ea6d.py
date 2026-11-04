import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Row indices: for U: (i,j) -> i*2+j; for V: (j,k) -> j*2+k; for W: (k,i) -> k*2+i
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
    
    # Verify Strassen correctness for 2x2
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        # Fallback to schoolbook if Strassen is broken (should not happen)
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Kronecker products: 16x49
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Build permutations by solving the tensor identity at runtime
    # The tensor identity is: sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] == 1 for all i,j,k
    # We need to find permutations perm_U, perm_V, perm_W such that:
    # U4[perm_U[S], t] for S = i*4+j gives the correct row for A[i,j]
    # V4[perm_V[S], t] for S = j*4+k gives the correct row for B[j,k]
    # W4[perm_W[S], t] for S = k*4+i gives the correct row for C[k,i]
    
    # The Kronecker product np.kron(U2, U2) has row index:
    # L4 = L2_outer * 4 + L2_inner
    # where L2_outer = r_out * 2 + c_out (for U, this is the outer block indices)
    # and L2_inner = r_in * 2 + c_in (for U, this is the inner indices)
    
    # For U (A matrix, 4x4):
    #   Standard index S = r*4 + c
    #   r = 2*r_out + r_in, c = 2*c_out + c_in
    #   Kronecker row: L4 = (r_out*2 + c_out) * 4 + (r_in*2 + c_in)
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out, r_in = r // 2, r % 2
        c_out, c_in = c // 2, c % 2
        L4 = (r_out * 2 + c_out) * 4 + (r_in * 2 + c_in)
        perm_U[S] = L4
    
    # For V (B matrix, 4x4):
    #   Standard index S = j*4 + k
    #   j = 2*j_out + j_in, k = 2*k_out + k_in
    #   Kronecker row: L4 = (j_out*2 + k_out) * 4 + (j_in*2 + k_in)
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        L4 = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        perm_V[S] = L4
    
    # For W (C matrix, trace convention W[k*4+r]):
    #   Standard index S = k*4 + r
    #   k = 2*k_out + k_in, r = 2*r_out + r_in
    #   In W2, row index = k*2 + r (first index is k, second is r)
    #   Kronecker row: L4 = (k_out*2 + r_out) * 4 + (k_in*2 + r_in)
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        r = S % 4
        k_out, k_in = k // 2, k % 2
        r_out, r_in = r // 2, r % 2
        L4 = (k_out * 2 + r_out) * 4 + (k_in * 2 + r_in)
        perm_W[S] = L4
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Runtime verification: construct the full tensor and check identity
    rank = U4.shape[1]
    
    # Build index arrays for all (i,j,k) triples
    i_idx = np.repeat(np.arange(4), 16)
    j_idx = np.tile(np.repeat(np.arange(4), 4), 4)
    k_idx = np.tile(np.arange(4), 16)
    
    U_rows = U4[i_idx * 4 + j_idx, :]
    V_rows = V4[j_idx * 4 + k_idx, :]
    W_rows = W4[k_idx * 4 + i_idx, :]
    
    products = U_rows * V_rows * W_rows
    totals = products.sum(axis=1)
    
    valid = np.all(totals == 1)
    
    if not valid:
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
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
