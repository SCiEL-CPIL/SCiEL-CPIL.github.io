import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    The key insight: we can use Kronecker products of factor matrices, but we need
    to apply the appropriate permutations to each factor to account for the different
    indexing conventions of the input, intermediate, and output matrices.
    
    For A (input, 4x4):  row index S = r*4 + c, where r = r_out*2 + r_in, c = c_out*2 + c_in
    For B (input, 4x4):  row index S = j*4 + k, where j = j_out*2 + j_in, k = k_out*2 + k_in
    For C (output, 4x4): row index S = k*4 + r, where k = k_out*2 + k_in, r = r_out*2 + r_in
    
    The Kronecker product U2 ⊗ U2 has rows indexed by (L2_out, L2_in) where:
    - L2_out = r_out*2 + c_out (for A) or j_out*2 + k_out (for B)
    - L2_in = r_in*2 + c_in (for A) or j_in*2 + k_in (for B)
    
    So the Kronecker row index is: K = L2_out*4 + L2_in
    """
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (input A), V2: 4x7 (input B), W2: 4x7 (output C)
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
    
    # Kronecker products
    U4_raw = np.kron(U2, U2)  # 16x49
    V4_raw = np.kron(V2, V2)  # 16x49
    W4_raw = np.kron(W2, W2)  # 16x49
    
    # Permutation for A (input): S = r*4 + c
    # r = r_out*2 + r_in, c = c_out*2 + c_in
    # L2_out = r_out*2 + c_out, L2_in = r_in*2 + c_in
    # K = L2_out*4 + L2_in
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out, r_in = r // 2, r % 2
        c_out, c_in = c // 2, c % 2
        L2_out = r_out * 2 + c_out
        L2_in = r_in * 2 + c_in
        perm_U[S] = L2_out * 4 + L2_in
    
    # Permutation for B (input): S = j*4 + k
    # j = j_out*2 + j_in, k = k_out*2 + k_in
    # L2_out = j_out*2 + k_out, L2_in = j_in*2 + k_in
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        L2_out = j_out * 2 + k_out
        L2_in = j_in * 2 + k_in
        perm_V[S] = L2_out * 4 + L2_in
    
    # Permutation for C (output): S = k*4 + r
    # k = k_out*2 + k_in, r = r_out*2 + r_in
    # In W2, the index is (k, r), so L2_out = k_out*2 + r_out, L2_in = k_in*2 + r_in
    # K = L2_out*4 + L2_in
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        r = S % 4
        k_out, k_in = k // 2, k % 2
        r_out, r_in = r // 2, r % 2
        L2_out = k_out * 2 + r_out
        L2_in = k_in * 2 + r_in
        perm_W[S] = L2_out * 4 + L2_in
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify the decomposition (optional, but good for debugging)
    # The tensor identity: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta
    # For each (i,j,k), we should get 1 if this is a valid product entry
    
    # Fallback to schoolbook if timeout
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
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
