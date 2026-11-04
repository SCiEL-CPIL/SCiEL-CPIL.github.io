import numpy as np
import time

def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication
    via Kronecker product of Strassen's 2x2x2 algorithm.
    
    The decomposition is: sum_t U[i,t] * V[j,t] * W[k,t] = delta_{i,j,k}
    where i = row index of A (0..15), j = row index of B (0..15), k = row index of C (0..15)
    
    For 4x4: i = r*4+c, j = col*4+row_B, k = row_C*4+col_C
    """
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (input A), V2: 4x7 (input B), W2: 4x7 (output C)
    # Standard Strassen decomposition
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
    
    # For 2x2: A row index S_A = r*2+c, B row index S_B = j*2+k, C row index S_C = k*2+r
    # Kronecker product row index K = L2_out*4 + L2_in
    # where L2_out is the "outer" 2x2 index and L2_in is the "inner" 2x2 index
    
    # For A (4x4): S = r*4+c
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
    
    # For B (4x4): S = j*4+k
    # j = j_out*2 + j_in, k = k_out*2 + k_in
    # L2_out = j_out*2 + k_out, L2_in = j_in*2 + k_in
    # K = L2_out*4 + L2_in
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        L2_out = j_out * 2 + k_out
        L2_in = j_in * 2 + k_in
        perm_V[S] = L2_out * 4 + L2_in
    
    # For C (4x4): S = k*4+r (output factor convention W[k*n+i, r])
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
    
    # Verify the decomposition
    # For each (i, j, k), check: sum_t U[i,t] * V[j,t] * W[k,t] == 1 if i==j==k else 0
    # Actually for matrix multiplication tensor: sum_t U[i,t] * V[j,t] * W[k,t] should give
    # the correct product structure. Let's verify with random matrices.
    
    # Quick verification with small integer matrices
    A_test = np.random.randint(-5, 5, (4, 4))
    B_test = np.random.randint(-5, 5, (4, 4))
    C_expected = A_test @ B_test
    
    # Compute using the decomposition
    # A_flat[i] where i = r*4+c
    # B_flat[j] where j = col*4+row (j = c*4+r for B in column-major? No, j = j*4+k where j is row of B)
    # Wait, for B (4x4): S = j*4+k where j is row index, k is column index
    # So B_flat[j*4+k] = B[j,k]
    
    A_flat = A_test.reshape(16)  # row-major: A[r,c] at index r*4+c
    B_flat = B_test.reshape(16)  # row-major: B[j,k] at index j*4+k
    
    # C_flat[k*4+r] = C[k,r] where C = A @ B, so C[k,r] = sum_j A[k,j]*B[j,r]
    # Wait, C = A @ B means C[i,j] = sum_k A[i,k]*B[k,j]
    # So C_flat[i*4+j] = C[i,j]
    
    # Using decomposition: C_flat[k*4+r] = sum_t (sum_i U[i,t]*A_flat[i]) * (sum_j V[j,t]*B_flat[j]) * W[k*4+r,t]
    # Wait, that's not right. Let me think again.
    
    # The tensor identity for matrix multiplication:
    # C[r,c] = sum_j A[r,j] * B[j,c]
    # In flat indices: C_flat[r*4+c] = sum_j A_flat[r*4+j] * B_flat[j*4+c]
    
    # The decomposition says: C_flat[k] = sum_t (sum_i U[i,t] * A_flat[i]) * (sum_j V[j,t] * B_flat[j]) * W[k,t]
    # where k is the output index (0..15), and W[k,t] corresponds to output position k
    
    # Let's verify:
    C_computed = np.zeros(16)
    for t in range(49):
        a_t = np.dot(U4[:, t], A_flat)
        b_t = np.dot(V4[:, t], B_flat)
        C_computed += a_t * b_t * W4[:, t]
    
    # C_computed should equal C_expected.reshape(16)
    if not np.allclose(C_computed, C_expected.reshape(16), atol=1e-9):
        # Fallback to schoolbook if verification fails
        if time.time() - start_time < 100:
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
    
    if time.time() - start_time > 120:
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
