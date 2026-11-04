import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U2, V2, W2 are 4x7 matrices
    # Row indices 0..3 correspond to (i,j) in 0..1,0..1 order
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
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

    # Construct rank-49 decomposition for 4x4 via Kronecker product
    # U4_kron = np.kron(U2, U2) has shape (16, 49)
    # Rows are indexed by K = L_out * 4 + L_in
    # Where L_out is the row index of the outer 2x2 block structure
    # and L_in is the row index of the inner 2x2 entry structure.
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)

    # We need to permute the rows of these matrices to match the standard
    # row-major indexing required by the evaluator.
    
    # For U (Input A):
    # Standard index S_A = i * 4 + j
    # i = 2 * i_out + i_in
    # j = 2 * j_out + j_in
    # S_A = (2*i_out + i_in)*4 + (2*j_out + j_in)
    #     = 8*i_out + 4*i_in + 2*j_out + j_in
    
    # Kronecker index K_U = L_out * 4 + L_in
    # L_out = i_out * 2 + j_out  (row index of outer 2x2)
    # L_in  = i_in * 2 + j_in   (row index of inner 2x2)
    # K_U = (i_out*2 + j_out)*4 + (i_in*2 + j_in)
    #     = 8*i_out + 4*j_out + 2*i_in + j_in
    
    # We need perm_U such that U_fixed[S_A] = U_kron[K_U]
    perm_U = np.zeros(16, dtype=int)
    for S_A in range(16):
        i = S_A // 4
        j = S_A % 4
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        
        L_out = i_out * 2 + j_out
        L_in = i_in * 2 + j_in
        K_U = L_out * 4 + L_in
        perm_U[S_A] = K_U

    # For V (Input B):
    # Standard index S_B = j * 4 + k
    # j = 2 * j_out + j_in
    # k = 2 * k_out + k_in
    # S_B = 8*j_out + 4*j_in + 2*k_out + k_in
    
    # Kronecker index K_V = L_out * 4 + L_in
    # L_out = j_out * 2 + k_out
    # L_in  = j_in * 2 + k_in
    # K_V = 8*j_out + 4*k_out + 2*j_in + k_in
    
    perm_V = np.zeros(16, dtype=int)
    for S_B in range(16):
        j = S_B // 4
        k = S_B % 4
        j_out = j // 2
        j_in = j % 2
        k_out = k // 2
        k_in = k % 2
        
        L_out = j_out * 2 + k_out
        L_in = j_in * 2 + k_in
        K_V = L_out * 4 + L_in
        perm_V[S_B] = K_V

    # For W (Output C):
    # Standard index S_C = k * 4 + i (Trace convention W[k*n+i, r])
    # k = 2 * k_out + k_in
    # i = 2 * i_out + i_in
    # S_C = 8*k_out + 4*k_in + 2*i_out + i_in
    
    # Kronecker index K_W = L_out * 4 + L_in
    # L_out = k_out * 2 + i_out
    # L_in  = k_in * 2 + i_in
    # K_W = 8*k_out + 4*i_out + 2*k_in + i_in
    
    perm_W = np.zeros(16, dtype=int)
    for S_C in range(16):
        k = S_C // 4
        i = S_C % 4
        k_out = k // 2
        k_in = k % 2
        i_out = i // 2
        i_in = i % 2
        
        L_out = k_out * 2 + i_out
        L_in = k_in * 2 + i_in
        K_W = L_out * 4 + L_in
        perm_W[S_C] = K_W

    # Apply permutations
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # Quick verification to ensure correctness before returning
    # Check sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] == 1
    # This is a 4x4x4 sum, very fast.
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(U4[i * 4 + j] * V4[j * 4 + k] * W4[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank 64 if construction fails (should not happen)
        rank = 64
        U_f = np.zeros((16, 64), dtype=int)
        V_f = np.zeros((16, 64), dtype=int)
        W_f = np.zeros((16, 64), dtype=int)
        col = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U_f[ii * 4 + jj, col] = 1
                    V_f[jj * 4 + kk, col] = 1
                    W_f[kk * 4 + ii, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
