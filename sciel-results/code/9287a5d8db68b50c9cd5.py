import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U, V, W are 4x7 matrices
    # Indices: rows 0..3 correspond to (i,j) in 0..1,0..1 order
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
    # We need to be careful with the row permutations to match the standard
    # row-major indexing for the 4x4 matrices.
    
    # The Kronecker product U2 ⊗ U2 has rows indexed by (i2_outer, i2_inner)
    # where i2_outer is the row index in the first U2 and i2_inner in the second.
    # Let i2_outer = r_out * 2 + c_out and i2_inner = r_in * 2 + c_in.
    # The linear index in the 4x4 matrix is S = (r_out*2 + r_in)*4 + (c_out*2 + c_in).
    # The Kronecker index is K = i2_outer * 4 + i2_inner = (r_out*2 + c_out)*4 + (r_in*2 + c_in).
    
    # We need a permutation P such that U4_fixed[S, :] = U4_kron[P[S], :]
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out = r // 2
        r_in = r % 2
        c_out = c // 2
        c_in = c % 2
        
        i2_out = r_out * 2 + c_out
        i2_in = r_in * 2 + c_in
        
        K = i2_out * 4 + i2_in
        perm[S] = K

    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]

    # Verify correctness
    # Check sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] == 1
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

    # Return the rank-49 solution
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
