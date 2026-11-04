import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0

def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition factors
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
    
    # Kronecker product for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation to map Kronecker row order to standard row-major order
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        R = S // 4
        C = S % 4
        r_in = R % 2
        r_out = R // 2
        c_in = C % 2
        c_out = C // 2
        K_in = r_in * 2 + c_in
        K_out = r_out * 2 + c_out
        K = K_out * 4 + K_in
        perm[S] = K
    
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Helper to verify and return
    def verify_and_return(U_arr, V_arr, W_arr):
        U_float = np.array(U_arr, dtype=float)
        V_float = np.array(V_arr, dtype=float)
        W_float = np.array(W_arr, dtype=float)
        
        T = np.zeros((16, 16, 16), dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    T[i*4+j, j*4+k, k*4+i] = 1.0
        
        T_decomp = np.einsum('ar,br,cr->abc', U_float, V_float, W_float)
        max_diff = np.max(np.abs(T_decomp - T))
        
        if max_diff < 1e-9:
            return {
                "U": U_arr.tolist(),
                "V": V_arr.tolist(),
                "W": W_arr.tolist()
            }
        return None

    # 1. Attempt Rank 48 Decomposition
    # Strategy: Try to reduce rank by checking if any column in the Rank 49 decomposition
    # is linearly dependent or can be removed (it won't be, but this is the structural attempt).
    # We can also try to find a rank 48 decomposition by solving for integer coefficients
    # if we had a template, but we don't. So we just check if Rank 49 is minimal.
    
    # Check if Rank 49 factors are full rank in a way that suggests no reduction is possible
    # by trying to drop one column and verifying.
    for drop_idx in range(49):
        if time.time() > _DEADLINE:
            break
        U_try = np.delete(U4_fixed, drop_idx, axis=1)
        V_try = np.delete(V4_fixed, drop_idx, axis=1)
        W_try = np.delete(W4_fixed, drop_idx, axis=1)
        
        result = verify_and_return(U_try, V_try, W_try)
        if result:
            return result

    # 2. Fallback to Rank 49
    result = verify_and_return(U4_fixed, V4_fixed, W4_fixed)
    if result:
        return result

    # 3. Fallback to Schoolbook Rank 64
    rank = 64
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][column] = 1
                V[j * 4 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
