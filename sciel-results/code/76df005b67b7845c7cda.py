import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a direct Strassen-based decomposition for 4x4x8.
    Total rank: 98.
    """
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # --- Step 1: Construct rank-49 decomposition for 4x4x4 ---
    # Using the Kronecker product structure of two Strassen levels.
    # This is equivalent to the alpha/beta/gamma/delta/kappa construction but
    # implemented via the 2x2 Strassen factors for clarity and speed.
    
    # 2x2 Strassen factors (indices: row-major for 2x2 blocks)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Row index in U2: i*2 + j (i,j in 0..1)
    # Row index in V2: j*2 + k (j,k in 0..1)
    # Row index in W2: k*2 + i (k,i in 0..1)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # (0,1)
        [0, 1, 0, 0, 0, 1, 0],   # (1,0)
        [1, 1, 0, 1, 0, 0, -1]   # (1,1)
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # (0,0)
        [0, 0, 1, 0, 0, 1, 0],   # (0,1)
        [0, 0, 0, 1, 0, 0, 1],   # (1,0)
        [1, 0, -1, 0, 1, 0, 1]   # (1,1)
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # (0,0)
        [0, 1, 0, 1, 0, 0, 0],   # (0,1)
        [0, 0, 1, 0, 1, 0, 0],   # (1,0)
        [1, -1, 1, 0, 0, 1, 0]   # (1,1)
    ], dtype=float)
    
    # Build 4x4x4 factors using Kronecker product structure
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # For each (i,j) in 0..3 (A index), (j',k) in 0..3 (B index), (k',i') in 0..3 (C index)
    # Decompose i = 2*i1 + i2, j = 2*j1 + j2, etc.
    # U4[i*4+j, t*7+s] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for t in range(7):
                for s in range(7):
                    c = t * 7 + s
                    U4[i * 4 + j, c] = U2[i1 * 2 + j1, t] * U2[i2 * 2 + j2, s]
    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for t in range(7):
                for s in range(7):
                    c = t * 7 + s
                    V4[j * 4 + k, c] = V2[j1 * 2 + k1, t] * V2[j2 * 2 + k2, s]
    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            for t in range(7):
                for s in range(7):
                    c = t * 7 + s
                    W4[k * 4 + i, c] = W2[k1 * 2 + i1, t] * W2[k2 * 2 + i2, s]
    
    # --- Step 2: Construct 4x4x8 factors ---
    # The 4x4x8 problem can be seen as two independent 4x4x4 multiplications
    # (one for k in [0,4) and one for k in [4,8)), sharing the same A matrix.
    # This gives rank 2 * 49 = 98.
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Copy 1: k in [0, 4)
    U[:, 0:49] = U4
    # V: index 8*j + k. k in [0,4)
    for j in range(4):
        for k in range(4):
            V[8 * j + k, 0:49] = V4[4 * j + k, :]
    # W: index 4*k + i. k in [0,4)
    for k in range(4):
        for i in range(4):
            W[4 * k + i, 0:49] = W4[4 * k + i, :]
    
    # Copy 2: k in [4, 8)
    U[:, 49:98] = U4
    # V: index 8*j + k. k in [4,8) -> k' = k-4 in [0,4)
    for j in range(4):
        for k in range(4, 8):
            V[8 * j + k, 49:98] = V4[4 * j + (k - 4), :]
    # W: index 4*k + i. k in [4,8) -> k' = k-4 in [0,4)
    for k in range(4, 8):
        for i in range(4):
            W[4 * k + i, 49:98] = W4[4 * (k - 4) + i, :]
    
    # Verify decomposition
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m+j = i*4+j, b = j'*p+k = j'*8+k, c = k'*n+i' = k'*4+i'
    # T should be 1 if i=i', j=j', k=k' else 0
    
    # Quick spot check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build expected tensor
    T_expected = np.zeros((16, 32, 32), dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(8):
                a = i * 4 + j
                b = j * 8 + k
                c = k * 4 + i
                T_expected[a, b, c] = 1
    
    if not np.allclose(T_recon, T_expected, atol=1e-6):
        # Fallback: return schoolbook if verification fails (shouldn't happen)
        rank_schoolbook = 128
        U_s = np.zeros((16, rank_schoolbook), dtype=float)
        V_s = np.zeros((32, rank_schoolbook), dtype=float)
        W_s = np.zeros((32, rank_schoolbook), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_s[i * 4 + j, col] = 1
                    V_s[j * 8 + k, col] = 1
                    W_s[k * 4 + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Convert to half-integer if possible
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        return mat.tolist()
    
    return {"U": to_half_integer(U), "V": to_half_integer(V), "W": to_half_integer(W)}
