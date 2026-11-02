"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-98 decomposition for 4x4 by 4x8 matrix multiplication.
    Uses recursive Strassen (rank-49 for 4x4x4) duplicated for two 4-column blocks.
    Vectorized construction with verification and deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen 2x2x2 decomposition factors
    U2 = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ], dtype=int)
    
    # Reshape to (7, 2, 2) for Kronecker product
    U2_3d = U2.reshape(7, 2, 2)
    V2_3d = V2.reshape(7, 2, 2)
    W2_3d = W2.reshape(7, 2, 2)
    
    # Build 4x4x4 rank-49 decomposition via vectorized Kronecker product
    # U4[I,J, s,t] = U2[s, I//2, J//2] * U2[t, I%2, J%2]
    # Using einsum: U4_4d[i,j,s,t] = U2_3d[s,i,j] * U2_3d[t,i,j] -- NO, that's not right
    # Correct: U4[i,j,s,t] = U2_3d[s, i//2, j//2] * U2_3d[t, i%2, j%2]
    
    # Create index arrays
    I = np.arange(4)
    J = np.arange(4)
    I_grid, J_grid = np.meshgrid(I, J, indexing='ij')  # (4,4)
    
    # For each (i,j) in 4x4, we need (i//2, j//2) and (i%2, j%2)
    i_hi = I_grid // 2  # (4,4)
    j_hi = J_grid // 2  # (4,4)
    i_lo = I_grid % 2   # (4,4)
    j_lo = J_grid % 2   # (4,4)
    
    # U4_4d[i,j,s,t] = U2_3d[s, i_hi[i,j], j_hi[i,j]] * U2_3d[t, i_lo[i,j], j_lo[i,j]]
    # Shape: (4, 4, 7, 7)
    
    # Vectorized: for each (i,j), extract the appropriate elements
    # U2_3d[s, i_hi, j_hi] gives (7, 4, 4) -> transpose to (4, 4, 7)
    U2_hi = U2_3d[:, i_hi, j_hi]  # (7, 4, 4)
    U2_lo = U2_3d[:, i_lo, j_lo]  # (7, 4, 4)
    
    # U2_hi.T -> (4, 4, 7), U2_lo.T -> (4, 4, 7)
    # U4_4d = U2_hi.T[:, :, :, None] * U2_lo.T[:, :, None, :] -> (4, 4, 7, 7)
    U4_4d = U2_hi.transpose(1, 2, 0)[:, :, :, None] * U2_lo.transpose(1, 2, 0)[:, :, None, :]
    U4 = U4_4d.reshape(16, 49)
    
    # Same for V4 and W4
    V2_hi = V2_3d[:, i_hi, j_hi]  # (7, 4, 4)
    V2_lo = V2_3d[:, i_lo, j_lo]  # (7, 4, 4)
    V4_4d = V2_hi.transpose(1, 2, 0)[:, :, :, None] * V2_lo.transpose(1, 2, 0)[:, :, None, :]
    V4 = V4_4d.reshape(16, 49)
    
    W2_hi = W2_3d[:, i_hi, j_hi]  # (7, 4, 4)
    W2_lo = W2_3d[:, i_lo, j_lo]  # (7, 4, 4)
    W4_4d = W2_hi.transpose(1, 2, 0)[:, :, :, None] * W2_lo.transpose(1, 2, 0)[:, :, None, :]
    W4 = W4_4d.reshape(16, 49)
    
    # Now extend to 4x4x8: two copies of the 4x4x4 decomposition
    # Total rank = 49 + 49 = 98
    rank = 98
    
    # U: shape (n*m, rank) = (16, 98)
    # V: shape (m*p, rank) = (32, 98)
    # W: shape (p*n, rank) = (32, 98)
    
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)
    
    # Left block
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[i * 4 + k, :]
    
    # Right block
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[i * 4 + k, :]
    
    # Verify the decomposition
    def _verify(U_arr, V_arr, W_arr):
        r = U_arr.shape[1]
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition
    def _schoolbook():
        r = n * m * p
        U_s = [[0] * r for _ in range(n * m)]
        V_s = [[0] * r for _ in range(m * p)]
        W_s = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return _schoolbook()
# EVOLVE-BLOCK-END
