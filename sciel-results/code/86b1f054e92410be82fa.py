"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _build_strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with Strassen's rank-7 decomposition.
    U: shape (4, 7), U[i*2+j, r]
    V: shape (4, 7), V[j*2+k, r]
    W: shape (4, 7), W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # M1: A[0,0] * (B[0,1] - B[1,1])
    U[0*2+0, 0] = 1
    V[0*2+1, 0] = 1
    V[1*2+1, 0] = -1
    W[0*2+0, 0] = 1   # C[0,0]
    W[1*2+0, 0] = -1  # C[0,1]
    
    # M2: (A[0,0] + A[1,1]) * B[1,1]
    U[0*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[1*2+1, 1] = 1
    W[1*2+1, 1] = 1   # C[1,1]
    
    # M3: (A[1,1] - A[0,0]) * B[0,0]
    U[1*2+1, 2] = 1
    U[0*2+0, 2] = -1
    V[0*2+0, 2] = 1
    W[0*2+0, 2] = -1  # C[0,0]
    W[1*2+1, 2] = 1   # C[1,1]
    
    # M4: (A[1,0] + A[1,1]) * (B[1,1] - B[0,1])
    U[1*2+0, 3] = 1
    U[1*2+1, 3] = 1
    V[1*2+1, 3] = 1
    V[0*2+1, 3] = -1
    W[0*2+1, 3] = 1   # C[1,0]
    W[1*2+1, 3] = 1   # C[1,1]
    
    # M5: (A[0,0] + A[0,1]) * B[0,0]
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[0*2+0, 4] = 1
    W[1*2+0, 4] = 1   # C[0,1]
    W[0*2+0, 4] = 1   # C[0,0]
    
    # M6: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[0*2+1, 5] = 1
    U[1*2+1, 5] = -1
    V[1*2+0, 5] = 1
    V[1*2+1, 5] = 1
    W[0*2+1, 5] = 1   # C[1,0]
    
    # M7: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[1*2+0, 6] = 1
    U[0*2+0, 6] = -1
    V[0*2+0, 6] = 1
    V[0*2+1, 6] = 1
    W[0*2+0, 6] = -1  # C[0,0]
    W[1*2+0, 6] = 1   # C[0,1]
    
    return U, V, W

def _kron_2x2x2_to_4x4x4(U2, V2, W2):
    """
    Tensor product of 2x2x2 with itself to get 4x4x4.
    U2: (4, 7), V2: (4, 7), W2: (4, 7)
    Result: U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n1, m1, p1, r1 = 2, 2, 2, 7
    n2, m2, p2, r2 = 2, 2, 2, 7
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            
            # U: index (i1*m2*n2 + i2*m2 + j1*n2 + j2, r) 
            # A is 4x4, A[i,j] where i = i1*2+i2, j = j1*2+j2
            # U[i*4+j, r] = U2[i1*2+j1, r1_idx] * U2[i2*2+j2, r2_idx]
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U2[i1*m1 + j1, r1_idx]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2*m2 + j2, r2_idx]
                            if u2 == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, r] += u1 * u2
            
            # V: B is 4x4, B[j,k] where j = j1*2+j2, k = k1*2+k2
            # V[j*4+k, r] = V2[j1*2+k1, r1_idx] * V2[j2*2+k2, r2_idx]
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V2[j1*p1 + k1, r1_idx]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2*p2 + k2, r2_idx]
                            if v2 == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, r] += v1 * v2
            
            # W: C is 4x4, C[i,k] where i = i1*2+i2, k = k1*2+k2
            # W[k*4+i, r] = W2[k1*2+i1, r1_idx] * W2[k2*2+i2, r2_idx]
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W2[k1*n1 + i1, r1_idx]
                    if w1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2 = W2[k2*n2 + i2, r2_idx]
                            if w2 == 0:
                                continue
                            i = i1 * n2 + i2
                            k = k1 * p2 + k2
                            W[k * n + i, r] += w1 * w2
    
    return U, V, W

def _build_4x4x8_decomposition():
    """
    Build a rank-98 decomposition for 4x4x8 by concatenating two 4x4x4 Strassen decompositions.
    U: (16, 98), V: (32, 98), W: (32, 98)
    """
    n, m, p = 4, 4, 8
    rank = 98
    
    base_U, base_V, base_W = _build_strassen_2x2x2()
    U4, V4, W4 = _kron_2x2x2_to_4x4x4(base_U, base_V, base_W)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First half: B columns 0-3
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half: B columns 4-7
    U[:, 49:98] = U4
    # V4[j*4+k, r] maps to V[j*8+(k+4), r]
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V4[j*4 + k, :]
    
    # W for second half: k in 4..7, so W[(k)*4+i, r] for k in 4..7
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, :]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    try:
        U, V, W = _build_4x4x8_decomposition()
        
        # Verify the decomposition
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
        
    except Exception:
        # Fall back to schoolbook
        n, m, p = DIMENSIONS
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
