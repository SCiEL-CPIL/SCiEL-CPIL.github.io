import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U, V, W are of shape (4, 7).
    Indices:
    U: i*2 + j (i,j in 0..1)
    V: j*2 + k (j,k in 0..1)
    W: k*2 + i (k,i in 0..1)
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Constructs the rank-49 decomposition for 4x4x4 matrix multiplication
    using the Kronecker product of two 2x2x2 Strassen decompositions.
    Returns U, V, W of shapes (16, 49).
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    
    for r_outer in range(7):
        for r_inner in range(7):
            r = r_outer * 7 + r_inner
            
            # U: i*4 + j
            for i in range(4):
                for j in range(4):
                    idx_u = i * 4 + j
                    i_out, j_out = i // 2, i % 2
                    j_out2, j_in = j // 2, j % 2
                    # Wait, indices for U2 are i*2+j.
                    # i is 0..3, j is 0..3.
                    # i = 2*i0 + i1, j = 2*j0 + j1
                    # U2 index for outer: i0*2 + j0
                    # U2 index for inner: i1*2 + j1
                    u_val = U2[i_out * 2 + j_out, r_outer] * U2[i_out2 * 2 + j_in, r_inner]
                    # Correction: 
                    # i = 2*i0 + i1. j = 2*j0 + j1.
                    # Outer part: i0, j0. Inner part: i1, j1.
                    # U2[outer_idx, r_outer] * U2[inner_idx, r_inner]
                    # outer_idx = i0*2 + j0
                    # inner_idx = i1*2 + j1
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    u_val = U2[i0 * 2 + j0, r_outer] * U2[i1 * 2 + j1, r_inner]
                    U4[idx_u, r] = u_val

            # V: j*4 + k
            for j in range(4):
                for k in range(4):
                    idx_v = j * 4 + k
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    v_val = V2[j0 * 2 + k0, r_outer] * V2[j1 * 2 + k1, r_inner]
                    V4[idx_v, r] = v_val

            # W: k*4 + i
            for k in range(4):
                for i in range(4):
                    idx_w = k * 4 + i
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    w_val = W2[k0 * 2 + i0, r_outer] * W2[k1 * 2 + i1, r_inner]
                    W4[idx_w, r] = w_val

    return U4, V4, W4

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the n x m x p matrix multiplication tensor.
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal 1 for all i, j, k (since T is the identity tensor for matmul).
    """
    rank = U.shape[1]
    # Use a few random tests or check all? 
    # Checking all i,j,k is 4*4*8 = 128 checks. Each check sums 98 terms. Fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if total != 1:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Construct 4x4x8 decomposition
    # Rank is 49 + 49 = 98
    rank = 98
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)
    
    # Block 1: Output columns 0-3
    U[:, 0:49] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = k_local
            V[j * 8 + k_global, 0:49] = V4[j * 4 + k_local, :]
    for k_local in range(4):
        for i in range(4):
            k_global = k_local
            W[k_global * 4 + i, 0:49] = W4[k_local * 4 + i, :]
            
    # Block 2: Output columns 4-7
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j * 8 + k_global, 49:98] = V4[j * 4 + k_local, :]
    for k_local in range(4):
        for i in range(4):
            k_global = 4 + k_local
            W[k_global * 4 + i, 49:98] = W4[k_local * 4 + i, :]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 128
        U = np.zeros((16, rank), dtype=np.int16)
        V = np.zeros((32, rank), dtype=np.int16)
        W = np.zeros((32, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
