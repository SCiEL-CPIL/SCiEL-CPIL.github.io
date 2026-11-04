import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _make_block(n, m, p, i1, j1, k1, i2, j2, k2):
    """
    Create decomposition for a block A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
    i1, j1, k1 are start indices (inclusive)
    i2, j2, k2 are end indices (exclusive)
    """
    i_size = i2 - i1
    j_size = j2 - j1
    k_size = k2 - k1
    
    # Determine rank and type
    if i_size == 2 and j_size == 2 and k_size == 2:
        return _make_strassen_2x2_block(n, m, p, i1, i1+1, j1, j1+1, k1, k1+1)
    else:
        # Schoolbook for the rest
        rank = i_size * j_size * k_size
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

def _make_strassen_2x2_block(n, m, p, i1, i2, j1, j2, k1, k2):
    # Reuse existing implementation
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # M1
    U[i1*m + j1, 0] = 1.0
    U[i2*m + j2, 0] = 1.0
    V[j1*p + k1, 0] = 1.0
    V[j2*p + k2, 0] = 1.0
    W[k1*n + i1, 0] = 1.0
    W[k2*n + i2, 0] = 1.0
    
    # M2
    U[i2*m + j1, 1] = 1.0
    U[i2*m + j2, 1] = 1.0
    V[j1*p + k1, 1] = 1.0
    W[k2*n + i1, 1] = 1.0
    
    # M3
    U[i1*m + j1, 2] = 1.0
    V[j2*p + k1, 2] = 1.0
    V[j2*p + k2, 2] = -1.0
    W[k1*n + i2, 2] = 1.0
    
    # M4
    U[i2*m + j2, 3] = 1.0
    V[j1*p + k2, 3] = 1.0
    V[j1*p + k1, 3] = -1.0
    W[k1*n + i1, 3] = 1.0
    W[k1*n + i2, 3] = 1.0
    
    # M5
    U[i1*m + j1, 4] = 1.0
    U[i1*m + j2, 4] = 1.0
    V[j2*p + k2, 4] = 1.0
    W[k1*n + i1, 4] = -1.0
    W[k2*n + i1, 4] = 1.0
    
    # M6
    U[i2*m + j1, 5] = 1.0
    U[i1*m + j1, 5] = -1.0
    V[j1*p + k1, 5] = 1.0
    V[j1*p + k2, 5] = 1.0
    W[k1*n + i1, 5] = 1.0
    
    # M7
    U[i1*m + j2, 6] = 1.0
    U[i2*m + j2, 6] = -1.0
    V[j2*p + k1, 6] = 1.0
    V[j2*p + k2, 6] = 1.0
    W[k2*n + i2, 6] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    
    diag = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat, optimize=True)
    if not np.allclose(diag, 1.0, atol=tol):
        return False
    
    rng = np.random.default_rng(12345)
    num_samples = 100
    for _ in range(num_samples):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        ip = rng.integers(0, n)
        kp = rng.integers(0, p)
        if ip == i and kp == k:
            continue
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+ip, :])
        if abs(val) > tol:
            return False
    return True

def _get_blocks(n, m, p):
    # Partition into 2s and 1s
    def partition(size):
        blocks = []
        start = 0
        while start < size:
            if size - start >= 2:
                blocks.append((start, start + 2))
                start += 2
            else:
                blocks.append((start, start + 1))
                start += 1
        return blocks

    i_blocks = partition(n)
    j_blocks = partition(m)
    k_blocks = partition(p)
    
    all_blocks = []
    for i1, i2 in i_blocks:
        for j1, j2 in j_blocks:
            for k1, k2 in k_blocks:
                all_blocks.append((i1, j1, k1, i2, j2, k2))
    return all_blocks

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    blocks = _get_blocks(n, m, p)
    
    U_list = []
    V_list = []
    W_list = []
    
    for i1, j1, k1, i2, j2, k2 in blocks:
        U_b, V_b, W_b = _make_block(n, m, p, i1, j1, k1, i2, j2, k2)
        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)
        
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback
        U_f, V_f, W_f = _make_schoolbook(n, m, p)
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W
