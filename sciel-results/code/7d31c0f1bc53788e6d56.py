import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {'U': U, 'V': V, 'W': W}

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-8:
                    return False
    # Check all zeros
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                if abs(T[a, b, c] - expected) > 1e-8:
                    return False
    return True

def _als_decompose(n, m, p, target_rank, max_iters=200):
    """Alternating least squares to find a rank-target_rank decomposition."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    np.random.seed(42)
    U = np.zeros((dim_a, target_rank))
    V = np.zeros((dim_b, target_rank))
    W = np.zeros((dim_c, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < target_rank:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    if col < target_rank:
        U[:, col:] = np.random.randn(dim_a, target_rank-col) * 0.1
        V[:, col:] = np.random.randn(dim_b, target_rank-col) * 0.1
        W[:, col:] = np.random.randn(dim_c, target_rank-col) * 0.1
    
    for it in range(max_iters):
        if _remaining() < 5:
            return None
        
        # Update U
        for a in range(dim_a):
            M = (V[:, :, None] * W[None, :, :]).reshape(dim_b * dim_c, target_rank)
            rhs = T[a].flatten()
            U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        
        # Update V
        for b in range(dim_b):
            M = (U[:, :, None] * W[None, :, :]).reshape(dim_a * dim_c, target_rank)
            rhs = T[:, b, :].reshape(dim_a * dim_c)
            V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        
        # Update W
        for c in range(dim_c):
            M = (U[:, :, None] * V[None, :, :]).reshape(dim_a * dim_b, target_rank)
            rhs = T[:, :, c].reshape(dim_a * dim_b)
            W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        
        # Check
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) < 1e-10:
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def construct():
    n, m, p = 2, 5, 5
    
    for r in range(49, 24, -1):
        if _remaining() < 15:
            break
        result = _als_decompose(n, m, p, r)
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                return result
    
    return _schoolbook(n, m, p)
