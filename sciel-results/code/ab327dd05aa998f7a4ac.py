import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _strassen_2x2x2(j1, j2, k1, k2, n=2, m=5, p=6):
    """Return U, V, W vectors for Strassen decomposition of 2x2x2 block."""
    U = []
    V = []
    W = []
    
    # M1 = (A[0,j1]+A[1,j2]) * (B[j1,k1]+B[j2,k2])
    u = [0] * (n * m)
    u[0 * m + j1] = 1
    u[1 * m + j2] = 1
    v = [0] * (m * p)
    v[j1 * p + k1] = 1
    v[j2 * p + k2] = 1
    w = [0] * (p * n)
    w[k1 * n + 0] = 1
    w[k2 * n + 1] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M2 = (A[1,j1]+A[1,j2]) * B[j1,k1]
    u = [0] * (n * m)
    u[1 * m + j1] = 1
    u[1 * m + j2] = 1
    v = [0] * (m * p)
    v[j1 * p + k1] = 1
    w = [0] * (p * n)
    w[k1 * n + 1] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M3 = A[0,j1] * (B[j1,k2]-B[j2,k2])
    u = [0] * (n * m)
    u[0 * m + j1] = 1
    v = [0] * (m * p)
    v[j1 * p + k2] = 1
    v[j2 * p + k2] = -1
    w = [0] * (p * n)
    w[k2 * n + 0] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M4 = A[1,j2] * (B[j2,k1]-B[j1,k1])
    u = [0] * (n * m)
    u[1 * m + j2] = 1
    v = [0] * (m * p)
    v[j2 * p + k1] = 1
    v[j1 * p + k1] = -1
    w = [0] * (p * n)
    w[k1 * n + 1] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M5 = (A[0,j1]+A[0,j2]) * B[j2,k2]
    u = [0] * (n * m)
    u[0 * m + j1] = 1
    u[0 * m + j2] = 1
    v = [0] * (m * p)
    v[j2 * p + k2] = 1
    w = [0] * (p * n)
    w[k2 * n + 0] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M6 = (A[1,j1]-A[1,j2]) * (B[j1,k1]+B[j1,k2])
    u = [0] * (n * m)
    u[1 * m + j1] = 1
    u[1 * m + j2] = -1
    v = [0] * (m * p)
    v[j1 * p + k1] = 1
    v[j1 * p + k2] = 1
    w = [0] * (p * n)
    w[k2 * n + 1] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M7 = (A[0,j2]-A[1,j2]) * (B[j2,k1]+B[j2,k2])
    u = [0] * (n * m)
    u[0 * m + j2] = 1
    u[1 * m + j2] = -1
    v = [0] * (m * p)
    v[j2 * p + k1] = 1
    v[j2 * p + k2] = 1
    w = [0] * (p * n)
    w[k1 * n + 0] = 1
    U.append(u); V.append(v); W.append(w)
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Verify the decomposition is correct."""
    rank = len(U)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomposed = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_decomposed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    return np.allclose(C_decomposed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Split m=5 into [2, 2, 1], p=6 into [2, 2, 2]
    m_groups = [(0, 2), (2, 4), (4, 5)]
    p_groups = [(0, 2), (2, 4), (4, 6)]
    
    U_all = []
    V_all = []
    W_all = []
    
    for (m_start, m_end) in m_groups:
        for (p_start, p_end) in p_groups:
            m_size = m_end - m_start
            p_size = p_end - p_start
            
            if m_size == 2 and p_size == 2:
                j1, j2 = m_start, m_start + 1
                k1, k2 = p_start, p_start + 1
                U, V, W = _strassen_2x2x2(j1, j2, k1, k2)
                U_all.extend(U)
                V_all.extend(V)
                W_all.extend(W)
            else:
                # Schoolbook
                for i in range(n):
                    for j in range(m_start, m_end):
                        for k in range(p_start, p_end):
                            u = [0] * (n * m)
                            u[i * m + j] = 1
                            v = [0] * (m * p)
                            v[j * p + k] = 1
                            w = [0] * (p * n)
                            w[k * n + i] = 1
                            U_all.append(u)
                            V_all.append(v)
                            W_all.append(w)
    
    # Verify
    if not _verify_decomposition(U_all, V_all, W_all):
        # Fall back to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_all, "V": V_all, "W": W_all}
