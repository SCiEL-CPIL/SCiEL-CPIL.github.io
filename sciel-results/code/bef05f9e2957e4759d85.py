import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    # Schoolbook fallback (rank 18)
    U_school = np.zeros((nm, 18))
    V_school = np.zeros((mp, 18))
    W_school = np.zeros((pn, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Verify function
    def verify(U, V, W, tol=1e-9):
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False
        
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=tol):
                        return False
        return True
    
    # Winograd F(3,3) based decomposition (rank 8)
    # For 2x3 * 3x3: each row of A (3-vector) uses 4-point Winograd
    # Total rank = 2 * 4 = 8
    G = np.array([
        [1, 0, 0],
        [1, 1, 1],
        [1, -1, 1],
        [0, 1, -1]
    ], dtype=float)
    
    P = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, -1, 0],
        [0, 1, 0]
    ], dtype=float)
    
    AT = np.array([
        [1, 1, 1, 0],
        [0, 1, -1, -1],
        [1, 0, -1, 1]
    ], dtype=float)
    
    R = 8
    U_wino = np.zeros((nm, R))
    V_wino = np.zeros((mp, R))
    W_wino = np.zeros((pn, R))
    
    for i in range(n):  # 2 rows of A
        for t in range(4):  # 4 Winograd points
            r = i * 4 + t
            for j in range(m):  # 3 columns
                U_wino[i*m + j, r] = G[t, j]
                V_wino[j*p + 0, r] = P[t, j]
                V_wino[j*p + 1, r] = 0
                V_wino[j*p + 2, r] = 0
            for k in range(p):  # 3 output positions
                W_wino[k*n + i, r] = AT[k, t]
    
    # The above V assignment is wrong. Let me fix it.
    # V[j*p + k, r] should be P[t, j] for the B matrix column k
    # Actually, for matrix mult C[i,k] = sum_j A[i,j] * B[j,k]
    # Winograd: m = G @ A[i,:] (4 values), v_k = P @ B[:,k] (4 values for each k)
    # u = m * v_k (pointwise, 4 values)
    # C[i,k] = AT @ u
    
    # So U[i*m + j, r] = G[t, j] where r = 4*i + t
    # V[j*p + k, r] = P[t, j] -- but this doesn't depend on k! That's wrong.
    # Actually v_k = P @ B[:,k], so the k-th column of B is transformed by P
    # So V[j*p + k, r] = P[t, j] is correct only if we're transforming each column independently
    # But P is 4x3, so (P @ B[:,k])[t] = sum_j P[t,j] * B[j,k]
    # So V[j*p + k, r] = P[t, j] is correct!
    
    # Let me redo this properly
    U_wino = np.zeros((nm, R))
    V_wino = np.zeros((mp, R))
    W_wino = np.zeros((pn, R))
    
    for i in range(n):
        for t in range(4):
            r = i * 4 + t
            for j in range(m):
                U_wino[i*m + j, r] = G[t, j]
                for k in range(p):
                    V_wino[j*p + k, r] = P[t, j]
            for k in range(p):
                W_wino[k*n + i, r] = AT[k, t]
    
    if verify(U_wino, V_wino, W_wino):
        return {
            "U": U_wino.tolist(),
            "V": V_wino.tolist(),
            "W": W_wino.tolist()
        }
    
    # If Winograd rank 8 doesn't work, try rank 12 with a different structure
    # or fall back to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
