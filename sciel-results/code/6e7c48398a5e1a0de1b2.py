import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    err = np.max(np.abs(T - Texp))
    return err < tol

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
    return {"U": U, "V": V, "W": W}

def _als_decompose(T1, T2, T3, n1, n2, n3, rank, iterations=100, tol=1e-10, deadline=None):
    """
    T1: n1 x (n2*n3)
    T2: n2 x (n1*n3)
    T3: n3 x (n1*n2)
    """
    U = np.random.randn(n1, rank) * 0.1
    V = np.random.randn(n2, rank) * 0.1
    W = np.random.randn(n3, rank) * 0.1
    
    for it in range(iterations):
        if deadline and time.time() > deadline:
            return None
            
        # Update U
        # T1 = U * (V odot W)^T
        # Z = V odot W has shape (n2*n3, rank)
        Z = (V[:, :, None] * W[None, :, :]).reshape(n2 * n3, rank)
        U_new = np.linalg.lstsq(Z.T, T1, rcond=None)[0].T
        U = U_new
        
        # Update V
        # T2 = V * (U odot W)^T
        Z = (U[:, :, None] * W[None, :, :]).reshape(n1 * n3, rank)
        V_new = np.linalg.lstsq(Z.T, T2, rcond=None)[0].T
        V = V_new
        
        # Update W
        # T3 = W * (U odot V)^T
        Z = (U[:, :, None] * V[None, :, :]).reshape(n1 * n2, rank)
        W_new = np.linalg.lstsq(Z.T, T3, rcond=None)[0].T
        W = W_new
        
        # Check convergence
        # Reconstruct T1
        Z = (V[:, :, None] * W[None, :, :]).reshape(n2 * n3, rank)
        T1_rec = U @ Z.T
        err = np.linalg.norm(T1 - T1_rec)
        if err < tol:
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Prepare unfoldings
    n1, n2, n3 = n*m, m*p, p*n
    T1 = np.zeros((n1, n2*n3))
    T2 = np.zeros((n2, n1*n3))
    T3 = np.zeros((n3, n1*n2))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T1[a, b*n3+c] = 1.0
                T2[b, a*n3+c] = 1.0
                T3[c, a*n2+b] = 1.0
                
    # Try ranks from lower bound to 55
    # Lower bound is max(n1, n2, n3) = 28
    for rank in range(28, 56):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts for small ranks
        restarts = 5 if rank < 35 else 2
        for _ in range(restarts):
            res = _als_decompose(T1, T2, T3, n1, n2, n3, rank, iterations=200, deadline=deadline)
            if res is not None:
                U, V, W = res
                # Verify
                if _verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    return _schoolbook(n, m, p)
