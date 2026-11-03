import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(str(u)) for u in row] for row in U]
    Vf = [[Fraction(str(v)) for v in row] for row in V]
    Wf = [[Fraction(str(w)) for w in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _snap_to_half(x, tol=1e-6):
    k = round(2 * x)
    if abs(2 * x - k) < tol:
        return k / 2.0
    return None

def _try_snap(U, V, W):
    """Try to snap all entries to half-integers and verify."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    Us = [[0.0]*rank for _ in range(n*m)]
    Vs = [[0.0]*rank for _ in range(m*p)]
    Ws = [[0.0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank):
            val = _snap_to_half(U[i][c])
            if val is None:
                return None
            Us[i][c] = val
    for i in range(m*p):
        for c in range(rank):
            val = _snap_to_half(V[i][c])
            if val is None:
                return None
            Vs[i][c] = val
    for i in range(p*n):
        for c in range(rank):
            val = _snap_to_half(W[i][c])
            if val is None:
                return None
            Ws[i][c] = val
    
    if _verify_exact(Us, Vs, Ws):
        return Us, Vs, Ws
    return None

def _als_decompose(n, m, p, rank, max_iter=300, seed=0):
    """ALS decomposition of the matrix multiplication tensor."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0)**2
        if err < 1e-12:
            break
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 16
    for rank in range(16, 49):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            U, V, W = _als_decompose(n, m, p, rank, max_iter=200, seed=seed)
            result = _try_snap(U, V, W)
            if result is not None:
                Us, Vs, Ws = result
                return {"U": Us, "V": Vs, "W": Ws}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
