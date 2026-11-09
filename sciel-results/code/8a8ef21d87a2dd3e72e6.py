"""Search for a low-rank exact tensor decomposition of the 3x4 by 4x4 matmul."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 140.0  # safety margin under the 150s kill

def _verify(U, V, W):
    """Check whether factor matrices exactly decompose the matmul tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    n, m, p = DIMENSIONS
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build tensor via einsum: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Reshape U to (n,m,r), V to (m,p,r), W to (p,n,r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook():
    """Return the schoolbook rank-48 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _als_optimize(U, V, W, n, m, p, max_iter=100, tol=1e-10):
    """Alternating least squares optimization for tensor decomposition."""
    r = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    prev_norm = np.inf
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # Solve for U: fix V, W
        # For each (i,j): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
        # A_U[i,j,k,c] = V[j,k,c] * W[k,i,c]
        A_U = np.einsum('jkc,kic->ijkc', V3, W3)
        for i in range(n):
            for j in range(m):
                Aj = A_U[i, j]  # (p, r)
                b = np.ones(p)
                try:
                    U[i*m+j], _, _, _ = np.linalg.lstsq(Aj, b, rcond=None)
                except:
                    pass
        
        U3 = U.reshape(n, m, r)
        
        # Solve for V: fix U, W
        # For each (j,k): sum_c V[j,k,c] * U[i,j,c] * W[k,i,c] = 1 for all i
        # A_V[j,k,i,c] = U[i,j,c] * W[k,i,c]
        A_V = np.einsum('ijc,kic->jkic', U3, W3)
        for j in range(m):
            for k in range(p):
                Ak = A_V[j, k]  # (n, r)
                b = np.ones(n)
                try:
                    V[j*p+k], _, _, _ = np.linalg.lstsq(Ak, b, rcond=None)
                except:
                    pass
        
        V3 = V.reshape(m, p, r)
        
        # Solve for W: fix U, V
        # For each (k,i): sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] = 1 for all j
        # A_W[k,i,j,c] = U[i,j,c] * V[j,k,c]
        A_W = np.einsum('ijc,jkc->kijc', U3, V3)
        for k in range(p):
            for i in range(n):
                Ak = A_W[k, i]  # (m, r)
                b = np.ones(m)
                try:
                    W[k*n+i], _, _, _ = np.linalg.lstsq(Ak, b, rcond=None)
                except:
                    pass
        
        W3 = W.reshape(p, n, r)
        
        # Check convergence
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        resid = T - 1.0
        norm = np.linalg.norm(resid)
        if norm < tol:
            break
        if abs(prev_norm - norm) < 1e-15:
            break
        prev_norm = norm
    
    return U, V, W

def _round_to_half(X):
    """Round to nearest half-integer."""
    return np.round(X * 2.0) / 2.0

def _try_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Try to polish a near-exact decomposition using ALS and rounding."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    # Run ALS to convergence
    U, V, W = _als_optimize(U, V, W, n, m, p, max_iter=max_iter)
    
    # Try rounding to integers first
    U_int = np.round(U).astype(int)
    V_int = np.round(V).astype(int)
    W_int = np.round(W).astype(int)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    # Try rounding to half-integers
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try small perturbations around the rounded values
    for delta in [0.25, -0.25, 0.5, -0.5]:
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    # This is too expensive, skip
                    pass
        break
    
    return None

def _random_init(rank, n, m, p, seed=None):
    """Random initialization."""
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    return U, V, W

def _sparse_init(rank, n, m, p, seed=None):
    """Sparse random initialization."""
    if seed is not None:
        np.random.seed(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c in range(rank):
        U[np.random.randint(n * m), c] = np.random.choice([-1.0, 1.0])
        V[np.random.randint(m * p), c] = np.random.choice([-1.0, 1.0])
        W[np.random.randint(p * n), c] = np.random.choice([-1.0, 1.0])
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p  # 48
    
    # Try ranks from 16 upward
    for rank in range(16, best_rank + 1):
        if time.time() > DEADLINE - 10:
            break
        max_attempts = 40 if rank <= 25 else 20
        found = False
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 5:
                break
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                U, V, W = _random_init(rank, n, m, p, seed=42 + attempt)
            elif attempt % 3 == 1:
                U, V, W = _sparse_init(rank, n, m, p, seed=100 + attempt)
            else:
                # Mixed initialization
                U, V, W = _random_init(rank, n, m, p, seed=200 + attempt)
                # Add some structure
                for c in range(min(rank, 12)):
                    i = c % n
                    j = c % m
                    k = c % p
                    U[i * m + j, c] += 1.0
                    V[j * p + k, c] += 1.0
                    W[k * n + i, c] += 1.0
            
            # Run ALS optimization
            U, V, W = _als_optimize(U, V, W, n, m, p, max_iter=80)
            
            # Polish
            result = _try_polish(U, V, W, n, m, p, max_iter=30)
            if result is not None:
                U_l, V_l, W_l = result
                if _verify(U_l, V_l, W_l):
                    best = {"U": U_l, "V": V_l, "W": W_l}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
    print("Verified:", _verify(result["U"], result["V"], result["W"]))
