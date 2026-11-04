"""Matrix multiplication 2x3x6 tensor decomposition with rank optimization."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_DEADLINE = 135.0  # safety margin under 150s kill
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _als_decompose(n, m, p, r, seed=0, iters=2000, tol=1e-10):
    """
    Alternating least squares to find U,V,W of rank r decomposing the tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Constraint: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = np.ones((p * n, r))  # initialize W with ones for faster convergence
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    err = float('inf')
    
    for it in range(iters):
        if _remaining() < 10.0:
            break
        
        # Solve for U: for each (i,j), find U[i*m+j,:] such that
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                b = np.ones(p)
                # Use lstsq for stability
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = sol
        
        # Solve for V: for each (j,k), find V[j*p+k,:] such that
        # sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                b = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = sol
        
        # Solve for W: for each (k,i), find W[k*n+i,:] such that
        # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                b = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = sol
        
        # Check convergence
        err = 0.0
        for (i, j, k) in triples:
            s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
            err += (s - 1.0) ** 2
        err /= len(triples)
        
        if err < tol:
            break
    
    return U, V, W, err

def _try_kronecker_structure(n, m, p):
    """
    Try to exploit Kronecker product structure.
    2x3x6: note that 6 = 2*3, so we can think of this as:
    - Split the output dimension p=6 into two parts of size 3
    - Or use the factorization to create a structured decomposition
    
    Actually, let's try a different approach: 
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    We want rank-r decomposition.
    
    One idea: if we can find matrices such that the outer product structure
    allows sharing. For n=2, m=3, p=6, let's try rank values below 36.
    """
    n, m, p = 2, 3, 6
    
    # Try various rank values
    best_result = None
    best_rank = n * m * p
    
    for r in range(12, best_rank):  # try from low ranks up
        if _remaining() < 15.0:
            break
        
        # Try multiple seeds for this rank
        for seed in range(5):
            if _remaining() < 10.0:
                break
            
            U, V, W, err = _als_decompose(n, m, p, r, seed=seed, iters=500, tol=1e-10)
            
            if err < 1e-6:
                # Verify more carefully
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-5):
                    if r < best_rank:
                        best_rank = r
                        best_result = (U.copy(), V.copy(), W.copy())
                        # Found a valid decomposition, try to go lower
                        break
        
        if best_result is not None and best_rank < n * m * p:
            # Found improvement, continue searching for even lower
            if _remaining() < 30.0:
                break
    
    return best_result, best_rank

def construct():
    """
    Return a valid tensor decomposition for 2x3x6 matrix multiplication.
    Attempts to find rank below schoolbook (36) using ALS optimization.
    Falls back to schoolbook if no improvement found.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as guaranteed fallback
    Us, Vs, Ws = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = Us, Vs, Ws
    
    # Try to find a lower rank decomposition
    if _remaining() > 20.0:
        result, found_rank = _try_kronecker_structure(n, m, p)
        if result is not None and found_rank < best_rank:
            best_U, best_V, best_W = result
            best_rank = found_rank
    
    # Final verification and return
    if not _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-5):
        # Fallback to schoolbook if something went wrong
        best_U, best_V, best_W = Us, Vs, Ws
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
