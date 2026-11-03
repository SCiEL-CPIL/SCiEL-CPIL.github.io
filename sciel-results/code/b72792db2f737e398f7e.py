import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_exact(U, V, W, n, m, p, r, tol=1e-9):
    """Verify that the decomposition is exact (within tolerance)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, r

def _als_decompose(n, m, p, r, max_iter=1000, tol=1e-10, time_budget=10.0, seed=0):
    """
    Alternating Least Squares to find rank-r decomposition.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for l in range(r):
                        A[k, l] = V[j*p+k, l] * W[k*n+i, l]
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for l in range(r):
                        A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    for l in range(r):
                        A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                    error += (s - 1.0) ** 2
        error = np.sqrt(error)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return U, V, W, error

def _snap_to_half_int(arr, tol=1e-5):
    snapped = np.round(arr * 2) / 2.0
    if np.allclose(arr, snapped, atol=tol):
        return snapped, True
    return arr, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    fallback = {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
    
    best_result = fallback
    best_rank = rank_sb
    
    # Search ranks from 20 to 36
    for r in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
            
        for seed in range(20):
            if time.time() > deadline:
                break
                
            try:
                U, V, W, error = _als_decompose(n, m, p, r, max_iter=500, tol=1e-10, time_budget=2.0, seed=seed)
            except:
                continue
                
            if error < 1e-5:
                U_snap, ok1 = _snap_to_half_int(U)
                V_snap, ok2 = _snap_to_half_int(V)
                W_snap, ok3 = _snap_to_half_int(W)
                
                if ok1 and ok2 and ok3:
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, r, tol=1e-8):
                        result = {
                            "U": [[float(x) for x in row] for row in U_snap],
                            "V": [[float(x) for x in row] for row in V_snap],
                            "W": [[float(x) for x in row] for row in W_snap]
                        }
                        if r < best_rank:
                            best_rank = r
                            best_result = result
                        return best_result
        
        if best_rank < rank_sb:
            break
            
    return best_result
