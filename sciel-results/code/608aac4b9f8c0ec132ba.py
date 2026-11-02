def _verify_arrays(U, V, W, n, m, p, tol=1e-9):
    """Check tensor decomposition validity using numpy arrays."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_64(n, m, p, deadline):
    import numpy as np
    R = 64
    np.random.seed(42)
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        for c in range(R):
            # Sparse random initialization
            for _ in range(4):
                idx_v = np.random.randint(0, m*p)
                V[idx_v, c] = np.random.choice(vals)
                idx_w = np.random.randint(0, p*n)
                W[idx_w, c] = np.random.choice(vals)
                
        A = np.zeros((n*m*p, n*m*R))
        b = np.ones(n*m*p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_start = (i*m + j) * R
                    # Vectorize the coefficient calculation
                    coeffs = V[j*p + k, :] * W[k*n + i, :]
                    A[idx, row_start:row_start+R] = coeffs
                    idx += 1
                    
        try:
            U_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        U = U_flat.reshape(n*m, R)
        U_rounded = np.round(U * 2) / 2
        
        if _verify_arrays(U_rounded, V, W, n, m, p, tol=1e-9):
            return U_rounded.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank 64
    result = _try_rank_64(n, m, p, deadline)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
