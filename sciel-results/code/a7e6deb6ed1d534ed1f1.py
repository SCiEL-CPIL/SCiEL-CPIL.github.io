import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n=2, m=3, p=5, tol=1e-6):
    """Verify that U, V, W form a valid tensor decomposition."""
    R = U.shape[1]
    # Check: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == delta_{j,j'}
    # for all i, j, j', k
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + i, r]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _schoolbook(n=2, m=3, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _compute_error(U, V, W, n=2, m=3, p=5):
    """Compute the sum of squared errors of the decomposition using vectorized ops."""
    R = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[jp * p + k] * W[k * n + i])
                    expected = 1.0 if j == jp else 0.0
                    error += (s - expected) ** 2
    return error

def _als_decompose(n, m, p, R, max_iters=500, time_limit=8.0, seed=None):
    """
    Attempt to find a rank-R decomposition using Alternating Least Squares.
    """
    start_time = time.time()
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Normalize columns to prevent scale drift
    for r in range(R):
        norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
        if norm > 1e-10:
            scale = norm ** (1/3)
            U[:, r] /= scale
            V[:, r] /= scale
            W[:, r] /= scale

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((m * p, R))
                b = np.zeros(m * p)
                for jp in range(m):
                    for k in range(p):
                        idx = jp * p + k
                        M[idx, :] = V[jp * p + k] * W[k * n + i]
                        b[idx] = 1.0 if jp == j else 0.0
                
                try:
                    u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    U[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n * m, R))
                b = np.zeros(n * m)
                for i in range(n):
                    for jp in range(m):
                        idx = i * m + jp
                        M[idx, :] = U[i * m + jp] * W[k * n + i]
                        b[idx] = 1.0 if jp == j else 0.0
                
                try:
                    v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    V[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m * p, R))
                b = np.zeros(m * p)
                for j in range(m):
                    for kp in range(p):
                        idx = j * p + kp
                        M[idx, :] = U[i * m + j] * V[j * p + kp]
                        b[idx] = 1.0 if kp == k else 0.0
                
                try:
                    w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    W[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    pass

        # Check convergence
        error = _compute_error(U, V, W, n, m, p)
        if error < 1e-10:
            return U, V, W, True

    return U, V, W, False

def _polish(U, V, W, n=2, m=3, p=5, max_iters=200, time_limit=5.0):
    """
    High-precision polish: run more ALS iterations with tighter tolerance.
    """
    start_time = time.time()
    R = U.shape[1]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((m * p, R))
                b = np.zeros(m * p)
                for jp in range(m):
                    for k in range(p):
                        idx = jp * p + k
                        M[idx, :] = V[jp * p + k] * W[k * n + i]
                        b[idx] = 1.0 if jp == j else 0.0
                
                try:
                    u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    U[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n * m, R))
                b = np.zeros(n * m)
                for i in range(n):
                    for jp in range(m):
                        idx = i * m + jp
                        M[idx, :] = U[i * m + jp] * W[k * n + i]
                        b[idx] = 1.0 if jp == j else 0.0
                
                try:
                    v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    V[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m * p, R))
                b = np.zeros(m * p)
                for j in range(m):
                    for kp in range(p):
                        idx = j * p + kp
                        M[idx, :] = U[i * m + j] * V[j * p + kp]
                        b[idx] = 1.0 if kp == k else 0.0
                
                try:
                    w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    W[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    pass

        error = _compute_error(U, V, W, n, m, p)
        if error < 1e-12:
            break

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 30
    
    # Try to find lower rank solutions using ALS
    # Start from rank 18, which is more likely to be feasible
    for R in range(18, best_rank):
        if time.time() - start_time > 130:
            break
            
        # Try multiple random restarts for this rank with different seeds
        found = False
        for attempt in range(15):
            if time.time() - start_time > 135:
                break
            
            seed = 42 + attempt * 7
            U, V, W, success = _als_decompose(n, m, p, R, max_iters=300, time_limit=6.0, seed=seed)
            
            if success:
                # Polish to drive error to machine precision
                U, V, W = _polish(U, V, W, n, m, p, max_iters=100, time_limit=3.0)
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
