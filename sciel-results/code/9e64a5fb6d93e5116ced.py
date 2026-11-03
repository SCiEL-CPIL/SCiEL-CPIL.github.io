import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)"""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_vector(T, n, m, p):
    """Flatten tensor to vector in order (i,j,k) -> index"""
    return T.flatten()

def _reconstruct_error(U, V, W, n, m, p, r):
    """Compute Frobenius norm of reconstruction error for rank-r decomposition"""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # For each (i,j,k), error = |sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] - 1|
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                error += (s - 1.0) ** 2
    return np.sqrt(error)

def _als_decompose(n, m, p, r, max_iter=500, tol=1e-8, time_budget=30.0):
    """
    Alternating Least Squares to find rank-r decomposition.
    Returns (U, V, W, error) or None if failed.
    """
    start_time = time.time()
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # We need: sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1 for all k
                # Let u_l = U[i*m+j, l]
                # For each k: sum_l u_l * (V[j*p+k,l] * W[k*n+i,l]) = 1
                # Build matrix A of size p x r
                A = np.zeros((p, r))
                for k in range(p):
                    for l in range(r):
                        A[k, l] = V[j*p+k, l] * W[k*n+i, l]
                # Solve A @ u = ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # We need: sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1 for all i
                # Let v_l = V[j*p+k, l]
                # For each i: sum_l v_l * (U[i*m+j,l] * W[k*n+i,l]) = 1
                # Build matrix A of size n x r
                A = np.zeros((n, r))
                for i in range(n):
                    for l in range(r):
                        A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                # Solve A @ v = ones(n)
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # We need: sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1 for all j
                # Let w_l = W[k*n+i, l]
                # For each j: sum_l w_l * (U[i*m+j,l] * V[j*p+k,l]) = 1
                # Build matrix A of size m x r
                A = np.zeros((m, r))
                for j in range(m):
                    for l in range(r):
                        A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                # Solve A @ w = ones(m)
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except:
                    pass
        
        # Compute error
        error = _reconstruct_error(U, V, W, n, m, p, r)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return U, V, W, error

def _snap_to_half_int(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough. Returns (snapped, success)."""
    snapped = np.round(arr * 2) / 2.0
    if np.allclose(arr, snapped, atol=tol):
        return snapped, True
    return arr, False

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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    fallback = {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
    
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But for matrix multiplication, the rank is at least the border rank
    # For 3x3x3, known rank is 23 (Strassen)
    # For 3x3x4, we expect rank around 23 + 3 = 26 or possibly lower
    
    # Search from rank 20 up to schoolbook
    best_result = fallback
    best_rank = rank_sb
    
    for r in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            try:
                U, V, W, error = _als_decompose(n, m, p, r, max_iter=300, tol=1e-10, time_budget=5.0)
            except:
                continue
            
            if error < 1e-6:
                # Snap to half-integers
                U_snap, ok1 = _snap_to_half_int(U, tol=1e-5)
                V_snap, ok2 = _snap_to_half_int(V, tol=1e-5)
                W_snap, ok3 = _snap_to_half_int(W, tol=1e-5)
                
                if ok1 and ok2 and ok3:
                    # Verify exact
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, r, tol=1e-9):
                        result = {
                            "U": [[float(x) for x in row] for row in U_snap],
                            "V": [[float(x) for x in row] for row in V_snap],
                            "W": [[float(x) for x in row] for row in W_snap]
                        }
                        if r < best_rank:
                            best_rank = r
                            best_result = result
                        return best_result
        
        # If we found a good solution, we can stop
        if best_rank < rank_sb:
            break
    
    return best_result
