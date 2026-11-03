"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, tol=1e-9):
    """Check the decomposition identity."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.einsum('r,r,r->', U[i*m+j], V[j*p+k], W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_structured_decomposition(n, m, p, deadline):
    """
    Try structured decompositions based on algebraic patterns.
    For 2x3x6, try to exploit the small dimensions.
    """
    rng = np.random.RandomState(42)
    n_rows_u = n * m  # 6
    n_rows_v = m * p  # 18
    n_rows_w = p * n  # 12
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 35 down to 28
    for r in range(35, 27, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook pattern for first min(r, 36) columns
            U = np.zeros((n_rows_u, r))
            V = np.zeros((n_rows_v, r))
            W = np.zeros((n_rows_w, r))
            
            fill_cols = min(r, n * m * p)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                        col += 1
            
            # Add small random perturbations to all columns
            U += rng.randn(n_rows_u, r) * 0.01
            V += rng.randn(n_rows_v, r) * 0.01
            W += rng.randn(n_rows_w, r) * 0.01
            
            # Try a few gradient descent steps
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Compute errors
                err = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err[i,j,k] = float(np.einsum('r,r,r->', U[i*m+j], V[j*p+k], W[k*n+i])) - 1.0
                
                max_err = np.max(np.abs(err))
                if max_err < 1e-10:
                    break
                
                # Gradient descent
                lr = 0.005
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = err[i,j,k]
                            U[i*m+j] -= lr * e * (V[j*p+k] * W[k*n+i])
                            V[j*p+k] -= lr * e * (U[i*m+j] * W[k*n+i])
                            W[k*n+i] -= lr * e * (U[i*m+j] * V[j*p+k])
            
            if _verify(U, V, W, tol=1e-9):
                best_result = (U, V, W)
                best_rank = r
                break
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130  # Safety margin before 150s limit

    # Build the guaranteed-valid schoolbook decomposition (rank 36)
    U, V, W = _build_schoolbook()
    
    # Verify the schoolbook decomposition
    if not _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try to find lower rank decompositions using structured approach
    result = _try_structured_decomposition(n, m, p, deadline)
    if result is not None:
        U_r, V_r, W_r = result
        if _verify(U_r, V_r, W_r, tol=1e-9):
            best_rank = U_r.shape[1]
            best_U, best_V, best_W = U_r, V_r, W_r

    # Final check: ensure we still have a valid decomposition
    if not _verify(best_U, best_V, best_W):
        # Fallback to original schoolbook
        best_U, best_V, best_W = _build_schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
