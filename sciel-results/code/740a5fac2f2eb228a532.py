"""Matrix multiplication 2,4,6 rank minimization via randomized ALS search."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _build_target_tensor(n, m, p):
    """Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return target

def _compute_error(n, m, p, U, V, W, target):
    """Compute the Frobenius norm of the error."""
    # Reconstructed tensor
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    error = np.linalg.norm(target - recon)
    return error

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    result = np.array(arr, dtype=np.float64)
    doubled = result * 2
    rounded = np.round(doubled)
    close = np.abs(doubled - rounded) < tol
    result[close] = rounded[close] / 2
    return result

def _verify_exact(n, m, p, U, V, W, target):
    """Check if the decomposition is exact after snapping."""
    U_snapped = _snap_to_half_integers(U)
    V_snapped = _snap_to_half_integers(V)
    W_snapped = _snap_to_half_integers(W)
    
    recon = np.einsum('ac,bc,dc->abd', U_snapped, V_snapped, W_snapped)
    error = np.linalg.norm(target - recon)
    return error < 1e-9, U_snapped, V_snapped, W_snapped

def _als_step(n, m, p, U, V, W, target, max_iter=100, tol=1e-8):
    """Alternating Least Squares refinement."""
    rank = U.shape[1]
    target_flat = target.reshape(n * m, -1) # (n*m, m*p*p*n)
    
    for _ in range(max_iter):
        if _deadline_reached():
            break
            
        # Update U
        # T[a] = U[a] @ P_VW
        # P_VW[b, d, c] = V[b, c] * W[d, c]
        # P_VW_flat: (m*p*p*n, rank)
        P_VW = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
        # U[a] = T[a] @ pinv(P_VW.T)
        # Vectorized: U = T_flat @ pinv(P_VW.T)
        pinv_P_VW = np.linalg.pinv(P_VW.T) # (rank, m*p*p*n)
        U = target_flat @ pinv_P_VW.T # (n*m, rank)
        
        if _deadline_reached():
            break
            
        # Update V
        # T[a, b] = V[b] @ P_UW
        # P_UW[a, d, c] = U[a, c] * W[d, c]
        # P_UW_flat: (n*m*p*n, rank)
        P_UW = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
        pinv_P_UW = np.linalg.pinv(P_UW.T)
        # V[b] = T[:, b, :] @ pinv(P_UW.T)
        # T[:, b, :] is a vector of length n*m*p*n
        # Reshape T to (m*p, n*m*p*n)
        target_for_V = target.reshape(m * p, -1)
        V = target_for_V @ pinv_P_UW.T # (m*p, rank)
        
        if _deadline_reached():
            break
            
        # Update W
        # T[a, b, d] = W[d] @ P_UV
        # P_UV[a, b, c] = U[a, c] * V[b, c]
        # P_UV_flat: (n*m*m*p, rank)
        P_UV = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
        pinv_P_UV = np.linalg.pinv(P_UV.T)
        # W[d] = T[:, :, d] @ pinv(P_UV.T)
        # Reshape T to (p*n, n*m*m*p)
        target_for_W = target.reshape(p * n, -1)
        W = target_for_W @ pinv_P_UV.T # (p*n, rank)
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    target = _build_target_tensor(n, m, p)
    
    # Try to find a decomposition with rank < 48
    # Start with 42 (Strassen bound), go up to 47
    for rank in range(42, 48):
        if _deadline_reached():
            break
            
        # Multiple random restarts
        for restart in range(50):
            if _deadline_reached():
                break
                
            rng = np.random.RandomState(42 + rank * 100 + restart)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Refine
            U, V, W = _als_step(n, m, p, U, V, W, target, max_iter=100)
            
            # Check error
            error = _compute_error(n, m, p, U, V, W, target)
            
            if error < 1e-6:
                # Try to snap to half-integers
                is_exact, U_snapped, V_snapped, W_snapped = _verify_exact(n, m, p, U, V, W, target)
                if is_exact:
                    return {
                        "U": U_snapped.tolist(),
                        "V": V_snapped.tolist(),
                        "W": W_snapped.tolist()
                    }
                
    # Fallback to rank 48 (Schoolbook)
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
