"""Matrix multiplication 2,4,6 rank minimization via randomized SVD + iterative refinement."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _tensor_index(n, m, p):
    """Return list of (i,j,k) triples for the multiplication tensor."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def _build_tensors(n, m, p, U, V, W):
    """Build the factor matrices from the decomposition."""
    rank = len(U[0]) if U else 0
    return U, V, W

def _compute_tensor_error(n, m, p, U, V, W):
    """Compute the Frobenius norm of the error between the tensor and its decomposition."""
    rank = len(U[0])
    # Build the full tensor T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor
    # Actually, the multiplication tensor has T[i,j,k] = 1 iff the output (i,k) depends on input (i,j) and (j,k)
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j, V index: j*p + k, W index: k*n + i
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute the decomposition tensor
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstructed tensor: sum_c U[a,c] * V[b,c] * W[c2,c]
    # where a = i*m+j, b = j*p+k, c2 = k*n+i
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error

def _try_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    error = _compute_tensor_error(n, m, p, U, V, W)
    return error < tol, error

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random decomposition of given rank."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _refine_decomposition(n, m, p, U, V, W, max_iter=50, tol=1e-9):
    """Iteratively refine the decomposition using alternating least squares."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    rank = U_arr.shape[1]
    
    for iteration in range(max_iter):
        if _deadline_reached():
            break
        
        # Update U: for each a, U[a,:] should satisfy sum_c U[a,c]*V[b,c]*W[d,c] = target[a,b,d]
        # Fix V and W, solve for U
        # For each a: sum_c U[a,c] * (V[b,c] * W[d,c]) = target[a,b,d]
        # This is: U[a,:] @ (V[b,:] * W[d,:])^T = target[a,b,d]
        # So: U[a,:] = target[a,:,:] @ (V*W)^+  where V*W is element-wise product reshaped
        
        # V*W has shape (m*p, p*n, rank) -> for each (b,d), we have V[b,:] * W[d,:]
        # We need: U[a,:] such that U[a,:] @ M = target[a,:,:] where M[b,d,c] = V[b,c]*W[d,c]
        # Reshape: M_flat has shape (m*p*p*n, rank), target_flat has shape (m*p*p*n,)
        
        M = V_arr[:, None, :] * W_arr[None, :, :]  # shape (m*p, p*n, rank)
        M_flat = M.reshape(-1, rank)  # shape (m*p*p*n, rank)
        target_flat = target.reshape(-1)  # shape (m*p*p*n,)
        
        # Solve: U[a,:] @ M_flat.T = target_flat for each a
        # U[a,:] = (M_flat @ M_flat.T)^-1 @ M_flat @ target_flat
        # But this is the same for all a! That can't be right.
        
        # Actually, let me reconsider. The tensor is T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
        # For fixed a, T[a,:,:] = sum_c U[a,c] * (V[:,c] * W[:,c])
        # So T[a,:,:] = U[a,:] @ (V*W)^T where (V*W) has shape (m*p*p*n, rank)
        # Wait, V[:,c] * W[:,c] has shape (m*p, p*n) for each c
        # So T[a,:,:] = sum_c U[a,c] * (V[:,c] * W[:,c])
        # This means: T[a,:] = U[a,:] @ P where P[c, b, d] = V[b,c] * W[d,c]
        # Reshape P to (rank, m*p*p*n): P_flat[c, b*p*n+d] = V[b,c] * W[d,c]
        
        P_flat = M.reshape(rank, -1)  # shape (rank, m*p*p*n)
        # T[a,:] = U[a,:] @ P_flat.T
        # So U[a,:] = T[a,:] @ pinv(P_flat.T)
        
        # P_flat.T has shape (m*p*p*n, rank)
        # U[a,:] = T[a,:] @ pinv(P_flat.T)
        # But T[a,:] has shape (m*p*p*n,) and pinv(P_flat.T) has shape (rank, m*p*p*n)
        # So U[a,:] = (P_flat.T @ pinv(P_flat.T).T @ T[a,:].T).T ... this is getting messy
        
        # Let me use a different approach: solve the linear system
        # For each a: U[a,:] @ P_flat.T = target[a, :].reshape(-1)
        # Let X = U[a,:], then X @ P_flat.T = y where y = target[a,:,:].reshape(-1)
        # X = y @ pinv(P_flat.T)
        
        P_T = P_flat.T  # shape (m*p*p*n, rank)
        P_pinv = np.linalg.pinv(P_T)  # shape (rank, m*p*p*n)
        
        # For each a: U[a,:] = target[a,:,:].reshape(-1) @ P_pinv
        new_U = np.zeros((n * m, rank))
        for a in range(n * m):
            y = target[a, :, :].reshape(-1)
            new_U[a, :] = y @ P_pinv
        
        U_arr = new_U
        
        if _deadline_reached():
            break
        
        # Update V: for each b, V[b,:] @ (U*W) = target[:,b,:]
        # T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
        # For fixed b: T[:,b,:] = V[b,:] @ Q where Q[c,a,d] = U[a,c] * W[d,c]
        Q_flat = (U_arr[:, None, :] * W_arr[None, :, :]).reshape(rank, -1)  # (rank, n*m*p*n)
        Q_T = Q_flat.T  # (n*m*p*n, rank)
        Q_pinv = np.linalg.pinv(Q_T)
        
        new_V = np.zeros((m * p, rank))
        for b in range(m * p):
            y = target[:, b, :].reshape(-1)
            new_V[b, :] = y @ Q_pinv
        
        V_arr = new_V
        
        if _deadline_reached():
            break
        
        # Update W: for each d, W[d,:] @ (U*V) = target[:,:,d]
        # T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
        # For fixed d: T[:,:,d] = W[d,:] @ R where R[c,a,b] = U[a,c] * V[b,c]
        R_flat = (U_arr[:, None, :] * V_arr[None, :, :]).reshape(rank, -1)  # (rank, n*m*m*p)
        R_T = R_flat.T  # (n*m*m*p, rank)
        R_pinv = np.linalg.pinv(R_T)
        
        new_W = np.zeros((p * n, rank))
        for d in range(p * n):
            y = target[:, :, d].reshape(-1)
            new_W[d, :] = y @ R_pinv
        
        W_arr = new_W
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    result = np.array(arr, dtype=np.float64)
    # Check if values are close to multiples of 0.5
    doubled = result * 2
    rounded = np.round(doubled)
    close = np.abs(doubled - rounded) < tol
    result[close] = rounded[close] / 2
    return result

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
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
