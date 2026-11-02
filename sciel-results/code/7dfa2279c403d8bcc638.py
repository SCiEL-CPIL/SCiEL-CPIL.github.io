<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition is always valid.
    # Rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if i,j,k are all in range, 0 otherwise
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] where C[i,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    
    # Check: for each r, the outer product structure
    # The tensor M[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # should equal 1 for all 0<=i<n, 0<=j<m, 0<=k<p
    
    # Efficient check using einsum
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute M[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    M = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # M should be all ones
    if not np.allclose(M, 1.0, atol=1e-6):
        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline, seed=42):
    """Try to find a decomposition of target_rank by projecting and optimizing."""
    rng = np.random.RandomState(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank]
    
    # Strategy: try to find a rank-target_rank subspace
    # Use random projections and least squares
    best_residual = np.inf
    best_result = None
    
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        # Random projection
        P = rng.randn(current_rank, target_rank)
        # Optimize U, V, W to fit the tensor
        # This is a hard problem, so use iterative approach
        
        # Initialize with projected matrices
        U_init = U @ P
        V_init = V @ P
        W_init = W @ P
        
        # Iterative refinement
        for it in range(50):
            if time.time() > deadline - 5:
                break
            
            # Fix V and W, optimize U
            # M[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed V, W: U[i*m+j,:] should satisfy:
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all j,k
            # This is: U[i*m+j,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) ... 
            # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let B[j,k] = V[j*p+k,:] * W[k*n+i,:] (element-wise product, shape (m,p,target_rank))
            # Then sum_r U[i*m+j,r] * B[j,k,r] = 1
            # This is a linear system in U[i*m+j,:] for each i,j
            
            U_new = np.zeros_like(U_init)
            for i in range(n):
                for j in range(m):
                    # Build system: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for k=0..p-1
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = V[j*p:(j+1)*p, :] * W[:, i, :]  # (p, target_rank)
                    # Solve A @ x = 1 (vector of ones)
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = x
                    except:
                        U_new[i*m+j, :] = U_init[i*m+j, :]
            
            # Fix U and W, optimize V
            V_new = np.zeros_like(V_init)
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for i=0..n-1
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A = U[:, j, :] * W[k, :, :]  # (n, target_rank)
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = x
                    except:
                        V_new[j*p+k, :] = V_init[j*p+k, :]
            
            # Fix U and V, optimize W
            W_new = np.zeros_like(W_init)
            for k in range(p):
                for i in range(n):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for j=0..m-1
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = U[i, :, :] * V[:, k, :]  # (m, target_rank)
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = x
                    except:
                        W_new[k*n+i, :] = W_init[k*n+i, :]
            
            U_init = U_new
            V_init = V_new
            W_init = W_new
            
            # Check residual
            U_r = U_init.reshape(n, m, target_rank)
            V_r = V_init.reshape(m, p, target_rank)
            W_r = W_init.reshape(p, n, target_rank)
            M = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = np.max(np.abs(M - 1.0))
            
            if residual < best_residual:
                best_residual = residual
                best_result = (U_init.copy(), V_init.copy(), W_init.copy())
            
            if residual < 1e-6:
                break
        
        if best_residual < 1e-6:
            break
    
    if best_result is not None and best_residual < 1e-6:
        return best_result
    return None

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    for target_rank in range(rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline, seed=target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if _validate_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
