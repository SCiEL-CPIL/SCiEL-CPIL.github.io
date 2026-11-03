import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _build_tensor():
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 if (i,j,k) is a valid multiplication term
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # The tensor for matrix multiplication: T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrix(T, mode):
    """Reshape tensor to matrix form for ALS."""
    n, m, p = T.shape
    if mode == 0:  # flatten (i) vs (j,k)
        return T.reshape(n, m * p)
    elif mode == 1:  # flatten (j) vs (i,k)
        return T.transpose(1, 0, 2).reshape(m, n * p)
    elif mode == 2:  # flatten (k) vs (i,j)
        return T.transpose(2, 0, 1).reshape(p, n * m)

def _matrix_to_tensor(M, mode, n, m, p):
    """Reshape matrix back to tensor form."""
    if mode == 0:
        return M.reshape(n, m, p)
    elif mode == 1:
        return M.reshape(m, n, p).transpose(1, 0, 2)
    elif mode == 2:
        return M.reshape(p, n, m).transpose(1, 2, 0)

def _als_decompose(T, rank, n, m, p, max_iter=500, tol=1e-10, seed=42):
    """Alternating Least Squares for tensor rank decomposition."""
    rng = np.random.RandomState(seed)
    
    # Initialize U (n*rank), V (m*rank), W (p*rank)
    U = rng.randn(n, rank) * 0.1
    V = rng.randn(m, rank) * 0.1
    W = rng.randn(p, rank) * 0.1
    
    for it in range(max_iter):
        # Update U: for each i, U[i,:] should satisfy sum_j,k T[i,j,k] = sum_r U[i,r] * (V*W)[:,r]
        # M_U = T.reshape(n, m*p)
        M_U = T.reshape(n, m * p)
        # For each r, we need U[i,r] such that sum_r U[i,r] * outer(V[:,r], W[:,r])_flattened = M_U[i,:]
        # This is: M_U = U @ (V*W)_flattened^T where (V*W)_flattened[r, j*p+k] = V[j,r]*W[k,r]
        VW = np.zeros((m * p, rank))
        for r in range(rank):
            VW[:, r] = (V[:, r] * W[:, r]).reshape(m * p)
        # M_U = U @ VW.T  =>  U = M_U @ pinv(VW.T)
        if rank > 0:
            U_new = M_U @ np.linalg.pinv(VW.T)
            U = U_new
        
        # Update V
        M_V = T.transpose(1, 0, 2).reshape(m, n * p)
        UW = np.zeros((n * p, rank))
        for r in range(rank):
            UW[:, r] = (U[:, r] * W[:, r]).reshape(n * p)
        V_new = M_V @ np.linalg.pinv(UW.T)
        V = V_new
        
        # Update W
        M_W = T.transpose(2, 0, 1).reshape(p, n * m)
        UV = np.zeros((n * m, rank))
        for r in range(rank):
            UV[:, r] = (U[:, r] * V[:, r]).reshape(n * m)
        W_new = M_W @ np.linalg.pinv(UV.T)
        W = W_new
        
        # Check convergence
        if it % 50 == 0 or it == max_iter - 1:
            # Compute reconstruction error
            recon = np.zeros((n, m, p))
            for r in range(rank):
                recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(recon - T)
            if err < tol:
                break
    
    # Final error check
    recon = np.zeros((n, m, p))
    for r in range(rank):
        recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    err = np.linalg.norm(recon - T)
    
    return U, V, W, err

def _to_half_integer(M):
    """Convert to half-integers if close."""
    M2 = M * 2
    M_rounded = np.round(M2)
    if np.allclose(M2, M_rounded, atol=1e-8):
        return M_rounded / 2.0
    return M

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # But let's be more aggressive and try from 20
    for rank in range(20, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 2:
                break
            
            U, V, W, err = _als_decompose(T, rank, n, m, p, max_iter=300, tol=1e-10, seed=seed)
            
            if err < 1e-8:
                # Convert to half-integers if possible
                U_h = _to_half_integer(U)
                V_h = _to_half_integer(V)
                W_h = _to_half_integer(W)
                
                # Verify with half-integer values
                recon = np.zeros((n, m, p))
                for r in range(rank):
                    recon += np.einsum('i,j,k->ijk', U_h[:, r], V_h[:, r], W_h[:, r])
                if np.allclose(recon, T, atol=1e-8):
                    found = True
                    best_rank = rank
                    best_U = U_h
                    best_V = V_h
                    best_W = W_h
                    break
        
        if found:
            break
    
    # If no exact decomposition found, fall back to schoolbook
    if best_U is None:
        rank = best_rank
        U = np.zeros((n, rank))
        V = np.zeros((m, rank))
        W = np.zeros((p, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Convert to lists of lists with proper indexing
    # U: shape (n*m, rank) where U[i*m+j, r]
    U_out = np.zeros((n * m, best_rank))
    for i in range(n):
        for j in range(m):
            U_out[i * m + j, :] = best_U[i, :]
    
    # V: shape (m*p, rank) where V[j*p+k, r]
    V_out = np.zeros((m * p, best_rank))
    for j in range(m):
        for k in range(p):
            V_out[j * p + k, :] = best_V[j, :]
    
    # W: shape (p*n, rank) where W[k*n+i, r]
    W_out = np.zeros((p * n, best_rank))
    for k in range(p):
        for i in range(n):
            W_out[k * n + i, :] = best_W[k, :]
    
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
