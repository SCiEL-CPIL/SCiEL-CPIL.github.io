"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if output index (k*n+i) and input indices (i*m+j, j*p+k)
    # The tensor is: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # and for all other index combinations where not (i,j,k) pattern, sum should be 0.
    # Actually the tensor is defined as: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a, b, c] = 0.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We need to check this equals the true tensor.
    
    # True tensor: T[a, b, c] = 1 iff exists i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # a ranges 0..n*m-1, b ranges 0..m*p-1, c ranges 0..p*n-1
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Reconstruct
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness
    if not np.allclose(T_recon, T_true, atol=1e-10):
        return False
    return True

def _try_rank(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition using structured approach."""
    if r < max(n, m, p):
        return None  # Rank must be at least max(n,m,p)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Strategy 1: Try structured decompositions based on tensor structure
    # The matrix multiplication tensor has a specific structure we can exploit
    
    # Try using the "rank-1" building blocks more cleverly
    # For each combination of (i,j,k), we need T[i*m+j, j*p+k, k*n+i] = 1
    # We can try to share factors across different (i,j,k) tuples
    
    # Approach: Use a grid-based decomposition
    # Let's try to find a decomposition where each rank-1 term covers a "block"
    
    # Try small structured patterns
    for pattern_seed in range(10):
        rng = np.random.RandomState(seed + pattern_seed * 1000)
        
        # Initialize with a structured pattern
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Try to assign each (i,j,k) to one or more rank terms
        # Use a greedy assignment
        assignments = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Assign to a random rank term
                    assignments[(i,j,k)] = rng.randint(0, r)
        
        # Initialize based on assignments
        for (i,j,k), rank_idx in assignments.items():
            U[i*m + j, rank_idx] += 1.0
            V[j*p + k, rank_idx] += 1.0
            W[k*n + i, rank_idx] += 1.0
        
        # Now refine using ALS
        for it in range(max_iters):
            # Update U using least squares
            # For each row a of U, solve: U[a,:] * (V^T * W) = T_true[a,:,:]
            # This is equivalent to: U[a,:] = T_true[a,:,:] @ (V * W_outer) @ pinv((V * W_outer)^T @ (V * W_outer))
            
            # Compute the Gram matrix for V and W
            # G = sum_{b,c} V[b,:] * W[c,:] (outer product summed over b,c)
            # Actually: for fixed a, we want to minimize ||T_true[a,:,:] - U[a,:] * (V ⊙ W)||_F^2
            # where (V ⊙ W)[b,c] = V[b,:] * W[c,:]
            
            # This is a linear system: U[a,:] @ M = T_true[a,:,:].reshape(-1)
            # where M is the matrix with columns being V[b,:] * W[c,:] for all b,c
            
            # Build M: shape (m*p * p*n, r)
            # M[b*p*n + c, r] = V[b, r] * W[c, r]
            M = np.zeros((m*p * p*n, r))
            for b in range(m*p):
                for c in range(p*n):
                    M[b * p*n + c, :] = V[b, :] * W[c, :]
            
            # Solve least squares for each row of U
            # Use numpy's lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(M, T_true.reshape(n*m, -1).T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Similarly update V
            M_v = np.zeros((n*m * p*n, r))
            for a in range(n*m):
                for c in range(p*n):
                    M_v[a * p*n + c, :] = U[a, :] * W[c, :]
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v, T_true.reshape(-1, m*p).T.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Similarly update W
            M_w = np.zeros((n*m * m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    M_w[a * m*p + b, :] = U[a, :] * V[b, :]
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w, T_true.reshape(-1, p*n).T.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence every 50 iterations
            if it % 50 == 0:
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-8:
                    # Try to round
                    for scale in [1, 2, 4, 8]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    break
        
        # Final check
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110 second budget to be safe
    
    # Try to find a lower-rank decomposition
    # For 3x4x8: schoolbook is 96. Let's try ranks below that.
    
    # Try candidate ranks in increasing order, starting from reasonable lower bounds
    # The rank must be at least max(n, m, p) = 8, but likely much higher
    # Try structured ranks first
    
    candidates = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
    
    for r in candidates:
        if time.time() > deadline - 20:
            break
        if r >= n * m * p:
            break  # Schoolbook already handles this
        
        result = _try_rank(r, n, m, p, max_iters=1000, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
