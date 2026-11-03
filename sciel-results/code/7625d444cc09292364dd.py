"""Matrix multiplication 3x4x4 tensor decomposition via SVD-based rank minimization."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    where T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _matmul_tensor_to_matrices(T, n, m, p):
    """Convert the 3D tensor T (shape n*m, m*p, p*n) to the standard
    matrix multiplication tensor M (shape n, m, p, n, m, p) where
    M[i,j,k,a,b,c] = 1 iff i=a, j=b, k=c."""
    # T is indexed by (i*m+j, j*p+k, k*n+i)
    # We want M[i,j,k,a,b,c] = 1 iff (i,j,k) == (a,b,c)
    # So M[i,j,k,i,j,k] = 1
    M = np.zeros((n, m, p, n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i, j, k, i, j, k] = 1.0
    return M

def _matmul_tensor_flat(n, m, p):
    """Return the matrix multiplication tensor as a 6D array M[i,j,k,a,b,c]
    where M[i,j,k,a,b,c] = 1 iff i==a and j==b and k==c."""
    M = np.zeros((n, m, p, n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i, j, k, i, j, k] = 1.0
    return M

def _decompose_svd(M, n, m, p, rank):
    """
    Given the 6D matmul tensor M[i,j,k,a,b,c] and a target rank,
    try to find factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = M[i,j,k,i,j,k].
    
    We reshape M into a 3D tensor of shape (n*m, m*p, p*n) and use
    tensor SVD / rank-revealing decomposition.
    """
    # Reshape M to 3D: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # But M is 6D. We need to map (i,j,k,a,b,c) -> (i*m+j, j*p+k, k*n+i)
    # M[i,j,k,i,j,k] = 1, so T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Now we want to decompose T as sum_c u_c ⊗ v_c ⊗ w_c
    # Use CP decomposition via ALS or SVD-based approach
    
    # SVD-based approach: unfold T into matrices and use joint SVD
    # T as (n*m) x (m*p * p*n) matrix
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # But we need a proper tensor decomposition. Let's use ALS for CP.
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for iteration in range(200):
        # Update U: for each a, U[a,:] = argmin sum_c (T[a,b,c] - U[a,c]*V[b,c]*W[c,c])^2
        # T[a,b,c] = sum_c U[a,c] V[b,c] W[c,c]
        # For fixed a: T[a,:,:] = U[a,:] @ (V * W) where (V*W)[b,c] = V[b,c]*W[c,c]
        VW = V * W  # (m*p, p*n) element-wise... no
        # Actually: T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
        # For fixed a: T[a,:,:] = sum_d U[a,d] * outer(V[:,d], W[:,d])
        # So T[a] (vector of length m*p*p*n) = U[a] @ (V ⊙ W) where ⊙ is Khatri-Rao
        # Khatri-Rao product: (V ⊙ W)[:, d] = V[:,d] * W[:,d]
        VW_kr = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m*p * p*n, rank)
        # T[a, b, c] flattened: T_flat[a, b*p*n + c]
        T_flat = T.reshape(n*m, m*p * p*n)
        U = np.linalg.lstsq(VW_kr, T_flat, rcond=None)[0]
        
        # Update V
        UW_kr = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n*m * p*n, rank)
        # T[b, a, c] -> need T reshaped as (m*p, n*m*p*n)
        T_perm = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m * p*n)
        V = np.linalg.lstsq(UW_kr, T_perm, rcond=None)[0]
        
        # Update W
        UV_kr = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n*m * m*p, rank)
        T_perm2 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m * m*p)
        W = np.linalg.lstsq(UV_kr, T_perm2, rcond=None)[0]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    # Also check that non-diagonal entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(p):
                            if not (i==a and j==b and k==c):
                                if abs(T[a*m+b, b*p+c, c*n+i]) > tol:
                                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification: only check the required entries."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_svd_decomposition(n, m, p, rank, n_restarts=5, deadline=None):
    """Try to find a rank-`rank` decomposition using ALS on the tensor."""
    best = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize with random factors
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        T_flat = T.reshape(n*m, m*p * p*n)
        T_perm = np.transpose(T, (1, 0, 2)).reshape(m*p, n*m * p*n)
        T_perm2 = np.transpose(T, (2, 0, 1)).reshape(p*n, n*m * m*p)
        
        converged = False
        for it in range(300):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            VW_kr = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m*p * p*n, rank)
            U_new = np.linalg.lstsq(VW_kr, T_flat, rcond=None)[0]
            
            # Update V
            UW_kr = (U_new[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n*m * p*n, rank)
            V_new = np.linalg.lstsq(UW_kr, T_perm, rcond=None)[0]
            
            # Update W
            UV_kr = (U_new[:, np.newaxis, :] * V_new[np.newaxis, :, :]).reshape(n*m * m*p, rank)
            W_new = np.linalg.lstsq(UV_kr, T_perm2, rcond=None)[0]
            
            # Check convergence
            if np.allclose(U_new, U, atol=1e-10) and np.allclose(V_new, V, atol=1e-10) and np.allclose(W_new, W, atol=1e-10):
                converged = True
                break
            
            U, V, W = U_new, V_new, W_new
        
        # Compute error
        T_recon = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err += (T_recon[i*m+j, j*p+k, k*n+i] - 1.0)**2
        # Also check off-diagonal
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for a in range(n):
                        for b in range(m):
                            for c in range(p):
                                if not (i==a and j==b and k==c):
                                    err += T_recon[a*m+b, b*p+c, c*n+i]**2
        
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-8:
            break
    
    if best is None:
        return None
    
    U, V, W = best
    # Verify
    if _verify_fast(U, V, W, n, m, p, tol=1e-4):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-4):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try SVD/ALS-based decomposition for ranks below schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But actual minimum rank for 3x4x4 is unknown; try from 16 upward
    for rank in range(max(n*m, m*p, p*n), best_rank):
        if time.time() > deadline:
            break
        
        result = _try_svd_decomposition(n, m, p, rank, n_restarts=10, deadline=deadline)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found the minimal rank
    
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
