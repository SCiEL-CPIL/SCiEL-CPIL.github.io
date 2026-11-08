"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W, tol=1e-6):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build target tensor T[i,j,k] = 1 if multiplying A[i,j]*B[j,k]
    # The decomposition requires: for all A, B: (AB)[i,k] = sum_c (U^T A)[c] (V^T B)[c] W[c]
    # Equivalently: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = U.shape[1]
    # Use einsum to compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is a contraction over c with indices i,j,k
    # Use np.einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # Target: T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] contributes to (AB)[i,k])
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _try_rank(rank, seed=42, max_iters=500, deadline=None):
    """Try to find a rank-rank decomposition using structured random search."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with random small values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        # Several rounds of alternating least squares
        for r in range(20):
            if deadline and time.time() > deadline:
                break
            
            # Update U: for each (i,j), minimize ||sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1||^2
            # This is linear in U[i*m+j,:]
            # Fix V, W. For each i,j: 
            #   error[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
            # Let A_ij[c,k] = V[j*p+k,c] * W[k*n+i,c]
            # Then we need: A_ij @ U[i*m+j,:] ≈ ones(p)
            # Least squares solution
            
            # Update V
            for j in range(m):
                # For fixed j, V[j*p: (j+1)*p, :] 
                # error[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
                # For fixed j: sum over i,k: (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                # Linear in V[j*p+k,:]: 
                # Let B_k[c,i] = U[i*m+j,c] * W[k*n+i,c]
                # Then sum_c V[j*p+k,c]*B_k[c,i] ≈ 1 for all i
                # Stack: for each k, we have a linear system in V[j*p+k,:]
                # V[j*p+k,:] @ B_k ≈ ones(n)
                # where B_k[c,i] = U[i*m+j,c] * W[k*n+i,c]
                # So: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                # This is: (V[j*p+k,:] * U[i*m+j,:] elementwise) dot W[k*n+i,:] = 1
                # Let's set up: for each k, matrix M_k[i,c] = U[i*m+j,c] * W[k*n+i,c]
                # Then: V[j*p+k,:] @ M_k = ones(n)
                # V[j*p+k,:] = pinv(M_k) @ ones(n)
                
                for k in range(p):
                    M_k = np.outer(U[:, j].reshape(n, -1)[:, :rank], W[k*n:(k+1)*n, :].T)
                    # M_k[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    # We want: sum_c V[j*p+k,c] * M_k[i,c] = 1 for all i
                    # So: M_k.T @ V[j*p+k,:] = ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M_k.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                # For fixed k: sum over i,j: (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                # Linear in W[k*n+i,:]: 
                # For each i: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                # M_i[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # W[k*n+i,:] @ M_i = ones(m)
                # M_i.T @ W[k*n+i,:] = ones(m)
                for i in range(n):
                    M_i = np.outer(U[i*m:(i+1)*m, :].T, V[:, k].reshape(m, -1)[:, :rank])
                    # M_i[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M_i.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For fixed i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # M_k[c] = V[j*p+k,c] * W[k*n+i,c]
                    # U[i*m+j,:] @ M = ones(p) where M[k,c] = V[j*p+k,c]*W[k*n+i,c]
                    M = np.outer(V[j*p:(j+1)*p, :].T, W[k*n+i, :])  # This is wrong
                    # M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    M = V[j*p:(j+1)*p, :] * W[k*n+i, :]  # shape (p, rank)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
        
        # Check convergence
        n2, m2, p2 = DIMENSIONS
        rank2 = U.shape[1]
        Ur = U.reshape(n2, m2, rank2)
        Vr = V.reshape(m2, p2, rank2)
        Wr = W.reshape(p2, n2, rank2)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-4:
        # Snap to half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        # Re-verify after snapping
        n2, m2, p2 = DIMENSIONS
        rank2 = best_U.shape[1]
        Ur = best_U.reshape(n2, m2, rank2)
        Vr = best_V.reshape(m2, p2, rank2)
        Wr = best_W.reshape(p2, n2, rank2)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        if np.max(np.abs(T - 1.0)) < 1e-6:
            return best_U, best_V, best_W
    return None

def construct():
    deadline = time.time() + 120
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try ranks from known lower bounds upward
    # For 3x4x6, known results suggest rank can be lower than 72
    # Try a range of ranks
    for rank in range(50, 72):
        if time.time() > deadline - 10:
            break
        result = _try_rank(rank, seed=rank, max_iters=100, deadline=deadline - 10)
        if result is not None:
            U, V, W = result
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
