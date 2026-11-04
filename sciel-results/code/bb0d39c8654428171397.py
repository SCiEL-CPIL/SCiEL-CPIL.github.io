import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 115
    
    # Build the 3x3x3 matrix multiplication tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Naive rank-27 decomposition
    U27 = np.zeros((n*m, 27), dtype=float)
    V27 = np.zeros((m*p, 27), dtype=float)
    W27 = np.zeros((p*n, 27), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i*m+j, col] = 1.0
                V27[j*p+k, col] = 1.0
                W27[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if U,V,W exactly decompose T."""
        R = U.shape[1]
        # Compute sum over t of U[:,t] * V[:,t] * W[:,t]
        # Using einsum for efficiency
        try:
            result = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            return np.max(np.abs(result - T)) < tol
        except Exception:
            return False
    
    def snap_to_halfint(U, V, W, tol=0.01):
        """Snap values to nearest half-integer if close."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if (np.max(np.abs(U_s - U)) < tol and
            np.max(np.abs(V_s - V)) < tol and
            np.max(np.abs(W_s - W)) < tol):
            return U_s, V_s, W_s
        return None
    
    def snap_to_int(U, V, W, tol=0.01):
        """Snap values to nearest integer if close."""
        U_s = np.round(U)
        V_s = np.round(V)
        W_s = np.round(W)
        if (np.max(np.abs(U_s - U)) < tol and
            np.max(np.abs(V_s - V)) < tol and
            np.max(np.abs(W_s - W)) < tol):
            return U_s, V_s, W_s
        return None
    
    best_rank = 27
    best_U, best_V, best_W = U27.copy(), V27.copy(), W27.copy()
    
    # Try to find lower rank decompositions using ALS (Alternating Least Squares)
    # with random restarts
    
    np.random.seed(42)
    
    for target_rank in [26, 25, 24, 23]:
        if time.time() > deadline - 10:
            break
        
        max_restarts = 50
        max_iters = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each a, solve least squares
                # T[a,:,:] = sum_t U[a,t] * (V[:,t] outer W[:,t])
                # Reshape: for fixed a, T[a] (b,c) = sum_t U[a,t] * V[b,t] * W[c,t]
                # = sum_t U[a,t] * (V * W^T)[b,c] where (V*W^T) is rank-1 per t
                # So T[a] = sum_t U[a,t] * (V[:,t] @ W[:,t].T)
                # Let M_t = V[:,t] @ W[:,t].T, shape (m*p, p*n)
                # T[a] = sum_t U[a,t] * M_t
                # So we need to solve: for each a, T[a].flatten() = U[a,:] @ M.flatten()
                
                # Build M: (m*p * p*n, target_rank)
                # M[:,t] = (V[:,t] @ W[:,t].T).flatten()
                M = np.zeros((m*p * p*n, target_rank))
                for t in range(target_rank):
                    M[:, t] = (V[:, t] @ W[:, t].T).flatten()
                
                # Solve U: T.reshape(n*m, m*p*p*n) = U @ M
                T_flat = T.reshape(n*m, m*p * p*n)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_new.T
                except Exception:
                    pass
                
                # Update V similarly
                # T[a,b,c]: for fixed b, T[:,b,:] = sum_t V[b,t] * (U[:,t] outer W[:,t])
                # N_t = U[:,t] @ W[:,t].T, shape (n*m, p*n)
                N = np.zeros((n*m * p*n, target_rank))
                for t in range(target_rank):
                    N[:, t] = (U[:, t] @ W[:, t].T).flatten()
                
                T_flat_v = T.reshape(m*p, n*m * p*n)  # T[b, a*n*m + c]... wait
                # T[a,b,c], for fixed b: T[:,b,:] has shape (n*m, p*n)
                # Flatten: T[:,b,:].flatten() = V[b,:] @ N
                T_flat_v = np.zeros((m*p, n*m * p*n))
                for b in range(m*p):
                    T_flat_v[b, :] = T[:, b, :].flatten()
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(N, T_flat_v.T, rcond=None)
                    V = V_new.T
                except Exception:
                    pass
                
                # Update W
                # For fixed c: T[:,:,c] = sum_t W[c,t] * (U[:,t] outer V[:,t])
                P_mat = np.zeros((n*m * m*p, target_rank))
                for t in range(target_rank):
                    P_mat[:, t] = (U[:, t] @ V[:, t].T).flatten()
                
                T_flat_w = np.zeros((p*n, n*m * m*p))
                for c in range(p*n):
                    T_flat_w[c, :] = T[:, :, c].flatten()
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(P_mat, T_flat_w.T, rcond=None)
                    W = W_new.T
                except Exception:
                    pass
                
                # Check convergence
                if it % 10 == 0:
                    try:
                        result = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
                        err = np.max(np.abs(result - T))
                        if err < 1e-6:
                            # Found a solution!
                            # Try to snap
                            snapped = snap_to_halfint(U, V, W)
                            if snapped is not None:
                                U, V, W = snapped
                            else:
                                snapped = snap_to_int(U, V, W)
                                if snapped is not None:
                                    U, V, W = snapped
                            
                            if verify(U, V, W):
                                best_rank = target_rank
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                break
                    except Exception:
                        pass
                
                if best_rank == target_rank:
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to naive
        best_U, best_V, best_W = U27, V27, W27
        best_rank = 27
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
