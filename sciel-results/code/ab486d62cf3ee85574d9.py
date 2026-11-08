"""Matrix multiplication 2x4x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find low-rank decomposition
    # Start with schoolbook rank 32
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook
    
    # Try rank reduction via ALS (Alternating Least Squares)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For fixed V,W: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
    # So T[a].flatten() = M_VW @ U[a,:] where M_VW[b,c,r] = V[b,r]*W[c,r]
    
    def matmul_tensor_residual(U, V, W):
        """Compute ||T - sum_r U[:,r] outer(V[:,r], W[:,r])||_F"""
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_approx += U[:, r][:, np.newaxis, np.newaxis] * V[:, r][np.newaxis, :, np.newaxis] * W[:, r][np.newaxis, np.newaxis, :]
        return np.linalg.norm(T - T_approx)
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W exactly decompose T"""
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_approx += U[:, r][:, np.newaxis, np.newaxis] * V[:, r][np.newaxis, :, np.newaxis] * W[:, r][np.newaxis, np.newaxis, :]
        return np.allclose(T, T_approx, atol=tol)
    
    # Try various ranks from 28 down
    for target_rank in range(28, rank_schoolbook):
        if time.time() - _start_time > TIME_LIMIT - 30:
            break
        
        found = False
        for trial in range(20):
            if time.time() - _start_time > TIME_LIMIT - 30:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            
            U_r = np.random.randn(n*m, target_rank) * 0.1
            V_r = np.random.randn(m*p, target_rank) * 0.1
            W_r = np.random.randn(p*n, target_rank) * 0.1
            
            prev_resid = matmul_tensor_residual(U_r, V_r, W_r)
            
            for it in range(200):
                if time.time() - _start_time > TIME_LIMIT - 30:
                    break
                
                # Update U: for each a, solve linear system
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # M_U[a, (b,c), r] = V[b,r] * W[c,r]
                # T[a].flatten() = M_U[a] @ U[a]
                
                # Build M_U: shape (n*m, m*p*p*n, target_rank)
                M_U = np.zeros((n*m, m*p*p*n, target_rank))
                for r in range(target_rank):
                    # outer(V[:,r], W[:,r]) has shape (m*p, p*n)
                    ov = np.outer(V_r[:, r], W_r[:, r])  # (m*p, p*n)
                    M_U[:, :, r] = ov.T.reshape(m*p*p*n, 1).T  # This is wrong
                    # Actually we need M_U[a, (b,c), r] = V[b,r]*W[c,r] for all a
                    # So M_U[:, (b,c), r] = V[b,r]*W[c,r] (same for all a)
                
                # Let me rebuild this properly
                M_U = np.zeros((m*p*p*n, target_rank))
                for r in range(target_rank):
                    ov = np.outer(V_r[:, r], W_r[:, r])  # (m*p, p*n)
                    M_U[:, r] = ov.reshape(m*p*p*n)
                
                # For each a: T[a].flatten() = M_U @ U[a]
                # So U[a] = (M_U^T M_U)^{-1} M_U^T T[a].flatten()
                try:
                    M_U_T_M_U = M_U.T @ M_U
                    # Add small regularization for numerical stability
                    M_U_T_M_U += 1e-10 * np.eye(target_rank)
                    U_new = np.zeros((n*m, target_rank))
                    for a in range(n*m):
                        U_new[a] = np.linalg.solve(M_U_T_M_U, M_U.T @ T[a].reshape(m*p*p*n))
                except np.linalg.LinAlgError:
                    break
                
                # Update V: for each b, T[:,b,:].reshape(n*m*p*n) = M_V @ V[b]
                # M_V[(a,c), r] = U[a,r] * W[c,r]
                M_V = np.zeros((n*m*p*n, target_rank))
                for r in range(target_rank):
                    # For each (a,c): M_V[(a,c), r] = U[a,r] * W[c,r]
                    M_V[:, r] = np.outer(U_r[:, r], W_r[:, r]).reshape(n*m*p*n)
                
                try:
                    M_V_T_M_V = M_V.T @ M_V
                    M_V_T_M_V += 1e-10 * np.eye(target_rank)
                    V_new = np.zeros((m*p, target_rank))
                    for b in range(m*p):
                        # T[:,b,:] has shape (n*m, p*n), flatten to (n*m*p*n)
                        T_b = T[:, b, :].reshape(n*m*p*n)
                        V_new[b] = np.linalg.solve(M_V_T_M_V, M_V.T @ T_b)
                except np.linalg.LinAlgError:
                    break
                
                # Update W: for each c, T[:,:,c].reshape(n*m*m*p) = M_W @ W[c]
                # M_W[(a,b), r] = U[a,r] * V[b,r]
                M_W = np.zeros((n*m*m*p, target_rank))
                for r in range(target_rank):
                    M_W[:, r] = np.outer(U_r[:, r], V_r[:, r]).reshape(n*m*m*p)
                
                try:
                    M_W_T_M_W = M_W.T @ M_W
                    M_W_T_M_W += 1e-10 * np.eye(target_rank)
                    W_new = np.zeros((p*n, target_rank))
                    for c in range(p*n):
                        T_c = T[:, :, c].reshape(n*m*m*p)
                        W_new[c] = np.linalg.solve(M_W_T_M_W, M_W.T @ T_c)
                except np.linalg.LinAlgError:
                    break
                
                U_r, V_r, W_r = U_new, V_new, W_new
                curr_resid = matmul_tensor_residual(U_r, V_r, W_r)
                
                if curr_resid < 1e-8:
                    found = True
                    break
                if abs(prev_resid - curr_resid) < 1e-12:
                    break
                prev_resid = curr_resid
            
            if found:
                if verify_decomposition(U_r, V_r, W_r):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = target_rank
                    break
                else:
                    # Try to clean up
                    found = False
                    break  # Move to next trial
        
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
