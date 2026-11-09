"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def als_solve(target_rank, U, V, W, max_iter=200, tol=1e-8):
        """Alternating least squares to refine a decomposition."""
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U: fix V and W
            # For each (i,j): sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])_k = 1
            # This is: U[i,j,:] @ A[i,j,k] = 1 for all k
            # Where A[i,j,k] = sum_c V[j,k,c] * W[k,i,c]
            # Actually, let's think more carefully.
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # For fixed i,j: sum_c U[i,j,c] * B[i,j,k,c] = 1 where B[i,j,k,c] = V[j,k,c]*W[k,i,c]
            # This is a linear system: U[i,j,:] @ B[i,j,:,:] = ones(p)
            # So U[i,j,:] = ones(p) @ B[i,j,:,:]^{-1} (if invertible)
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # B[i,j,k,c] = V[j,k,c] * W[k,i,c]
                    B = V[j, :, :] * W[:, i, :]  # shape (p, r)
                    # Need: U[i,j,:] @ B.T = ones(p)
                    # So U[i,j,:] = ones(p) @ pinv(B.T)
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            # Solve for V: fix U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # B[i,j,k,c] = U[i,j,c] * W[k,i,c]
                    B = U.reshape(n, m, target_rank)[:, j, :] * W[k, :, :]  # shape (n, r)
                    # Need: V[j,k,:] @ B.T = ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            # Solve for W: fix U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # B[i,j,k,c] = U[i,j,c] * V[j,k,c]
                    B = U.reshape(n, m, target_rank)[i, :, :] * V[:, k, :]  # shape (m, r)
                    # Need: W[k,i,:] @ B.T = ones(m)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(B.T, np.ones(m), rcond=None)[0]
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,target_rank), V.reshape(m,p,target_rank), W.reshape(p,n,target_rank))
            if np.max(np.abs(T - 1.0)) < tol:
                return U, V, W
        
        return U, V, W
    
    def try_rank(target_rank, num_trials=5):
        """Try to find a decomposition of given rank."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # Run ALS
            U, V, W = als_solve(target_rank, U, V, W)
            
            # Check
            if check_decomposition(U, V, W, tol=1e-5):
                return U, V, W
        
        return None
    
    # Try various ranks from low to high
    for target_rank in [35, 40, 45, 50, 55, 60, 65, 70]:
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, num_trials=3)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
