"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 60
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution as fallback
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Use ALS (Alternating Least Squares) to find lower rank decompositions
    # The tensor T can be decomposed as sum_r u_r v_r^T w_r^T where
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    def check_decomposition(U, V, W, T, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        R = U.shape[1]
        # Reconstruct T from the decomposition
        T_recon = np.zeros(T.shape, dtype=float)
        for r in range(R):
            T_recon += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        # Check max absolute difference
        max_diff = np.max(np.abs(T_recon - T))
        return max_diff < tol
    
    def als_optimize(T, R, max_iters=200, tol=1e-10, seed=42):
        """
        Alternating Least Squares for tensor decomposition.
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        We minimize ||T - sum_r u_r v_r^T w_r^T||_F^2
        """
        np.random.seed(seed)
        nm, mp, pn = T.shape
        
        # Random initialization
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # Normalize to avoid issues
        for r in range(R):
            norm_U = np.linalg.norm(U[:, r])
            norm_V = np.linalg.norm(V[:, r])
            norm_W = np.linalg.norm(W[:, r])
            if norm_U > 0 and norm_V > 0 and norm_W > 0:
                U[:, r] /= norm_U
                V[:, r] /= norm_V
                W[:, r] /= norm_W
        
        prev_cost = np.inf
        for iteration in range(max_iters):
            # Update U: for each a, we have T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V, W. For each a, the contribution is U[a,:] dot (V[:,r] * W[:,r]) over b,c
            # Reshape T as (nm, mp*pn) where T[a, b*pn+c] = sum_r U[a,r] * (V[b,r]*W[c,r])
            T_mat = T.reshape(nm, mp * pn)
            
            for a in range(nm):
                # Target for row a: T_mat[a, :]
                # Model: U[a, r] * (V[:, r] * W[:, r]) flattened
                # Build matrix M where M[:, r] = (V[:, r] * W[:, r]).reshape(-1)
                M = np.zeros((mp * pn, R))
                for r in range(R):
                    M[:, r] = (V[:, r] * W[:, r]).reshape(-1)
                
                # Solve U[a, :] to best match T_mat[a, :]
                # T_mat[a, :] ≈ U[a, :] @ M
                # This is: M^T U[a, :]^T = M^T T_mat[a, :]^T
                # U[a, :] = T_mat[a, :] @ M @ (M^T M)^{-1}
                try:
                    MtM = M.T @ M
                    Mty = M.T @ T_mat[a]
                    # Add small regularization for stability
                    MtM[range(R), range(R)] += 1e-12
                    U[a, :] = np.linalg.solve(MtM, Mty)
                except np.linalg.LinAlgError:
                    pass
            
            # Update V: for each b, T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_mat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            for b in range(mp):
                M = np.zeros((nm * pn, R))
                for r in range(R):
                    M[:, r] = (U[:, r] * W[:, r]).reshape(-1)
                
                try:
                    MtM = M.T @ M
                    Mty = M.T @ T_mat_v[b]
                    MtM[range(R), range(R)] += 1e-12
                    V[b, :] = np.linalg.solve(MtM, Mty)
                except np.linalg.LinAlgError:
                    pass
            
            # Update W: for each c, T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_mat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            for c in range(pn):
                M = np.zeros((nm * mp, R))
                for r in range(R):
                    M[:, r] = (U[:, r] * V[:, r]).reshape(-1)
                
                try:
                    MtM = M.T @ M
                    Mty = M.T @ T_mat_w[c]
                    MtM[range(R), range(R)] += 1e-12
                    W[c, :] = np.linalg.solve(MtM, Mty)
                except np.linalg.LinAlgError:
                    pass
            
            # Compute cost
            cost = 0.0
            for r in range(R):
                cost += np.sum((T - np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r]))**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        return U, V, W, cost
    
    # Try ranks from lower to higher, attempting to find exact decompositions
    # Based on known results, matrix multiplication of 2x3 by 3x10 might have lower rank
    # Let's try ranks from 30 up to 59
    for R in range(30, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found_exact = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            seed = 42 + R * 100 + trial
            U, V, W, cost = als_optimize(T, R, max_iters=100, tol=1e-12, seed=seed)
            
            if check_decomposition(U, V, W, T, tol=1e-8):
                best_rank = R
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found_exact = True
                break
        
        if found_exact:
            # Found an exact decomposition, continue to see if we can do better
            continue
        else:
            # If we've tried several ranks without success, break early
            if R > 45:
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

