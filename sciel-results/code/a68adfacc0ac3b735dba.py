import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook fallback (rank = n*m*p = 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        P = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(P, T, atol=tol)

    def project_to_half_integers(U, V, W):
        """Project matrices to nearest half-integers."""
        U_proj = np.round(2 * U) / 2.0
        V_proj = np.round(2 * V) / 2.0
        W_proj = np.round(2 * W) / 2.0
        return U_proj, V_proj, W_proj

    def als(U, V, W, max_iter=100, tol=1e-10):
        """Alternating Least Squares to refine U, V, W."""
        for it in range(max_iter):
            # Update U
            # T[i, :, :] = sum_r U[i, r] * V[:, r] * W[:, r]
            # For fixed i, T[i] (mp x pn) = sum_r U[i, r] * outer(V[:, r], W[:, r])
            # Let A_r = outer(V[:, r], W[:, r]).flatten() (mp*pn)
            # T_flat = T.reshape(nm, mp*pn)
            # We want to solve T_flat[i] = U[i] @ A_matrix
            # A_matrix is rank x (mp*pn) where A_matrix[r, :] = A_r
            
            VW = np.einsum('jr,kr->jrkr', V, W).reshape(V.shape[1], -1) # rank x (mp*pn)
            T_mat = T.reshape(nm, -1)
            
            # Solve U @ VW.T = T_mat
            # U = T_mat @ (VW.T)^+ 
            try:
                U_new = np.linalg.lstsq(VW.T, T_mat, rcond=tol)[0].T
            except:
                U_new = U
            
            # Update V
            # T[:, j, :] = sum_r U[:, r] * V[j, r] * W[:, r]
            # For fixed j, T[:, j, :] (nm x pn) = sum_r V[j, r] * outer(U[:, r], W[:, r])
            UW = np.einsum('ir,kr->irkr', U_new, W).reshape(U_new.shape[1], -1) # rank x (nm*pn)
            T_mat2 = T.transpose(0, 2, 1).reshape(mp, -1) # (mp, nm*pn)
            
            try:
                V_new = np.linalg.lstsq(UW.T, T_mat2, rcond=tol)[0].T
            except:
                V_new = V
                
            # Update W
            # T[:, :, k] = sum_r U[:, r] * V[:, r] * W[k, r]
            # For fixed k, T[:, :, k] (nm x mp) = sum_r W[k, r] * outer(U[:, r], V[:, r])
            UV = np.einsum('ir,jr->irjr', U_new, V_new).reshape(U_new.shape[1], -1) # rank x (nm*mp)
            T_mat3 = T.transpose(1, 2, 0).reshape(pn, -1) # (pn, nm*mp)
            
            try:
                W_new = np.linalg.lstsq(UV.T, T_mat3, rcond=tol)[0].T
            except:
                W_new = W
                
            # Check convergence
            P = np.einsum('ur,vr,wr->uvw', U_new, V_new, W_new)
            err = np.linalg.norm(P - T)
            if err < tol:
                break
                
            U, V, W = U_new, V_new, W_new
            
        return U, V, W

    def try_find_decomposition(rank, max_restarts=50, max_iter_als=50):
        """Try to find a rank decomposition using random restarts and ALS."""
        start_time = time.time()
        deadline = time.time() + 100 # 100s budget
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Run ALS
            U_f, V_f, W_f = als(U, V, W, max_iter=max_iter_als)
            
            # Project to half-integers
            U_p, V_p, W_p = project_to_half_integers(U_f, V_f, W_f)
            
            # Verify
            if verify(U_p, V_p, W_p, tol=1e-9):
                return U_p, V_p, W_p
                
        return None

    best_result = None
    best_rank = rank_sb
    
    # Try ranks below schoolbook
    # Known rank is 28. Try 28, 30, 32, 34, 36, 38, 40
    target_ranks = [28, 30, 32, 34, 36, 38, 40]
    
    for target_rank in target_ranks:
        if time.time() > 140: # Safety margin
            break
            
        result = try_find_decomposition(target_rank, max_restarts=20, max_iter_als=30)
        
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = target_rank
            best_result = (U_best, V_best, W_best)
            break # Found a valid one, return it (lowest rank first)

    if best_result is not None:
        U_best, V_best, W_best = best_result
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
