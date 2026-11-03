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

    def verify_exact(U, V, W):
        """Check exact decomposition using fractions."""
        P = np.einsum('ur,vr,wr->uvw', U, V, W)
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    if abs(P[i,j,k] - T[i,j,k]) > 1e-10:
                        return False
        return True

    def project_to_half_integers(U, V, W):
        """Project matrices to nearest half-integers."""
        U_proj = np.round(2 * U) / 2.0
        V_proj = np.round(2 * V) / 2.0
        W_proj = np.round(2 * W) / 2.0
        return U_proj, V_proj, W_proj

    def alternating_minimization(U, V, W, max_iter=50, tol=1e-8):
        """Perform alternating minimization to reduce rank."""
        for it in range(max_iter):
            # Update U: fix V, W, solve for U
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For each r, we need U[:,r] such that sum_r U[:,r] * (V[:,r] * W[:,r]) approximates T
            # Reshape T to (nm, mp*pn)
            T_mat = T.reshape(nm, mp * pn)
            
            # Build the coefficient matrix for U
            # For each r, the contribution is U[:,r] * (V[:,r] * W[:,r]).T
            # So we want to solve: U * (V*W).T = T_mat where (V*W)[j,k] = sum_r V[j,r]*W[k,r]
            VW = np.einsum('jr,kr->jk', V, W)  # mp x pn
            VW_flat = VW.flatten()  # mp*pn
            
            # We want U such that U @ VW_flat.T approximates T_mat
            # This is a linear least squares problem
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW_flat.reshape(1, -1).T, T_mat, rcond=tol)
                U_new = U_new.T  # back to nm x rank
            except:
                U_new = U
            
            # Update V: fix U, W, solve for V
            UW = np.einsum('ir,kr->ik', U, W)  # nm x pn
            UW_flat = UW.flatten()
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_flat.reshape(1, -1).T, T_mat.T, rcond=tol)
                V_new = V_new.T  # back to mp x rank
            except:
                V_new = V
            
            # Update W: fix U, V, solve for W
            UV = np.einsum('ir,jr->ij', U, V)  # nm x mp
            UV_flat = UV.flatten()
            T_mat2 = T.transpose(0, 2, 1).reshape(pn, nm * mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_flat.reshape(1, -1).T, T_mat2, rcond=tol)
                W_new = W_new.T  # back to pn x rank
            except:
                W_new = W
            
            # Check convergence
            P = np.einsum('ur,vr,wr->uvw', U_new, V_new, W_new)
            err = np.linalg.norm(P - T)
            if err < tol:
                break
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W

    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_iter=30):
        """Try to reduce rank to target_rank using SVD-based truncation."""
        rank = U_init.shape[1]
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        
        for _ in range(max_iter):
            if U.shape[1] <= target_rank:
                break
            
            # Compute the current decomposition tensor
            P = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            # Use SVD to find the best rank-r approximation
            # Flatten T to (nm, mp*pn)
            T_mat = T.reshape(nm, mp * pn)
            
            # Truncated SVD
            try:
                U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
                # Take top target_rank components
                U_trunc = U_svd[:, :target_rank]
                S_trunc = S[:target_rank]
                Vt_trunc = Vt[:target_rank, :]
                
                # Reconstruct
                T_approx = np.outer(U_trunc, Vt_trunc * S_trunc)
                T_approx = T_approx.reshape(nm, mp, pn)
                
                # Now we need to factor this back into U, V, W
                # This is still a rank target_rank tensor
                # We can use the SVD factors directly
                # U_new = U_trunc, but we need V and W such that sum_r U_r V_r W_r = T_approx
                
                # Actually, let's just keep the current U, V, W and do alternating minimization
                # with the constraint that we want to reduce rank
                pass
            except:
                pass
            
            # Perform alternating minimization
            U, V, W = alternating_minimization(U, V, W, max_iter=10)
            
            # Try to drop the smallest singular value direction
            # Compute the effective rank by looking at the singular values of the combined tensor
            P = np.einsum('ur,vr,wr->uvw', U, V, W)
            T_mat = P.reshape(nm, mp * pn)
            try:
                _, S, _ = np.linalg.svd(T_mat, full_matrices=False)
                if len(S) > target_rank:
                    # Check if we can drop the last component
                    if S[-1] < 1e-6:
                        U = U[:, :-1]
                        V = V[:, :-1]
                        W = W[:, :-1]
            except:
                pass
        
        return U, V, W

    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit
    
    best_result = None
    best_rank = rank_sb
    
    # Start from schoolbook and try to reduce rank
    U_curr = U_sb.copy()
    V_curr = V_sb.copy()
    W_curr = W_sb.copy()
    
    # Try target ranks in decreasing order
    target_ranks = [35, 30, 28, 25, 21, 18, 15]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try to achieve this rank
        U_try, V_try, W_try = try_rank_reduction(U_curr, V_curr, W_curr, target_rank)
        
        if U_try.shape[1] <= target_rank:
            # Project to half-integers
            U_proj, V_proj, W_proj = project_to_half_integers(U_try, V_try, W_try)
            
            # Verify exact decomposition
            if verify(U_proj, V_proj, W_proj, tol=1e-8):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = (U_proj, V_proj, W_proj)
                    U_curr, V_curr, W_curr = U_proj, V_proj, W_proj
                    break  # Found a good rank, stop
    
    if best_result is not None:
        U_best, V_best, W_best = best_result
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
