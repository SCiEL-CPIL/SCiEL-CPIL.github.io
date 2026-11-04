"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np
from numpy.linalg import lstsq

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose T."""
        rank = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    def error_norm(U, V, W):
        """Compute the error."""
        rank = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_recon - T)
    
    best_result = {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    best_rank = rank_school
    best_err = 0.0
    
    def als_step(U, V, W, T, max_iter=50, tol=1e-10, verbose=False):
        """Alternating Least Squares for tensor decomposition."""
        T_reshaped = T.reshape(nm, -1)
        n_cols = mp * pn
        
        for it in range(max_iter):
            # Update U: T_reshaped ≈ U @ (V ⊗ W)^T
            # (V ⊗ W) has shape (r, mp*pn)
            VW = (V * W[None, :, :]).reshape(r, -1)
            # T_reshaped = U @ VW
            # Solve for U: U @ VW = T_reshaped => U = T_reshaped @ VW^T @ (VW @ VW^T)^-1
            # Use lstsq: U * VW = T_reshaped => solve (VW.T) X = T_reshaped.T for X
            try:
                U_new, _, _, _ = lstsq(VW.T, T_reshaped.T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Update V: T reshaped to (mp, nm*pn)
            T_V = T.reshape(mp, -1)
            UW = (U * W[:, None, :]).reshape(pn, -1)  # Wait, need U and W for each a,c
            # T[b, a, c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b: T[b, :, :] = sum_r V[b,r] (U[:,r] ⊗ W[:,r])
            # Reshape T to (mp, nm*pn): T_V[b, (a,c)] = sum_r V[b,r] * U[a,r] * W[c,r]
            # Let M_V[r, (a,c)] = U[a,r] * W[c,r]
            M_V = (U[:, :, None] * W[None, :, :]).reshape(r, -1)
            try:
                V_new, _, _, _ = lstsq(M_V.T, T_V.T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Update W: T reshaped to (pn, nm*mp)
            T_W = T.reshape(pn, -1)
            # T[c, a, b] = sum_r W[c,r] * U[a,r] * V[b,r]
            M_W = (U[:, None, :] * V[None, :, :]).reshape(r, -1)
            try:
                W_new, _, _, _ = lstsq(M_W.T, T_W.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Check convergence
            if it > 0:
                err = error_norm(U, V, W)
                if err < tol:
                    break
        
        return U, V, W
    
    # Try to find a lower-rank decomposition
    # Start from rank 30 and go up, or from 44 down?
    # Let's try a range. The theoretical lower bound for matmul(3,3,5) is not 45.
    # Strassen is for 2x2. For 3x3 it's known to be < 27 (schoolbook 27? No, 3*3*3=27).
    # Wait, for 3x3 * 3x5, schoolbook is 3*3*5 = 45.
    # Let's try ranks from 30 to 44.
    
    # To save time, we might just try a few specific ranks.
    # Let's try ranks [35, 38, 40, 42, 44]
    target_ranks = list(range(44, 34, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            U, V, W = als_step(U, V, W, T, max_iter=20, tol=1e-12)
            
            err = error_norm(U, V, W)
            
            if err < 1e-9:
                # Exact decomposition found
                best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                best_rank = target_rank
                break
            
            # Keep track of best error for this rank (optional, but we want exact)
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
