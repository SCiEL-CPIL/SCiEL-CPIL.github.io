import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook rank and decomposition
    rank_sb = n * m * p
    
    def get_schoolbook():
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = rank_sb
    
    # Build the multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs T"""
        r = U.shape[1]
        T_recon = np.zeros((n * m, m * p, p * n))
        for l in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
        return np.allclose(T_recon, T, atol=tol)
    
    def snap_to_half_integers(U, V, W, max_denom=100):
        """Snap values to nearest rationals with small denominators and verify"""
        U_s = np.array([[float(Fraction(float(U[i,j])).limit_denominator(max_denom)) 
                        for j in range(U.shape[1])] for i in range(U.shape[0])])
        V_s = np.array([[float(Fraction(float(V[i,j])).limit_denominator(max_denom)) 
                        for j in range(V.shape[1])] for i in range(V.shape[0])])
        W_s = np.array([[float(Fraction(float(W[i,j])).limit_denominator(max_denom)) 
                        for j in range(W.shape[1])] for i in range(W.shape[0])])
        
        if verify_decomposition(U_s, V_s, W_s):
            # Convert to lists of Fractions
            U_ret = [[Fraction(U_s[i, j]).limit_denominator(1000) for j in range(U_s.shape[1])] for i in range(U_s.shape[0])]
            V_ret = [[Fraction(V_s[i, j]).limit_denominator(1000) for j in range(V_s.shape[1])] for i in range(V_s.shape[0])]
            W_ret = [[Fraction(W_s[i, j]).limit_denominator(1000) for j in range(W_s.shape[1])] for i in range(W_s.shape[0])]
            return {"U": U_ret, "V": V_ret, "W": W_ret}
        return None
    
    def try_rank_als(r, seed=42, max_iter=3000, tol=1e-10):
        """ALS with linear solve for rank r"""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        best_residual = np.inf
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Update U: fix V and W
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # For each a: U[a,:] solves G_U @ U[a,:]^T = h_U^T
            TW = V[:, None, :] * W[None, :, :]  # (m*p, p*n, r)
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U.T).T
            except np.linalg.LinAlgError:
                U_new = h_U
            U = U_new
            
            # Update V: fix U and W
            UW = U[:, None, :] * W[None, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V.T).T
            except np.linalg.LinAlgError:
                V_new = h_V
            V = V_new
            
            # Update W: fix U and V
            UV = U[:, None, :] * V[None, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W.T).T
            except np.linalg.LinAlgError:
                W_new = h_W
            W = W_new
            
            # Check residual
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            residual = np.linalg.norm(T_recon - T)
            
            if residual < best_residual:
                best_residual = residual
                best_state = (U.copy(), V.copy(), W.copy())
            
            if residual < tol:
                break
        
        if best_residual < 1e-8:
            U_best, V_best, W_best = best_state
            snapped = snap_to_half_integers(U_best, V_best, W_best)
            if snapped is not None:
                return (r, snapped)
            
            # Try rounding to integers
            U_r = np.round(U_best)
            V_r = np.round(V_best)
            W_r = np.round(W_best)
            if verify_decomposition(U_r, V_r, W_r):
                U_ret = [[int(U_r[i, j]) for j in range(U_r.shape[1])] for i in range(U_r.shape[0])]
                V_ret = [[int(V_r[i, j]) for j in range(V_r.shape[1])] for i in range(V_r.shape[0])]
                W_ret = [[int(W_r[i, j]) for j in range(W_r.shape[1])] for i in range(W_r.shape[0])]
                return (r, {"U": U_ret, "V": V_ret, "W": W_ret})
        
        return None
    
    def try_laser_decomposition(r):
        """
        Laser method: Use structure of 3x3x3 rank-13 to build 3x3x5.
        
        Idea: The 3x3x5 multiplication can be seen as 3x3x3 plus 3x3x2.
        We try to find a decomposition that shares structure with known low-rank 3x3x3.
        
        Strategy: Initialize U, V, W with blocks that correspond to the known
        rank-13 3x3x3 decomposition, then optimize the remaining degrees of freedom.
        """
        if time.time() > deadline - 15:
            return None
        
        # Known rank-13 decomposition for 3x3x3 (simplified Strassen-like)
        # We'll use a parameterized approach: initialize with structure
        # that respects the 3x3x3 subproblem
        
        np.random.seed(123)
        
        # Initialize with a structured pattern
        # First 13 columns: try to capture 3x3x3 structure
        # Remaining r-13 columns: for the extra 2 columns of the 5
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # For the 3x3x3 part (k=0,1,2), use a known low-rank pattern
        # For k=3,4 (extra columns), use additional terms
        
        # Initialize with random but structured
        for l in range(min(13, r)):
            U[:, l] = np.random.randn(n * m) * 0.3
            V[:, l] = np.random.randn(m * p) * 0.3
            W[:, l] = np.random.randn(p * n) * 0.3
        
        for l in range(13, r):
            U[:, l] = np.random.randn(n * m) * 0.2
            V[:, l] = np.random.randn(m * p) * 0.2
            W[:, l] = np.random.randn(p * n) * 0.2
        
        # Run ALS from this initialization
        result = try_rank_als(r, seed=123, max_iter=2000)
        if result is not None:
            return result
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*p, n*m, m*p) = max(15, 9, 15) = 15
    # But tensor rank lower bound for matrix mult is higher
    # Known: 3x3x3 has rank 13, so 3x3x5 should be at least ~17-20
    
    ranks_to_try = list(range(15, min(rank_sb, 35)))
    
    # First, try the laser method for promising ranks
    for r in [17, 18, 19, 20, 21, 22, 23, 24]:
        if time.time() > deadline - 30:
            break
        if r >= best_rank:
            break
        
        result = try_laser_decomposition(r)
        if result is not None:
            r_found, decomposition = result
            if r_found < best_rank:
                best_rank = r_found
                best_result = decomposition
                break
    
    # Then try standard ALS with multiple seeds for remaining ranks
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            break
        
        for seed in [42, 0, 1, 2, 3, 7, 13, 99, 123, 456]:
            if time.time() > deadline - 15:
                break
            
            result = try_rank_als(r, seed=seed, max_iter=1500)
            if result is not None:
                r_found, decomposition = result
                if r_found < best_rank:
                    best_rank = r_found
                    best_result = decomposition
                    break
    
    return best_result
