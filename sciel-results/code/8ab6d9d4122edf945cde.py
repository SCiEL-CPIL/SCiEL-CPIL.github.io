"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def try_rank_alt(r, n_trials=20, max_iter=500):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
                
            # Random initialization
            # Sparse initialization often works better for tensor decompositions
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Scale to unit norm rows to avoid scaling issues
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                    
                # Update W
                K_W = np.einsum('ad,bd,abc->dc', U, V, T)
                G_W = (U.T @ U) * (V.T @ V)
                W_new = np.linalg.solve(G_W + 1e-8 * np.eye(r), K_W)
                
                # Update U
                K_U = np.einsum('bd,cd,abc->ad', V, W_new, T)
                G_U = (V.T @ V) * (W_new.T @ W_new)
                U_new = np.linalg.solve(G_U + 1e-8 * np.eye(r), K_U)
                
                # Update V
                K_V = np.einsum('ad,cd,abc->bd', U_new, W_new, T)
                G_V = (U_new.T @ U_new) * (W_new.T @ W_new)
                V_new = np.linalg.solve(G_V + 1e-8 * np.eye(r), K_V)
                
                # Check convergence
                T_rec = tensor_from_factors(U_new, V_new, W_new)
                err = np.linalg.norm(T_rec - T)
                
                U, V, W = U_new, V_new, W_new
                
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            
            # Final check
            T_rec = tensor_from_factors(U, V, W)
            if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec_r = tensor_from_factors(U_r, V_r, W_r)
                if np.allclose(T_rec_r, T, atol=1e-8, rtol=1e-8):
                    best_U, best_V, best_W = U_r, V_r, W_r
                else:
                    # If rounding fails, try to keep real values if allowed? 
                    # Problem says "integer/half-integer entries". 
                    # So we must round. If it fails, this rank is not valid with half-integers.
                    # We should probably return False to try other ranks or trials.
                    pass
                if np.allclose(T_rec_r, T, atol=1e-8, rtol=1e-8):
                    best_rank = r
                    return True
        return False
    
    # Try ranks from lower bound upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = 28
    # But we know 54 is achievable via block decomposition.
    # Let's try from 50 to 56.
    lower_bound = 50
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        success = try_rank_alt(r, n_trials=20)
        if success:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
