"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin
    
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
    
    # Precompute tensor views for efficient alternating minimization
    T_vec_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # for W updates
    T_vec_a = T.reshape(nm, mp * pn)                      # for U updates  
    T_vec_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)   # for V updates
    
    def try_rank_alt(r, n_trials=10, max_iter=300):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization with various strategies
            if trial % 3 == 0:
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 3 == 1:
                # Sparse initialization
                U = np.random.choice([-1, 0, 1], size=(nm, r))
                V = np.random.choice([-1, 0, 1], size=(mp, r))
                W = np.random.choice([-1, 0, 1], size=(pn, r))
            else:
                # Scale random
                U = np.random.randn(nm, r) * np.sqrt(nm / r)
                V = np.random.randn(mp, r) * np.sqrt(mp / r)
                W = np.random.randn(pn, r) * np.sqrt(pn / r)
            
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Optimize W: fix U and V
                # X shape: (nm*mp, r), X[a*mp+b, d] = U[a,d] * V[b,d]
                X = np.zeros((nm * mp, r))
                for d in range(r):
                    X[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                
                XTX = X.T @ X + 1e-8 * np.eye(r)
                XTT = X.T @ T_vec_c.T
                W_new = np.linalg.solve(XTX, XTT).T
                
                # Optimize U: fix V and W
                # Y shape: (nm*pn, r), Y[a*pn+c, d] = V[b,d] * W[c,d] for each b
                # Actually for each a: T[a,:, :] ≈ sum_d U[a,d] * (V[:,d] * W[:,d])
                Y = np.zeros((nm * pn, r))
                for d in range(r):
                    # V[:, d] has shape (mp,), W[:, d] has shape (pn,)
                    # Outer product has shape (mp, pn)
                    # But we need: for each a, the "feature" is V[:,d] * W[:,d]
                    # This doesn't quite work. Let me reconsider.
                    # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                    # For fixed a: T[a,:,:] = sum_d U[a,d] (V[:,d] ⊗ W[:,d])
                    # So the feature for entry (b,c) at rank d is V[b,d]*W[c,d]
                    Y[:, d] = np.kron(V[:, d], W[:, d])
                
                T_vec_a = T.reshape(nm, mp * pn)
                YTY = Y.T @ Y + 1e-8 * np.eye(r)
                YTT = Y.T @ T_vec_a.T
                U_new = np.linalg.solve(YTY, YTT).T
                
                # Optimize V: fix U and W
                Z = np.zeros((mp * pn, r))
                for d in range(r):
                    Z[:, d] = np.kron(U[:, d], W[:, d])
                
                ZTZ = Z.T @ Z + 1e-8 * np.eye(r)
                ZTT = Z.T @ T_vec_b.T
                V_new = np.linalg.solve(ZTZ, ZTT).T
                
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
                    best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        return False
    
    # Try ranks from lower bound upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    lower_bound = max(n*m, m*p, p*n)
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        success = try_rank_alt(r, n_trials=6)
        if success:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
