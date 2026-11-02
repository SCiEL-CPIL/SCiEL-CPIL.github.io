"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

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
    
    # Precompute T in different orientations for efficiency
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # For optimizing W
    T_vec_u = T.reshape(nm, mp * pn)  # For optimizing U
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # For optimizing V
    
    # Try alternating minimization for various ranks
    def try_rank_alt(r, n_trials=10, max_iter=300):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization with structured patterns
            if trial % 3 == 0:
                # Standard random
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 3 == 1:
                # Sparse-like initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    idx_u = np.random.choice(nm, size=max(1, nm//4), replace=False)
                    idx_v = np.random.choice(mp, size=max(1, mp//4), replace=False)
                    idx_w = np.random.choice(pn, size=max(1, pn//4), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            else:
                # Block-structured initialization
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
                # Add some structure
                for d in range(r):
                    U[d % nm, d] += 1.0
                    V[d % mp, d] += 1.0
                    W[d % pn, d] += 1.0
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Build X for optimizing W: X[a*mp+b, d] = U[a,d]*V[b,d]
                X = np.zeros((nm * mp, r))
                for d in range(r):
                    X[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                
                # Solve for W
                XTX = X.T @ X + 1e-10 * np.eye(r)
                XTT = X.T @ T_vec_w.T
                try:
                    W_new = np.linalg.solve(XTX, XTT).T
                except np.linalg.LinAlgError:
                    W_new = W
                
                # Build Y for optimizing U: Y[b*pn+c, d] = V[b,d]*W[c,d]
                Y = np.zeros((mp * pn, r))
                for d in range(r):
                    Y[:, d] = (V[:, d][:, None] * W_new[:, d][None, :]).flatten()
                
                # Solve for U
                YTY = Y.T @ Y + 1e-10 * np.eye(r)
                YTT = Y.T @ T_vec_u.T
                try:
                    U_new = np.linalg.solve(YTY, YTT).T
                except np.linalg.LinAlgError:
                    U_new = U
                
                # Build Z for optimizing V: Z[a*pn+c, d] = U[a,d]*W[c,d]
                Z = np.zeros((nm * pn, r))
                for d in range(r):
                    Z[:, d] = (U_new[:, d][:, None] * W_new[:, d][None, :]).flatten()
                
                # Solve for V
                ZTZ = Z.T @ Z + 1e-10 * np.eye(r)
                ZTT = Z.T @ T_vec_v.T
                try:
                    V_new = np.linalg.solve(ZTZ, ZTT).T
                except np.linalg.LinAlgError:
                    V_new = V
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_rec = tensor_from_factors(U, V, W)
                err = np.linalg.norm(T_rec - T)
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
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But known results suggest lower ranks may exist
    lower_bound = max(n*m, m*p, p*n)
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        success = try_rank_alt(r, n_trials=8)
        if success:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
