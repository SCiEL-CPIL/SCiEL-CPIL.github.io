"""Matrix multiplication 2,3,9 - rank-minimal decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    nm, mp, pn = n * m, m * p, p * n  # 6, 27, 18

    # Schoolbook rank = n*m*p = 54
    schoolbook_rank = n * m * p

    # Build the target tensor: T[a, b, c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-10):
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def schoolbook():
        U = np.zeros((nm, schoolbook_rank))
        V = np.zeros((mp, schoolbook_rank))
        W = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = schoolbook_rank

    # Strategy 1: Block-diagonal decomposition (rank = n*m = 6)
    # For each (i,j) pair, we need one rank-1 term that covers all k
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1 for all k, W[k*n+i, r] = 1 for all k
    def block_diagonal():
        r_total = n * m
        U = np.zeros((nm, r_total))
        V = np.zeros((mp, r_total))
        W = np.zeros((pn, r_total))
        
        for i in range(n):
            for j in range(m):
                r = i * m + j  # rank index for this (i,j) pair
                U[i * m + j, r] = 1.0
                V[j * p:(j + 1) * p, r] = 1.0
                for k in range(p):
                    W[k * n + i, r] = 1.0
        
        return U, V, W

    # Try block diagonal first
    U_bd, V_bd, W_bd = block_diagonal()
    if verify(U_bd, V_bd, W_bd):
        U_best, V_best, W_best = U_bd, V_bd, W_bd
        best_rank = n * m

    # Strategy 2: Fast ALS using batched operations
    # Key insight: use np.linalg.lstsq on stacked matrices for speed
    def fast_als_update_U(U, V, W, T, r):
        """Update U given V, W using batched lstsq."""
        # For each a: U[a,:] solves (V*W^T)[b,c,:] * U[a,:] = T[a,b,c]
        # M_U[b,c,:] = sum_r V[b,r]*W[c,r] -> shape (mp, pn, r)
        M_U = np.einsum('br,cr->bcr', V, W)  # (mp, pn, r)
        # Flatten to (mp*pn, r)
        M_U_flat = M_U.reshape(-1, r)
        # T_flat[a] = T[a,:,:].flatten() -> shape (nm, mp*pn)
        T_flat = T.reshape(nm, -1)  # (nm, mp*pn)
        
        # Solve: for each a, U[a,:] = argmin ||M_U_flat^T * U[a,:] - T_flat[a]||
        # This is: U[a,:] = (M_U_flat @ M_U_flat.T)^{-1} M_U_flat @ T_flat[a]
        # Or use lstsq: U[a,:] solves M_U_flat.T @ U[a,:] = T_flat[a]
        # Batch: U = lstsq(M_U_flat.T, T_flat.T).T
        # Actually: min_U ||M_U_flat.T @ U.T - T_flat.T||_F
        # Solution: U.T = lstsq(M_U_flat.T, T_flat.T)
        
        # Use the normal equations for speed: U = T_flat @ M_U_flat @ pinv(M_U_flat.T @ M_U_flat)
        # Or simpler: U = T_flat @ np.linalg.pinv(M_U_flat.T)
        # But pinv is slow. Let's use: U = T_flat @ M_U_flat @ inv(G) where G = M_U_flat.T @ M_U_flat
        G = M_U_flat.T @ M_U_flat  # (r, r)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        # U[a,:] = T_flat[a] @ M_U_flat @ G_inv
        U_new = T_flat @ M_U_flat @ G_inv  # (nm, mp*pn) @ (mp*pn, r) @ (r, r) = (nm, r)
        return U_new

    def fast_als_update_V(U, V, W, T, r):
        """Update V given U, W using batched lstsq."""
        M_V = np.einsum('ar,cr->acr', U, W)  # (nm, pn, r)
        M_V_flat = M_V.reshape(-1, r)  # (nm*pn, r)
        T_flat = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
        
        G = M_V_flat.T @ M_V_flat  # (r, r)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        V_new = T_flat @ M_V_flat @ G_inv  # (mp, r)
        return V_new

    def fast_als_update_W(U, V, W, T, r):
        """Update W given U, V using batched lstsq."""
        M_W = np.einsum('ar,br->abr', U, V)  # (nm, mp, r)
        M_W_flat = M_W.reshape(-1, r)  # (nm*mp, r)
        T_flat = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
        
        G = M_W_flat.T @ M_W_flat  # (r, r)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        W_new = T_flat @ M_W_flat @ G_inv  # (pn, r)
        return W_new

    def try_rank_als(r, max_iters=300, n_restarts=10, seed=42):
        rng = np.random.default_rng(seed)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            if restart == 0:
                # Structured init: use block-diagonal pattern perturbed
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                # Initialize first n*m columns with block-diagonal pattern
                for i in range(n):
                    for j in range(m):
                        col = i * m + j
                        if col < r:
                            U[i * m + j, col] = 1.0
                            V[j * p:(j + 1) * p, col] = 1.0
                            for k in range(p):
                                W[k * n + i, col] = 1.0
                # Perturb remaining columns
                for col in range(n * m, r):
                    U[:, col] = rng.standard_normal(nm) * 0.1
                    V[:, col] = rng.standard_normal(mp) * 0.1
                    W[:, col] = rng.standard_normal(pn) * 0.1
                # Small perturbation everywhere to break symmetry
                U += rng.standard_normal((nm, r)) * 0.01
                V += rng.standard_normal((mp, r)) * 0.01
                W += rng.standard_normal((pn, r)) * 0.01
            else:
                U = rng.standard_normal((nm, r)) * 0.3
                V = rng.standard_normal((mp, r)) * 0.3
                W = rng.standard_normal((pn, r)) * 0.3
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                U = fast_als_update_U(U, V, W, T, r)
                V = fast_als_update_V(U, V, W, T, r)
                W = fast_als_update_W(U, V, W, T, r)
                
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(recon - T))
                if err < 1e-10:
                    return U, V, W
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if verify(U, V, W):
                return U, V, W
        
        return None

    # Try ranks below the current best
    for r in range(best_rank - 1, max(best_rank - 15, 1), -1):
        if time.time() > deadline:
            break
        result = try_rank_als(r, max_iters=200, n_restarts=8, seed=42 + r)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
