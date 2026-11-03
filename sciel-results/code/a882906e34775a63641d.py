"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_school = n * m * p
    
    # Build the matrix multiplication tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute the "cross" matrices for efficiency
    def try_rank(r, max_iters=50, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U using vectorized operations
            for a in range(nm):
                # Z_flat[:, r_idx] = (V[:, r_idx] * W[:, r_idx]).flatten()
                Z_flat = np.zeros((mp * pn, r))
                for r_idx in range(r):
                    Z_flat[:, r_idx] = (V[:, r_idx] * W[:, r_idx]).flatten()
                U[a, :] = np.linalg.lstsq(Z_flat, T[a].flatten(), rcond=None)[0]
            
            # Update V
            for b in range(mp):
                Z_flat = np.zeros((nm * pn, r))
                for r_idx in range(r):
                    Z_flat[:, r_idx] = (U[:, r_idx] * W[:, r_idx]).flatten()
                V[b, :] = np.linalg.lstsq(Z_flat, T[:, b, :].T.flatten(), rcond=None)[0]
            
            # Update W
            for c in range(pn):
                Z_flat = np.zeros((nm * mp, r))
                for r_idx in range(r):
                    Z_flat[:, r_idx] = (U[:, r_idx] * V[:, r_idx]).flatten()
                W[c, :] = np.linalg.lstsq(Z_flat, T[:, :, c].T.flatten(), rcond=None)[0]
            
            # Fast error check
            # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Use einsum for speed
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_approx - T)
            
            if err < 1e-6:
                for denom in [1, 2, 4]:
                    U_r = np.round(U * denom) / denom
                    V_r = np.round(V * denom) / denom
                    W_r = np.round(W * denom) / denom
                    T_approx_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.max(np.abs(T_approx_r - T)) < 1e-9:
                        return (r, U_r, V_r, W_r)
        
        return (r, U, V, W, err)
    
    # Try increasing ranks
    for r in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]:
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=30, seed=42)
        if len(result) == 4:
            rank_found, U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
