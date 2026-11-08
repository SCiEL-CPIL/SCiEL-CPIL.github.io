"""Matrix multiplication 2x3 x 3x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 36

    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_school

    def verify(U, V, W, atol=1e-8):
        rank = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_recon, T, atol=atol)

    def verify_half_integer(U, V, W, atol=1e-6):
        for arr in [U, V, W]:
            scaled = arr * 2
            if not np.allclose(scaled, np.round(scaled), atol=atol):
                return False
        return True

    # ALS for tensor decomposition
    def try_rank_als(r, max_iters=50, seed=42):
        if r < 1:
            return None
        if time.time() > DEADLINE:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                return None
            
            # Update U
            for a in range(n*m):
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed a: T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
                # Let M[b, c, r] = V[b, r] * W[c, r]
                M = V[:, :, None] * W[None, :, :]  # (m*p, p*n, r)
                M_flat = M.reshape(m*p * p*n, r)
                T_flat = T[a, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
                U[a, :] = sol
            
            # Update V
            for b in range(m*p):
                M = U[:, :, None] * W[None, :, :]  # (n*m, p*n, r)
                M_flat = M.reshape(n*m * p*n, r)
                T_flat = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
                V[b, :] = sol
            
            # Update W
            for c in range(p*n):
                M = U[:, :, None] * V[None, :, :]  # (n*m, m*p, r)
                M_flat = M.reshape(n*m * m*p, r)
                T_flat = T[:, :, c].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            T_recon = np.zeros((n*m, m*p, p*n))
            for rr in range(r):
                T_recon += np.einsum('i,j,k->ijk', U[:, rr], V[:, rr], W[:, rr])
            if np.linalg.norm(T - T_recon) < 1e-6:
                break
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for rr in range(r):
            T_recon += np.einsum('i,j,k->ijk', U[:, rr], V[:, rr], W[:, rr])
        if np.allclose(T_recon, T, atol=1e-5):
            return U, V, W
        return None

    # Try ranks from lower to upper
    # Lower bound: at least max(n*m, m*p, p*n) = 18
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, rank_school):
        if time.time() > DEADLINE - 10:
            break
        
        found = False
        for seed in range(5):
            if time.time() > DEADLINE - 5:
                break
            result = try_rank_als(r, max_iters=30, seed=seed)
            if result is not None:
                U, V, W = result
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify(U_snap, V_snap, W_snap) and verify_half_integer(U_snap, V_snap, W_snap):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = r
                    found = True
                    break
                elif verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    def mat_to_list(M):
        return [[float(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    
    return {"U": mat_to_list(best_U), "V": mat_to_list(best_V), "W": mat_to_list(best_W)}
