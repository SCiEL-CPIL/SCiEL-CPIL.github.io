"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the matrix multiplication tensor T[a, b, c]
    T = np.zeros((n * m, m * p, p * n))
    for i_n in range(n):
        for j in range(m):
            for k_p in range(p):
                a = i_n * m + j
                b = j * p + k_p
                c = k_p * n + i_n
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        rank = U.shape[1]
        recon = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.max(np.abs(recon - T)) < 1e-9
    
    def to_list(mat):
        return mat.tolist()
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i_n in range(n):
            for j in range(m):
                for k_p in range(p):
                    a = i_n * m + j
                    b = j * p + k_p
                    c = k_p * n + i_n
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try a fast randomized approach with very limited iterations
    def try_rank_quick(r, max_iters=10, n_restarts=2):
        """Very fast ALS attempt."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                A_kr = np.zeros((m * p * p * n, r))
                for rr in range(r):
                    A_kr[:, rr] = np.kron(V[:, rr], W[:, rr])
                T_flat = T.reshape(n * m, -1)
                U = np.linalg.lstsq(A_kr.T, T_flat.T, rcond=None)[0].T
                
                # Update V
                A_kr = np.zeros((n * m * p * n, r))
                for rr in range(r):
                    A_kr[:, rr] = np.kron(U[:, rr], W[:, rr])
                T_flat = T.transpose(1, 0, 2).reshape(m * p, -1)
                V = np.linalg.lstsq(A_kr.T, T_flat.T, rcond=None)[0].T
                
                # Update W
                A_kr = np.zeros((n * m * m * p, r))
                for rr in range(r):
                    A_kr[:, rr] = np.kron(U[:, rr], V[:, rr])
                T_flat = T.transpose(2, 0, 1).reshape(p * n, -1)
                W = np.linalg.lstsq(A_kr.T, T_flat.T, rcond=None)[0].T
            
            # Check
            recon = np.zeros((n * m, m * p, p * n))
            for rr in range(r):
                recon += np.outer(np.outer(U[:, rr], V[:, rr]), W[:, rr])
            if np.max(np.abs(recon - T)) < 1e-9:
                # Try rounding
                for scale in [1, 2]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    recon_r = np.zeros((n * m, m * p, p * n))
                    for rr in range(r):
                        recon_r += np.outer(np.outer(U_r[:, rr], V_r[:, rr]), W_r[:, rr])
                    if np.max(np.abs(recon_r - T)) < 1e-9:
                        return (U_r, V_r, W_r, r)
                return (U, V, W, r)
        return None
    
    # Try a few ranks quickly
    for r in [30, 40, 50, 60, 70]:
        if time.time() > deadline:
            break
        if r >= best[3]:
            break
        result = try_rank_quick(r, max_iters=8, n_restarts=2)
        if result is not None:
            U_r, V_r, W_r, rank_r = result
            if verify(U_r, V_r, W_r):
                best = (U_r, V_r, W_r, rank_r)
                break
    
    U_best, V_best, W_best, rank_best = best
    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END
