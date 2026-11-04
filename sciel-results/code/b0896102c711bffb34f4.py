import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs T."""
        recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            # term[i, j, k] = u[i, j] * v[j, k] * w[k, i]
            # We can compute this efficiently using einsum or broadcasting
            # Let's use broadcasting for clarity and speed on small tensors
            # u: (n, m, 1), v: (1, m, p), w: (p, 1, n) -> need to align indices
            # Actually, let's just do the loop for small n,m,p to be safe and simple,
            # or vectorize per rank term.
            
            # Vectorized update for this rank term:
            # recon += np.einsum('ij,jk,ki->ijk', u, v, w)
            recon += np.einsum('ij,jk,ki->ijk', u, v, w)
            
        return np.allclose(recon, T, atol=1e-8)

    def snap_to_half(U, V, W):
        """Snap values to nearest 0.5 to satisfy integer/half-integer constraint."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    def peel_greedy_vectorized(seed, max_iters=50):
        """
        Greedy peeling with random restarts and vectorized scoring.
        """
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if np.allclose(residual, 0, atol=1e-10):
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            # Try multiple random initializations for u, v
            n_trials = 200
            for t in range(n_trials):
                if time.time() > deadline - 5.0:
                    break
                    
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                # Compute optimal w for given u, v
                # We want to minimize || residual - u \circ v \circ w ||
                # This is linear in w.
                # Let A be the linear operator mapping w to u \circ v \circ w.
                # A[w] has entries: (A[w])[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                # We want to solve A^T A w = A^T residual
                # Or simply, for each (k,i), w[k,i] is determined by:
                # sum_{j} residual[i,j,k] * u[i,j] * v[j,k] = w[k,i] * sum_{j} (u[i,j] * v[j,k])^2
                
                u_exp = u_mat[:, :, np.newaxis] # (n, m, 1)
                v_exp = v_mat[np.newaxis, :, :] # (1, m, p)
                
                prod = u_exp * v_exp # (n, m, p)
                
                # num[i, k] = sum_j residual[i, j, k] * prod[i, j, k]
                num = np.sum(residual * prod, axis=1) # (n, p)
                
                # den[i, k] = sum_j prod[i, j, k]^2
                den = np.sum(prod * prod, axis=1) # (n, p)
                
                mask = den > 1e-12
                w_opt = np.zeros((n, p))
                np.divide(num, den, out=w_opt, where=mask)
                
                # w_flat index: k*n + i
                # w_opt index: i, k
                w_flat = w_opt.T.flatten() # (p, n) flattened to (p*n)
                
                # Compute reconstruction
                recon = u_exp * v_exp * w_opt[:, np.newaxis, :] # (n, m, p)
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            # Verify improvement
            u_mat = best_u.reshape(n, m)
            v_mat = best_v.reshape(m, p)
            w_opt = best_w.reshape(p, n).T # (n, p)
            
            recon = u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_opt[:, np.newaxis, :]
            new_residual = residual - recon
            
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-12:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        # Verify
        if verify(U, V, W, rank):
            return U, V, W, rank
        else:
            return None

    # Try peeling with multiple seeds
    for seed in range(200):
        if time.time() > deadline - 10.0:
            break
        result = peel_greedy_vectorized(seed, max_iters=30)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
                print(f"Found rank {rank} with seed {seed}")

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
