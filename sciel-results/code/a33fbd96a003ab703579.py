"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
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
    
    def peel_greedy(seed, max_iters=50):
        """Greedy rank-1 peeling."""
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if np.allclose(residual, 0, atol=1e-10):
                break
            
            # Try to find a good rank-1 component
            # Sample random u, v, then compute optimal w
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            n_trials = 200
            for t in range(n_trials):
                # Random u and v
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                
                # Reshape for computation
                u_mat = u.reshape(n, m)  # u[i,j]
                v_mat = v.reshape(m, p)  # v[j,k]
                
                # For each (i,k), the contribution to T[i,j,k] summed over j:
                # sum_j u[i,j] * v[j,k] * w[k,i]
                # We want this to approximate residual[i,j,k] for all j
                # Optimal w[k,i] = sum_j residual[i,j,k] * u[i,j] * v[j,k] / sum_j (u[i,j]*v[j,k])^2
                # But we need it to work for all j simultaneously.
                # Score: how well does a single w[k,i] per (k,i) fit?
                
                # Compute for each (i,k):
                # numerator[i,k] = sum_j residual[i,j,k] * u[i,j] * v[j,k]
                # denominator[i,k] = sum_j (u[i,j] * v[j,k])^2
                # w_candidate[k*n+i] = numerator[i,k] / denominator[i,k]
                
                # Build numerator and denominator arrays
                num = np.zeros((n, p))
                den = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        uj = u_mat[i, :]  # shape (m,)
                        vk = v_mat[:, k]  # shape (m,)
                        ujvk = uj * vk  # shape (m,)
                        num[i, k] = np.dot(residual[i, :, k], ujvk)
                        den[i, k] = np.dot(ujvk, ujvk)
                
                # Avoid division by zero
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                for i in range(n):
                    for k in range(p):
                        if mask[i, k]:
                            w_flat[k * n + i] = num[i, k] / den[i, k]
                
                # Score: norm of residual - u⊗v⊗w
                # Compute the rank-1 tensor
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_flat[k * n + i]
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            # Check if this peel actually helps
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] = best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]
            
            new_residual = residual - recon
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-12:
                # This peel doesn't help, try a different approach
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
        recon = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        if np.allclose(recon, T, atol=1e-8):
            return U, V, W, rank
        else:
            return None
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() > deadline - 15.0:
            break
        result = peel_greedy(seed)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
                print(f"Found rank {rank} with seed {seed}")
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
