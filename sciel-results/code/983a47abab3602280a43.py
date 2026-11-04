import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def is_exact(U, V, W):
        r = U.shape[1]
        result = np.zeros((n, m, p))
        for r_idx in range(r):
            Ur = U[:, r_idx].reshape(n, m)
            Vr = V[:, r_idx].reshape(m, p)
            Wr = W[:, r_idx].reshape(p, n)
            term = Ur[:, :, None] * Vr[None, :, :] * Wr.T[None, None, :]
            result += term
        return np.allclose(result, 1.0, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U0, V0, W0 = get_schoolbook()
    
    start_time = time.time()
    time_limit = 100
    
    best_rank = 30
    best_U, best_V, best_W = U0, V0, W0
    
    for rank in range(29, 14, -1):
        if time.time() - start_time > time_limit:
            break
            
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = U0[:, :rank].copy()
            V = V0[:, :rank].copy()
            W = W0[:, :rank].copy()
            
            # Add small random noise to break symmetry
            U += np.random.normal(0, 0.1, U.shape)
            V += np.random.normal(0, 0.1, V.shape)
            W += np.random.normal(0, 0.1, W.shape)
            
            # ALS iterations
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                    
                # Solve for U
                # For each i, j:
                # sum_r U[i, j, r] * (V[j, k, r] * W[k, i, r]) = 1 (for all k)
                # This is a linear system for U[i, j, :].
                # Let A[i, j] be a matrix of size (p, rank) where A[i, j, k, r] = V[j, k, r] * W[k, i, r]
                # We want A[i, j] @ U[i, j, :] = 1 (vector of length p)
                
                # Construct A for all i, j
                # A shape: (n, m, p, rank)
                # V shape: (m, p, rank)
                # W shape: (p, n, rank) -> W.T is (n, p, rank)
                
                # We can compute this efficiently.
                # A[i, j, k, r] = V[j, k, r] * W.T[i, k, r]
                
                # Let's just loop i, j for simplicity given small size
                for i in range(n):
                    for j in range(m):
                        # Build matrix A of size (p, rank)
                        A = np.zeros((p, rank))
                        for k in range(p):
                            for r_idx in range(rank):
                                A[k, r_idx] = V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        
                        # Solve A @ x = 1
                        # x is U[i * m + j, :]
                        try:
                            x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Similarly for V and W
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            for r_idx in range(rank):
                                A[i, r_idx] = U[i * m + j, r_idx] * W[k * n + i, r_idx]
                        try:
                            x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                            
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            for r_idx in range(rank):
                                A[j, r_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        try:
                            x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Snap to nearest 0.5
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                if is_exact(U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
            
            if best_rank == rank:
                break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
