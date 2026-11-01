import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Seed solution (Rank 24) - schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)

    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def check_exact(U, V, W):
        # Verify exactness with integer/half-integer logic
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def try_rank_als(r, max_iters=100, num_restarts=200, seed_base=42):
        if time.time() > deadline - 10.0:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                return None
                
            np.random.seed(seed_base + restart)
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Normalize initial guesses to prevent explosion/collapse
            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    return None
                
                # Build linear systems for U, V, W
                A_U = np.zeros((num_constraints, n * m * r))
                A_V = np.zeros((num_constraints, m * p * r))
                A_W = np.zeros((num_constraints, p * n * r))
                
                # Fill matrices efficiently
                for c in range(num_constraints):
                    u_c, v_c, w_c = constraints[c]
                    idx_u = u_c * r + np.arange(r)
                    idx_v = v_c * r + np.arange(r)
                    idx_w = w_c * r + np.arange(r)
                    
                    A_U[c, idx_u] = V[v_c, :] * W[w_c, :]
                    A_V[c, idx_v] = U[u_c, :] * W[w_c, :]
                    A_W[c, idx_w] = U[u_c, :] * V[v_c, :]
                
                # Solve least squares
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_constraints), rcond=None)
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_constraints), rcond=None)
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_constraints), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                
                U = U_flat.reshape(n * m, r)
                V = V_flat.reshape(m * p, r)
                W = W_flat.reshape(p * n, r)
                
                # Normalize to keep values in check (scaling invariance)
                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)
                
                # Check error
                err = error_function(U, V, W)
                
                if err < 1e-8:
                    # Try to round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if check_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # If direct rounding fails, try integer rounding
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    
                    if check_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                        
                    # If both fail, it might be a scaling issue or local min.
                    # Break to try next restart.
                    break
                
                if err > prev_err * 1.01:
                    # Diverging or stuck, break early for this restart
                    break
                prev_err = err

        return None

    # Try lower ranks in descending order of probability of success
    # Rank 23 is likely the easiest to find if 24 is not minimal.
    # Rank 22 is next.
    
    ranks_to_try = [23, 22, 21, 20]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Allocate more time/restarts for higher ranks (easier)
        if r == 23:
            sol = try_rank_als(r, max_iters=50, num_restarts=300, seed_base=r*100)
        elif r == 22:
            sol = try_rank_als(r, max_iters=50, num_restarts=200, seed_base=r*100)
        else:
            sol = try_rank_als(r, max_iters=50, num_restarts=100, seed_base=r*100)
            
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    # If we didn't find a better rank, return the seed
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
