import numpy as np
import time
import random
import itertools
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def validate(U, V, W, atol=1e-5):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    def compute_residual(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return (prod - T).flatten()

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = None
    best_rank = n * m * p # 50
    
    # Precompute schoolbook matrices for initialization
    U_sb = np.zeros((U_dim, 50))
    V_sb = np.zeros((V_dim, 50))
    W_sb = np.zeros((W_dim, 50))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    # Try ranks from 49 down to 25
    # Prioritize higher ranks as they are easier to find
    ranks_to_try = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 35, 30, 25]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        R = target_rank
        # If we already found a better rank, stop
        if best_solution and R >= best_rank:
            continue
            
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            random.seed(42 + restart * 100 + R)
            np.random.seed(42 + restart * 100 + R)
            
            # Initialization Strategy 1: Collapse Schoolbook
            # Pick R random columns from schoolbook
            kept_indices = random.sample(range(50), R)
            U_init = U_sb[:, kept_indices].copy()
            V_init = V_sb[:, kept_indices].copy()
            W_init = W_sb[:, kept_indices].copy()
            
            # Add small noise to escape local minima
            U_init += np.random.randn(U_dim, R) * 0.1
            V_init += np.random.randn(V_dim, R) * 0.1
            W_init += np.random.randn(W_dim, R) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual_func(x):
                u = x[:U_dim*R].reshape(U_dim, R)
                v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                return compute_residual(u, v, w)
            
            try:
                # Use least_squares for better convergence on sum of squares
                res = least_squares(residual_func, x0, method='trf', max_nfev=1000)
                
                if res.cost < 1e-8:
                    u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                    v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    
                    valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        break # Found valid solution for this rank, try lower rank
            except Exception as e:
                pass
            
            # Initialization Strategy 2: Random
            U_init = np.random.randn(U_dim, R) * 0.5
            V_init = np.random.randn(V_dim, R) * 0.5
            W_init = np.random.randn(W_dim, R) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residual_func, x0, method='trf', max_nfev=500)
                
                if res.cost < 1e-8:
                    u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                    v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    
                    valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        break
            except Exception as e:
                pass

    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
