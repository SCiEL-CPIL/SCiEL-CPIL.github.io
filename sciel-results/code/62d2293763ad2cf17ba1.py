import numpy as np
import time
import itertools
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
DEADLINE = 145.0  # 5s margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((N, rank_sb))
    V_sb = np.zeros((M, rank_sb))
    W_sb = np.zeros((P, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 for all i, j, k
    # The decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # We will search for a rank-minimal decomposition.
    # Known results:
    # - 3x3x3: rank 23 (Strassen-like)
    # - 4x4x4: rank 49 (Strassen-like)
    # - 3x4x5: likely around 50-60?
    # Let's try to find a low rank.
    
    # We use a grid search or random search with local optimization.
    # Since we need exact rational solutions, we can try to find a solution
    # with small integer entries.
    
    # Let's try ranks from a lower bound upwards.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # But tensor rank is usually higher.
    # For 3x4x5, let's try ranks 30 to 60.
    
    def reconstruct(U, V, W):
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T
    
    def check_error(U, V, W):
        T = reconstruct(U, V, W)
        T_target = np.ones((n, m, p))
        err = np.sum((T - T_target)**2)
        return err, T
    
    def try_rank(R, max_restarts=10):
        from scipy.optimize import minimize
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
        def unflatten(x):
            size_U = N * R
            size_V = M * R
            U = x[:size_U].reshape(N, R)
            V = x[size_U:size_U+size_V].reshape(M, R)
            W = x[size_U+size_V:].reshape(P, R)
            return U, V, W
            
        def objective(x):
            U, V, W = unflatten(x)
            err, _ = check_error(U, V, W)
            return err
        
        best_err = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 10.0:
                break
                
            # Random init
            U0 = np.random.randn(N, R) * 0.5
            V0 = np.random.randn(M, R) * 0.5
            W0 = np.random.randn(P, R) * 0.5
            
            x0 = flatten(U0, V0, W0)
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-4})
            
            if res.fun < best_err:
                best_err = res.fun
                best_U, best_V, best_W = unflatten(res.x)
                
                if best_err < 1e-8:
                    break
        
        if best_err < 1e-6:
            # Try to snap to rationals
            # We can try to find a rational approximation
            # For now, just check if it's close to half-integers
            U_snap = np.round(best_U * 2) / 2.0
            V_snap = np.round(best_V * 2) / 2.0
            W_snap = np.round(best_W * 2) / 2.0
            err_snap, _ = check_error(U_snap, V_snap, W_snap)
            if err_snap < 1e-8:
                return U_snap, V_snap, W_snap, R
            else:
                # Try integer snap
                U_snap = np.round(best_U)
                V_snap = np.round(best_V)
                W_snap = np.round(best_W)
                err_snap, _ = check_error(U_snap, V_snap, W_snap)
                if err_snap < 1e-8:
                    return U_snap, V_snap, W_snap, R
        return None

    # Try ranks from 30 to 60
    for R in range(30, 61):
        if time.time() > DEADLINE - 20.0:
            break
        sol = try_rank(R, max_restarts=5)
        if sol:
            U, V, W, r = sol
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
