import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def make_rank11():
        # ... (same as before) ...
        pass

    def verify_solution(sol, n_trials=10):
        # ... (same as before) ...
        pass

    sol11 = make_rank11()

    def try_rank10():
        if time_remaining() < 10:
            return None
            
        rank = 10
        n_vars = (n*m + m*p + p*n) * rank
        # n*m=4, m*p=6, p*n=6 -> 16*10=160 vars
        
        def objective(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            T_target = np.ones((n, m, p))
            T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            
            error = np.sum((T_computed - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        # Strategy 1: Sparse Random Restarts
        # The optimal solution is likely sparse.
        # Initialize with random sparse matrices.
        for i in range(200): # Try 200 restarts
            if time_remaining() < 5:
                break
                
            # Generate random sparse initialization
            U = np.zeros((n, m, rank))
            V = np.zeros((m, p, rank))
            W = np.zeros((p, n, rank))
            
            for r in range(rank):
                # Pick 2 random indices for U
                u_idx = np.random.choice(n*m, 2, replace=False)
                u_vals = np.random.choice([-1, 1], 2)
                for idx, val in zip(u_idx, u_vals):
                    U[idx // m, idx % m, r] = val
                    
                # Pick 2 random indices for V
                v_idx = np.random.choice(m*p, 2, replace=False)
                v_vals = np.random.choice([-1, 1], 2)
                for idx, val in zip(v_idx, v_vals):
                    V[idx // p, idx % p, r] = val
                    
                # Pick 2 random indices for W
                w_idx = np.random.choice(p*n, 2, replace=False)
                w_vals = np.random.choice([-1, 1], 2)
                for idx, val in zip(w_idx, w_vals):
                    W[idx // n, idx % n, r] = val
                    
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-10:
                break

        # Strategy 2: Warm start from rank 11 (drop 1 term)
        # Already in current program, but let's ensure it's there.
        sol11 = make_rank11()
        U11 = np.array(sol11["U"], dtype=float).reshape(n, m, 11)
        V11 = np.array(sol11["V"], dtype=float).reshape(m, p, 11)
        W11 = np.array(sol11["W"], dtype=float).reshape(p, n, 11)
        
        for drop_idx in range(11):
            if time_remaining() < 5:
                break
            mask = [i for i in range(11) if i != drop_idx]
            U0 = U11[:,:,mask]
            V0 = V11[:,:,mask]
            W0 = W11[:,:,mask]
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-10:
                break

        if best_error < 1e-8 and best_params is not None:
            # Verify and return
            U_flat = best_params[:n*m*rank]
            V_flat = best_params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = best_params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            U_list = [[0]*rank for _ in range(n*m)]
            V_list = [[0]*rank for _ in range(m*p)]
            W_list = [[0]*rank for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        U_list[i*m+j][r] = U[i,j,r]
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        V_list[j*p+k][r] = V[j,k,r]
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        W_list[k*n+i][r] = W[k,i,r]
                        
            sol10 = {"U": U_list, "V": V_list, "W": W_list}
            
            if verify_solution(sol10, n_trials=10):
                return sol10
                
        return None

    sol10 = try_rank10()
    if sol10 is not None:
        return sol10
    else:
        return sol11

# EVOLVE-BLOCK-END
