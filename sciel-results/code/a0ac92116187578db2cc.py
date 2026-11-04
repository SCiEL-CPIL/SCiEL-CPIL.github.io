import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Target ranks to try
    # Lower bound is max(n*m, m*p, p*n) = 20
    # Try from 20 up to 30
    target_ranks = list(range(20, 31))
    
    # If time is short, try fewer ranks or fewer restarts
    # We will dynamically adjust restarts based on time
    max_restarts_per_rank = 5
    
    for rank in target_ranks:
        if time.time() > deadline - 10: # Leave 10s for cleanup
            break
            
        found = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            rng = np.random.RandomState(rank * 100 + restart)
            # Scaling: values should be around 1/sqrt(rank) or similar to keep tensor values around 1
            # Actually, if U,V,W are ~N(0,1), sum of R products is ~N(0, R). 
            # We want sum to be 1. So scale by 1/sqrt(R) or 1/R?
            # E[U V W] = 0. Var(U V W) = 1. Sum of R independent vars has Var R.
            # So std dev is sqrt(R). We want mean 1.
            # Better init: small random values.
            scale = 1.0 / np.sqrt(rank)
            U_flat = rng.normal(0, scale, (n * m, rank))
            V_flat = rng.normal(0, scale, (m * p, rank))
            W_flat = rng.normal(0, scale, (p * n, rank))
            
            x0 = np.concatenate([U_flat.flatten(), V_flat.flatten(), W_flat.flatten()])
            
            def loss_and_grad(x):
                U = x[:n*m*rank].reshape(n, m, rank)
                V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
                W = x[n*m*rank + m*p*rank :].reshape(p, n, rank)
                
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Using einsum
                T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                residual = T - 1.0
                loss = np.sum(residual**2)
                
                if loss < 1e-12:
                    return loss, np.zeros_like(x)
                
                # Gradients
                # dL/dU[i,j,r] = 2 * sum_k residual[i,j,k] V[j,k,r] W[k,i,r]
                dU = 2 * np.einsum('ijk,jkr,kir->ijr', residual, V, W)
                dV = 2 * np.einsum('ijk,ijr,kir->jkr', residual, U, W)
                dW = 2 * np.einsum('ijk,ijr,jkr->kir', residual, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            # Use L-BFGS-B
            # Options: maxiter, ftol, gtol
            # We want to converge quickly.
            try:
                res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxfun': 2000})
                if res.success or res.fun < 1e-10:
                    # Check validity
                    x_sol = res.x
                    U_sol = x_sol[:n*m*rank].reshape(n, m, rank)
                    V_sol = x_sol[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
                    W_sol = x_sol[n*m*rank + m*p*rank :].reshape(p, n, rank)
                    
                    # Verify
                    T_check = np.einsum('ijr,jkr,kir->ijk', U_sol, V_sol, W_sol)
                    if np.allclose(T_check, np.ones((n, m, p)), atol=1e-8):
                        # Try to snap to integers/half-integers
                        def snap(arr):
                            arr = np.array(arr, dtype=float)
                            snapped_half = np.round(arr * 2) / 2
                            if np.allclose(arr, snapped_half, atol=1e-5):
                                return snapped_half
                            snapped_int = np.round(arr)
                            if np.allclose(arr, snapped_int, atol=1e-5):
                                return snapped_int
                            return arr
                        
                        U_snap = snap(U_sol)
                        V_snap = snap(V_sol)
                        W_snap = snap(W_sol)
                        
                        # Re-verify snapped
                        T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
                        if np.allclose(T_snap, np.ones((n, m, p)), atol=1e-6):
                            best_rank = rank
                            best_result = {
                                "U": U_snap.flatten().tolist(),
                                "V": V_snap.flatten().tolist(),
                                "W": W_snap.flatten().tolist()
                            }
                            found = True
                            break
                        else:
                            # Use non-snapped if valid
                            best_rank = rank
                            best_result = {
                                "U": U_sol.flatten().tolist(),
                                "V": V_sol.flatten().tolist(),
                                "W": W_sol.flatten().tolist()
                            }
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            # If we found a solution at this rank, we can stop 
            # because we are iterating from low to high rank.
            break
            
    return best_result
