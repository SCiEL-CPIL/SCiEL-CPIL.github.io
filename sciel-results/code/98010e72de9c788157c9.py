"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute the indices for the 24 constraints
    # Constraint t corresponds to (i, j, k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                constraints.append((a, b, c))
    
    num_constraints = len(constraints)
    
    def objective_and_grads(params, rank):
        """
        params: flattened array of U, V, W
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        """
        # Reshape params
        U = params[:nm * rank].reshape((nm, rank))
        V = params[nm * rank : nm * rank + mp * rank].reshape((mp, rank))
        W = params[nm * rank + mp * rank :].reshape((pn, rank))
        
        errors = np.zeros(num_constraints)
        grads_U = np.zeros_like(U)
        grads_V = np.zeros_like(V)
        grads_W = np.zeros_like(W)
        
        # Compute errors and gradients
        for t, (a, b, c) in enumerate(constraints):
            # Term sum: sum_r U[a,r] * V[b,r] * W[c,r]
            term = np.sum(U[a] * V[b] * W[c])
            err = term - 1.0
            errors[t] = err
            
            # Gradients
            # d/dU[a,r] = err * V[b,r] * W[c,r]
            # d/dV[b,r] = err * U[a,r] * W[c,r]
            # d/dW[c,r] = err * U[a,r] * V[b,r]
            
            grads_U[a] += err * V[b] * W[c]
            grads_V[b] += err * U[a] * W[c]
            grads_W[c] += err * U[a] * V[b]
        
        obj = np.sum(errors * errors)
        # Note: objective is sum(err^2), so gradient is 2 * err * ...
        # The minimize function expects gradient of the objective function.
        # d/dx (sum err^2) = 2 * sum (err * d(err)/dx)
        # Our grads above are d(err)/dx. So multiply by 2.
        
        grad_U = 2.0 * grads_U
        grad_V = 2.0 * grads_V
        grad_W = 2.0 * grads_W
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return obj, grad
    
    def check_decomposition(U, V, W, tol=1e-6):
        for a, b, c in constraints:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        return True

    def check_and_return(U, V, W):
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Naive decomposition as fallback
    rank_naive = n * m * p
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    for idx, (a, b, c) in enumerate(constraints):
        U_naive[a, idx] = 1
        V_naive[b, idx] = 1
        W_naive[c, idx] = 1
        
    best_result = check_and_return(U_naive, V_naive, W_naive)
    best_rank = rank_naive
    
    # Try to find lower rank decompositions
    # Known rank for 2x3x4 is likely 19. Try 19 to 23.
    for target_rank in range(19, 24):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Multiple restarts
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(attempt * 123 + target_rank)
            
            # Initialize with small random values
            U_init = np.random.randn(nm, target_rank) * 0.1
            V_init = np.random.randn(mp, target_rank) * 0.1
            W_init = np.random.randn(pn, target_rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                res = minimize(
                    objective_and_grads, 
                    params_init, 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-8:
                    # Extract solution
                    U_sol = res.x[:nm * target_rank].reshape((nm, target_rank))
                    V_sol = res.x[nm * target_rank : nm * target_rank + mp * target_rank].reshape((mp, target_rank))
                    W_sol = res.x[nm * target_rank + mp * target_rank :].reshape((pn, target_rank))
                    
                    # Check validity
                    result = check_and_return(U_sol, V_sol, W_sol)
                    if result is not None:
                        best_rank = target_rank
                        best_result = result
                        found = True
                        break
                        
            except Exception as e:
                continue
        
        if found:
            # Found a valid decomposition at this rank, try to go lower?
            # The loop goes 19, 20, 21... so if we find 19, we break.
            # If we find 20, we continue to try 19? No, the loop is increasing.
            # We want the MINIMAL rank. So we should try 19 first.
            # If 19 fails, try 20.
            # So the loop `for target_rank in range(19, 24)` is correct order.
            # If we find 20, we break the inner loop, but should we break the outer?
            # Yes, because we found a solution at rank 20. But maybe 19 exists?
            # We are iterating 19, 20, 21...
            # If 19 fails, we go to 20. If 20 succeeds, we break.
            # This ensures we find the lowest rank we can find.
            break

    if best_result is None:
        best_result = check_and_return(U_naive, V_naive, W_naive)
        
    return best_result
# EVOLVE-BLOCK-END
