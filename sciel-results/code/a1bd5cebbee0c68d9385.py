import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to find rank 32
    target_rank = 32
    
    start_time = time.time()
    deadline = 110.0
    
    best_solution = None
    best_error = float('inf')
    
    # Pre-compute indices
    i_indices = range(n)
    j_indices = range(m)
    k_indices = range(p)
    
    # Function to compute error and gradient
    def compute_error_and_grad(U, V, W):
        # Flatten
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        # Reshape for computation
        # U: (n*m, R) -> (n, m, R)
        U_r = U_flat.reshape((n, m, target_rank))
        V_r = V_flat.reshape((m, p, target_rank))
        W_r = W_flat.reshape((p, n, target_rank))
        
        # Compute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # S = np.einsum('ijr,jkr,kin->ijk', U_r, V_r, W_r)
        S = np.einsum('ijr,jkr,kni->ijk', U_r, V_r, W_r)
        
        # Error
        error = np.sum((S - 1.0)**2)
        
        # Gradient
        # dE/dU[i,j,r] = 2 * sum_k (S[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
        # dE/dV[j,k,r] = 2 * sum_i (S[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
        # dE/dW[k,i,r] = 2 * sum_j (S[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
        
        grad_U = np.einsum('ijk,jkr,kin->ijr', (S-1.0)*2, V_r, W_r)
        grad_V = np.einsum('ijk,ijr,kin->jkr', (S-1.0)*2, U_r, W_r)
        grad_W = np.einsum('ijk,ijr,jkr->kin', (S-1.0)*2, U_r, V_r)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def objective(x):
        U_flat = x[:n*m*target_rank]
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = x[n*m*target_rank + m*p*target_rank :]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        return compute_error_and_grad(U, V, W)[0]
        
    def objective_grad(x):
        U_flat = x[:n*m*target_rank]
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = x[n*m*target_rank + m*p*target_rank :]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        return compute_error_and_grad(U, V, W)[1]

    # Try random restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize
        # Use small random integers or half-integers
        # Scale matters. If values are too large, gradient is large.
        # If too small, gradient is small.
        # Typical values in decompositions are small.
        
        np.random.seed(restart)
        U_init = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
        V_init = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
        W_init = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success or res.fun < best_error:
            # Extract and round
            U_opt = res.x[:n*m*target_rank].reshape((n*m, target_rank))
            V_opt = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
            W_opt = res.x[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
            
            # Round to nearest 0.5
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Check exactness
            U_r = U_rounded.reshape((n, m, target_rank))
            V_r = V_rounded.reshape((m, p, target_rank))
            W_r = W_rounded.reshape((p, n, target_rank))
            
            S = np.einsum('ijr,jkr,kni->ijk', U_r, V_r, W_r)
            
            if np.allclose(S, 1.0, atol=1e-9):
                # Found exact solution
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Update best
            if res.fun < best_error:
                best_error = res.fun
                best_solution = (U_rounded, V_rounded, W_rounded)
                
    # If no exact solution found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
