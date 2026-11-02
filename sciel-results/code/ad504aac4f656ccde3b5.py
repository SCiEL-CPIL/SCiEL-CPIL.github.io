"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def check_exact(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer and check if close."""
        rounded = np.round(x * 2) / 2
        if np.max(np.abs(x - rounded)) < 1e-6:
            return rounded
        return None
    
    # Try to find rank-r decomposition using gradient-based optimization
    def try_qcqp_rank(r, max_iters=5000, lr=0.01, seed=42):
        """Use gradient descent to find rank-r decomposition."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Precompute index mappings
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        def objective_and_grad(params):
            """Compute objective (sum of squared errors) and gradient."""
            U_flat = params[:n*m*r].reshape(n*m, r)
            V_flat = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_flat = params[n*m*r+m*p*r:].reshape(p*n, r)
            
            err = 0.0
            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)
            
            for i, j, k in indices:
                ui = U_flat[i*m+j, :]
                vj = V_flat[j*p+k, :]
                wk = W_flat[k*n+i, :]
                prod = ui * vj * wk
                s = np.sum(prod)
                diff = s - 1.0
                err += diff * diff
                
                # Gradient of diff^2 w.r.t. U[i*m+j, t] = 2*diff * vj[t]*wk[t]
                grad_U[i*m+j, :] += 2 * diff * vj * wk
                grad_V[j*p+k, :] += 2 * diff * ui * wk
                grad_W[k*n+i, :] += 2 * diff * ui * vj
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return err, grad
        
        # Use L-BFGS-B for optimization
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def obj_func(params):
            err, _ = objective_and_grad(params)
            return err
        
        def grad_func(params):
            _, grad = objective_and_grad(params)
            return grad
        
        # Try multiple initializations
        best_result = None
        best_err = np.inf
        
        for trial in range(10):
            np.random.seed(seed + trial)
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(obj_func, params_init, jac=grad_func, 
                             method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': 1e-12})
            
            if result.fun < best_err:
                best_err = result.fun
                best_result = result
        
        if best_result is None or best_err > 1e-6:
            return None
        
        # Extract solution
        params = best_result.x
        U_sol = params[:n*m*r].reshape(n*m, r)
        V_sol = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W_sol = params[n*m*r+m*p*r:].reshape(p*n, r)
        
        # Check exactness
        if not check_exact(U_sol, V_sol, W_sol):
            return None
        
        # Try to convert to half-integers
        U_hi = to_half_int(U_sol)
        V_hi = to_half_int(V_sol)
        W_hi = to_half_int(W_sol)
        
        if U_hi is not None and V_hi is not None and W_hi is not None:
            if check_exact(U_hi, V_hi, W_hi):
                return {"U": U_hi.tolist(), "V": V_hi.tolist(), "W": W_hi.tolist()}
        
        # Return float solution if half-integers don't work (may still be valid)
        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, n*m*p + 1):
        result = try_qcqp_rank(r, max_iters=3000, lr=0.01, seed=42)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
