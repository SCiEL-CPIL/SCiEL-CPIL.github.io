"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    def schoolbook():
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

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def check_half_integer(mat, tol=1e-6):
        """Check if all entries are close to half-integers (multiples of 0.5)."""
        doubled = mat * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)

    def compute_error(U, V, W):
        """Compute the sum of squared errors for the decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        return error

    def try_rank(target_rank, max_restarts=3, seed_offset=0):
        """Try to find a valid decomposition with the given rank."""
        if time.time() > deadline - 20:
            return None
            
        num_vars = (n*m + m*p + p*n) * target_rank
        
        def objective(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
                        
            return error

        def gradient(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
                        grad_U[i*m+j, :] += 2 * residual * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] += 2 * residual * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * residual * U[i*m+j, :] * V[j*p+k, :]
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        best_solution = None
        best_error = float('inf')
        
        np.random.seed(42 + seed_offset)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
                
            # Initialize with small random values
            x0 = np.random.randn(num_vars) * 0.5
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_solution = res.x
                    
            except Exception:
                pass
        
        if best_solution is None:
            return None
            
        # Extract and round to half-integers
        U_flat = best_solution[:n*m*target_rank]
        V_flat = best_solution[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = best_solution[n*m*target_rank + m*p*target_rank:]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Verify the rounded solution
        if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
        return None

    # Try ranks from low to high
    for target_rank in range(20, 36):
        if time.time() > deadline - 30:
            break
            
        result = try_rank(target_rank, max_restarts=4, seed_offset=target_rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
