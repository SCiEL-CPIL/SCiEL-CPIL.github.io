"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Try to find lower rank decompositions
    best_result = None
    best_rank = schoolbook_rank
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try ranks from schoolbook-1 down to some minimum
    for rank in range(schoolbook_rank - 1, 15, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                # If we found a good rank, try to go lower
                continue
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0
    # Actually for matrix multiplication, the tensor is defined as:
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    
    # Let's set up the problem: we want U, V, W such that
    # for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of n*m*p equations with 3*rank*(n*m + m*p + p*n) unknowns
    # But it's nonlinear (trilinear)
    
    # Use random initialization and gradient descent
    np.random.seed(42)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Define the objective: sum of squared errors
    def objective(params):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    error += (val - 1.0) ** 2
        return error
    
    def gradient(params):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    err = 2 * (val - 1.0)
                    # d/dU[i*m+j] = err * V[j*p+k] * W[k*n+i]
                    grad_U[i*m+j] += err * V_flat[j*p+k] * W_flat[k*n+i]
                    # d/dV[j*p+k] = err * U[i*m+j] * W[k*n+i]
                    grad_V[j*p+k] += err * U_flat[i*m+j] * W_flat[k*n+i]
                    # d/dW[k*n+i] = err * U[i*m+j] * V[j*p+k]
                    grad_W[k*n+i] += err * U_flat[i*m+j] * V_flat[j*p+k]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    result = minimize(objective, params0, jac=gradient, method='BFGS', 
                     options={'maxiter': 500, 'gtol': 1e-8})
    
    if result.success or result.fun < 1e-6:
        U_opt = result.x[:n*m*rank].reshape(n*m, rank)
        V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Check if we can snap to half-integers
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
        # Verify the snapped solution
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_snap[i*m+j], V_snap[j*p+k] * W_snap[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-9:
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
        
        # If snapping doesn't work, check if original is close enough
        # and try to find exact rational solution
        if result.fun < 1e-10:
            # Try to find exact solution using the near-optimal as starting point
            # For now, return the numerical solution if it's very accurate
            # But we need integer/half-integer entries
            pass
    
    return None

# EVOLVE-BLOCK-END
