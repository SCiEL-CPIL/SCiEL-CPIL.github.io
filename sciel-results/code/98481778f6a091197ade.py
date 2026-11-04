"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = 145  # seconds, safety margin

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_rank(r, n, m, p, max_iters=200, seed=42):
    """Try to find rank-r decomposition via local search."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(10):
        if _elapsed() > _DEADLINE:
            break
            
        # Initialize
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        # Try to optimize
        def objective(params):
            U_flat = params[:n*m*r].reshape(n*m, r)
            V_flat = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_flat = params[n*m*r+m*p*r:].reshape(p*n, r)
            
            # Compute error: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                        err += (val - 1.0)**2
            return err
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.fun < best_err:
                best_err = result.fun
                params = result.x
                best_solution = params
        except Exception:
            pass
    
    if best_solution is None:
        return None, float('inf')
    
    U = best_solution[:n*m*r].reshape(n*m, r)
    V = best_solution[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W = best_solution[n*m*r+m*p*r:].reshape(p*n, r)
    
    return (U, V, W), best_err

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a lower-rank solution
    # Start from rank 95 and go down
    for r in range(95, 24, -1):
        if _elapsed() > _DEADLINE - 10:
            break
        
        (U, V, W), err = _try_rank(r, n, m, p, max_iters=50, seed=r)
        
        if U is not None and err < 1e-8:
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if _verify(U_round, V_round, W_round, n, m, p, tol=1e-6):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            
            # Try integer rounding
            U_round = np.round(U)
            V_round = np.round(V)
            W_round = np.round(W)
            
            if _verify(U_round, V_round, W_round, n, m, p, tol=1e-6):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
