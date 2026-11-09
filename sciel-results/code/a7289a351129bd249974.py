"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (21, rank)
    # W shape: (p*n, rank) = (21, rank)
    
    def residual(params, rank):
        """Compute sum of squared residuals for given parameters."""
        U = params[0:rank*n*m].reshape(rank, n*m)
        V = params[rank*n*m:rank*n*m + rank*m*p].reshape(rank, m*p)
        W = params[rank*n*m + rank*m*p:].reshape(rank, p*n)
        
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i]
                    res += (s - 1.0) ** 2
        return res
    
    def residual_grad(params, rank):
        """Compute gradient of residual."""
        U = params[0:rank*n*m].reshape(rank, n*m)
        V = params[rank*n*m:rank*n*m + rank*m*p].reshape(rank, m*p)
        W = params[rank*n*m + rank*m*p:].reshape(rank, p*n)
        
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i]
                    err = 2.0 * (s - 1.0)
                    for c in range(rank):
                        gU[c, i*m+j] += err * V[c, j*p+k] * W[c, k*n+i]
                        gV[c, j*p+k] += err * U[c, i*m+j] * W[c, k*n+i]
                        gW[c, k*n+i] += err * U[c, i*m+j] * V[c, j*p+k]
        
        gparams = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        return gparams
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_solution = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from low to high
    for rank in range(3, n*m*p + 1):
        if time.time() > deadline - 10.0:
            break
        
        nparams = rank * (n*m + m*p + p*n)
        
        # Try multiple random initializations
        max_restarts = 5 if rank < 10 else 3
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + rank * 100 + restart)
            params = np.random.randn(nparams) * 0.5
            
            # Use Nelder-Mead or BFGS
            try:
                result = minimize(
                    lambda x: residual(x, rank),
                    params,
                    jac=lambda x: residual_grad(x, rank),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < 1e-6:
                    # Found a valid solution
                    U = result.x[0:rank*n*m].reshape(rank, n*m)
                    V = result.x[rank*n*m:rank*n*m + rank*m*p].reshape(rank, m*p)
                    W = result.x[rank*n*m + rank*m*p:].reshape(rank, p*n)
                    
                    # Convert to list format with integer/half-integer if possible
                    U_list = [[float(U[c, row]) for c in range(rank)] for row in range(n*m)]
                    V_list = [[float(V[c, row]) for c in range(rank)] for row in range(m*p)]
                    W_list = [[float(W[c, row]) for c in range(rank)] for row in range(p*n)]
                    
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break  # Found solution for this rank, try next rank
            except Exception:
                pass
        
        if best_rank < n * m * p:
            break
    
    # If no better solution found, return schoolbook
    if best_solution is None:
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return best_solution
# EVOLVE-BLOCK-END

