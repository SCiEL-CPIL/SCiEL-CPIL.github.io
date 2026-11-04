import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Build target tensor
    target = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    
    # Schoolbook baseline
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    def verify_decomp(U, V, W, tol=1e-9):
        """Verify decomposition correctness using the tensor equation."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank, max_restarts=5):
        """Try to find a decomposition of given rank using least_squares."""
        if time.time() > deadline - 10:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(rank * 100 + restart)
            
            # Initialize with structured random values
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            def flatten(params):
                U_flat = params[:n*m*rank]
                V_flat = params[n*m*rank : n*m*rank + m*p*rank]
                W_flat = params[n*m*rank + m*p*rank :]
                return (U_flat.reshape(n*m, rank), 
                        V_flat.reshape(m*p, rank), 
                        W_flat.reshape(p*n, rank))
            
            def residuals(params):
                U, V, W = flatten(params)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            res.append(val - 1.0)
                return np.array(res)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                
                if result.success or result.cost < 1e-15:
                    U_opt, V_opt, W_opt = flatten(result.x)
                    
                    # Check exactness
                    if verify_decomp(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), tol=1e-8):
                        return U_opt, V_opt, W_opt
            except Exception:
                continue
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 20, but start higher for efficiency
    for target_rank in range(30, base_rank):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, max_restarts=3)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
