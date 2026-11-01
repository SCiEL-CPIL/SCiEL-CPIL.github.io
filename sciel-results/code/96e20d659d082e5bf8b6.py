"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a low-rank decomposition
    # The tensor has 24 entries, all equal to 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    def residual(params, rank):
        """Compute residual for given parameter vector and rank."""
        # Unpack parameters
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute residual
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res.append(1.0 - val)
        return np.array(res)
    
    def try_rank(rank, max_iter=5000):
        """Try to find a solution with given rank."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        np.random.seed(42 + rank)
        total_vars = (n*m + m*p + p*n) * rank
        params = np.random.randn(total_vars) * 0.1
        
        best_result = None
        best_cost = np.inf
        
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                p_init = params.copy()
            else:
                p_init = np.random.randn(total_vars) * (0.1 + restart * 0.05)
            
            try:
                result = least_squares(
                    residual, p_init, args=(rank,),
                    method='lm', max_nfev=max_iter,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    best_result = result
                    if cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_result is not None and best_result.cost < 1e-8:
            # Extract solution
            U = best_result.x[:n*m*rank].reshape(n*m, rank)
            V = best_result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = best_result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Verify
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(1.0 - val))
            
            if max_err < 1e-8:
                # Convert to lists with integer/half-integer if possible
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {'U': U_list, 'V': V_list, 'W': W_list}
        
        return None
    
    # Try ranks from low to high
    for rank in range(8, 25):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition
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
