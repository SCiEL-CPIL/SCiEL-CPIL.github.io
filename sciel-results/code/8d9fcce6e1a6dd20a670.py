"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
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
    
    # Try to find a lower-rank decomposition
    # The tensor equation: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    deadline = time.time() + 140  # safety margin
    
    # Try ranks from lower bound up
    # Lower bound: at least max(n*p, m*p, n*m) = max(12, 24, 8) = 24
    # But let's try smaller ranks too
    
    for rank in range(1, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Set up the problem: find U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Total variables: nm*rank + mp*rank + pn*rank
        total_vars = (nm + mp + pn) * rank
        
        def residuals(params):
            U = params[:nm*rank].reshape(nm, rank)
            V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + attempt * 1000 + rank)
            params0 = np.random.randn(total_vars) * 0.5
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=10000)
                if result.success and np.max(np.abs(result.fun)) < 1e-10:
                    U = result.x[:nm*rank].reshape(nm, rank)
                    V = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounded version still works
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-9:
                        # Convert to lists
                        U_list = [[float(x) for x in row] for row in U_rounded]
                        V_list = [[float(x) for x in row] for row in V_rounded]
                        W_list = [[float(x) for x in row] for row in W_rounded]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Use the non-rounded solution if it's exact enough
                        # But we need integer/half-integer entries
                        # Try rounding to integers
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        
                        max_err2 = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                                    max_err2 = max(max_err2, abs(val - 1.0))
                        
                        if max_err2 < 1e-9:
                            U_list = [[float(x) for x in row] for row in U_int]
                            V_list = [[float(x) for x in row] for row in V_int]
                            W_list = [[float(x) for x in row] for row in W_int]
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
        
    return best_result
# EVOLVE-BLOCK-END
