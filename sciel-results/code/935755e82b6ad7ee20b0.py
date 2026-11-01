"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try ranks from low to high
    best_result = None
    
    for rank in range(1, 29):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank)
        if result is not None:
            best_result = result
            break  # Found the minimal rank
    
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

def _try_rank(n, m, p, rank):
    """Try to find a decomposition of the given rank."""
    from scipy.optimize import fsolve
    import numpy as np
    
    # Number of equations: n * m * p
    num_eq = n * m * p
    
    # Unknowns: (n*m + m*p + p*n) * rank
    num_unknowns = (n * m + m * p + p * n) * rank
    
    if num_unknowns < num_eq:
        return None
    
    def equations(vars):
        # vars is a flat array of length num_unknowns
        U_flat = vars[:n*m*rank]
        V_flat = vars[n*m*rank:n*m*rank + m*p*rank]
        W_flat = vars[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        eqs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    eqs.append(s - 1.0)
        return eqs
    
    # Initial guess
    np.random.seed(42)
    x0 = np.random.randn(num_unknowns) * 0.5
    
    try:
        sol, info, ier, msg = fsolve(equations, x0, full_output=True, maxfev=10000)
        if ier == 1:
            # Verify the solution
            U_flat = sol[:n*m*rank]
            V_flat = sol[n*m*rank:n*m*rank + m*p*rank]
            W_flat = sol[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            # Check accuracy
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-10:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    except:
        pass
    
    return None

# EVOLVE-BLOCK-END
