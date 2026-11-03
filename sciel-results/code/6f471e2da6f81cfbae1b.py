import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Sizes
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Target equations
    # For each i, k, j, jp:
    # sum_r W[k*n+i, r] U[i*m+j, r] V[jp*p+k, r] = delta_{j, jp}
    
    def make_residuals(R):
        U_size = U_rows * R
        V_size = V_rows * R
        W_size = W_rows * R
        
        def residuals(params):
            U_flat = params[:U_size]
            V_flat = params[U_size:U_size+V_size]
            W_flat = params[U_size+V_size:]
            
            U = U_flat.reshape((U_rows, R))
            V = V_flat.reshape((V_rows, R))
            W = W_flat.reshape((W_rows, R))
            
            res = np.zeros(n * p * m * m)
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            target = 1.0 if j == jp else 0.0
                            val = np.dot(W[k*n+i], U[i*m+j] * V[jp*p+k])
                            res[idx] = val - target
                            idx += 1
            return res
        return residuals

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    best_solution = None
    best_rank = rank_school
    
    # Try ranks from 15 to 56
    # 15 is a known lower bound estimate for 2x4
    for R in range(15, rank_school + 1):
        func = make_residuals(R)
        
        # Try multiple random restarts
        for trial in range(20):
            np.random.seed(trial * 1000 + R)
            # Initialize with random values in [-1, 1]
            params_init = np.random.uniform(-1, 1, size=(U_rows + V_rows + W_rows) * R)
            
            try:
                sol = least_squares(func, params_init, method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if sol.cost < 1e-10:
                    U_sol = sol.x[:U_rows*R].reshape((U_rows, R))
                    V_sol = sol.x[U_rows*R:U_rows*R+V_rows*R].reshape((V_rows, R))
                    W_sol = sol.x[U_rows*R+V_rows*R:].reshape((W_rows, R))
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                for jp in range(m):
                                    target = 1.0 if j == jp else 0.0
                                    val = np.dot(W_round[k*n+i], U_round[i*m+j] * V_round[jp*p+k])
                                    if abs(val - target) > 1e-9:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    
                    if valid:
                        best_rank = R
                        best_solution = (U_round, V_round, W_round)
                        break
            except Exception:
                pass
        
        if best_solution is not None:
            break

    if best_solution is None:
        U, V, W = U_school, V_school, W_school
    else:
        U, V, W = best_solution

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
