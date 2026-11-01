import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Precompute the non-zero indices of the target tensor
    # u = i*m + j, v = j*p + k, w = k*n + i
    indices_u = []
    indices_v = []
    indices_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_u.append(i * m + j)
                indices_v.append(j * p + k)
                indices_w.append(k * n + i)
                
    indices_u = np.array(indices_u)
    indices_v = np.array(indices_v)
    indices_w = np.array(indices_w)
    
    def residual(params):
        # params is a flat array of length R * (n*m + m*p + p*n)
        R = len(params) // (n * m + m * p + p * n)
        if R == 0: return np.zeros(18)
        
        U = params[:R*n*m].reshape(n*m, R)
        V = params[R*n*m:R*n*m + R*m*p].reshape(m*p, R)
        W = params[R*n*m + R*m*p:].reshape(p*n, R)
        
        # Compute sum_r U[u,r] V[v,r] W[w,r] for the 18 points
        # U[indices_u] is 18 x R
        # V[indices_v] is 18 x R
        # W[indices_w] is 18 x R
        U_sel = U[indices_u]
        V_sel = V[indices_v]
        W_sel = W[indices_w]
        
        # Product element-wise then sum over R
        # Shape 18 x R
        prod = U_sel * V_sel * W_sel
        result = prod.sum(axis=1)
        
        # Target is 1 for all 18 points
        return result - 1.0

    best_result = None
    best_rank = rank_school
    
    # Try ranks from 16 down to 1
    # We use a deadline guard
    start_time = time.time()
    deadline = 140 # 10s margin
    
    for R in range(16, 0, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for each rank
        # The number of restarts can be adaptive based on remaining time
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale initialization to help convergence
            init_scale = 1.0
            params = np.random.randn(R * (n * m + m * p + p * n)) * init_scale
            
            try:
                res = least_squares(residual, params, method='trf', max_nfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if res.success and res.cost < 1e-10:
                    # Check if we can round to half-integers
                    # The solution might be real-valued. 
                    # If it's close to integers/half-integers, round it.
                    U_opt = res.x[:R*n*m].reshape(n*m, R)
                    V_opt = res.x[R*n*m:R*n*m + R*m*p].reshape(m*p, R)
                    W_opt = res.x[R*n*m + R*m*p:].reshape(p*n, R)
                    
                    # Try rounding to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify rounded solution
                    U_sel_r = U_round[indices_u]
                    V_sel_r = V_round[indices_v]
                    W_sel_r = W_round[indices_w]
                    prod_r = U_sel_r * V_sel_r * W_sel_r
                    result_r = prod_r.sum(axis=1)
                    
                    if np.allclose(result_r, 1.0, atol=1e-9):
                        # Found a valid half-integer solution
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                    else:
                        # If rounding fails, check if the real solution is "close enough" 
                        # to be accepted as exact by the evaluator (unlikely if it requires integers)
                        # The prompt says "integer/half-integer entries".
                        # So we must return integers or half-integers.
                        # If the real solution doesn't round to a valid solution, 
                        # we might need to try a different initialization or rank.
                        pass
                        
            except Exception as e:
                pass
                
    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
