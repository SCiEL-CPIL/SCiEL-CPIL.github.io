"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(6,12,8) = 12
    min_rank = max(n * m, m * p, n * p)
    
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 20:
            break
        
        success = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Define the objective: we want Σ_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = δ_{j,j'}
            # This is a system of equations. Let's use a least-squares approach.
            
            def residual(params):
                U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
                U_mat = U_flat.reshape(n*m, rank)
                V_mat = V_flat.reshape(m*p, rank)
                W_mat = W_flat.reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for jp in range(m):
                                # Target: 1 if j==jp else 0
                                val = 0
                                for r in range(rank):
                                    val += W_mat[k*n+i, r] * U_mat[i*m+j, r] * V_mat[jp*p+k, r]
                                target = 1.0 if j == jp else 0.0
                                res.append(val - target)
                return np.array(res)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or similar for small problems
            try:
                result = minimize(lambda x: np.sum(residual(x)**2), params0, 
                                  method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if result.success or np.sum(result.fun) < 1e-8:
                    U_mat = result.x[:n*m*rank].reshape(n*m, rank)
                    V_mat = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_mat = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Verify
                    max_err = max(abs(r) for r in residual(result.x))
                    if max_err < 1e-6:
                        # Convert to lists with rounding for half-integers
                        U_list = [[round(val, 2) for val in row] for row in U_mat]
                        V_list = [[round(val, 2) for val in row] for row in V_mat]
                        W_list = [[round(val, 2) for val in row] for row in W_mat]
                        
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        success = True
                        break
            except Exception:
                continue
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
