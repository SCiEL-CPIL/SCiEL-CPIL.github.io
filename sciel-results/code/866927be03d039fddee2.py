import numpy as np
import time
import scipy.optimize as so
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute indices for the 18 non-zero entries of the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    # Schoolbook fallback (Rank 18)
    rank_school = n * m * p
    U_school = np.zeros((size_u, rank_school))
    V_school = np.zeros((size_v, rank_school))
    W_school = np.zeros((size_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-8): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-8): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-8): return False
        
        # Check exact decomposition using rounded values
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        U_sel = U_r[u_indices, :]
        V_sel = V_r[v_indices, :]
        W_sel = W_r[w_indices, :]
        sums = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if not np.allclose(sums, 1.0, atol=1e-8):
            return False
        return True

    def get_residuals(x, R):
        # x is a flat vector of length (n*m + m*p + p*n) * R
        U = x[:size_u * R].reshape(size_u, R)
        V = x[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = x[size_u * R + size_v * R :].reshape(size_w, R)
        
        U_sel = U[u_indices, :]
        V_sel = V[v_indices, :]
        W_sel = W[w_indices, :]
        
        sums = np.sum(U_sel * V_sel * W_sel, axis=1)
        return sums - 1.0

    def solve_rank(R, max_restarts=50):
        total_vars = (size_u + size_v + size_w) * R
        
        for restart in range(max_restarts):
            # Random initialization with half-integers
            x0 = np.random.randint(-4, 5, total_vars) * 0.5
            
            try:
                # Use Levenberg-Marquardt for fast convergence
                res = so.least_squares(
                    lambda x: get_residuals(x, R), 
                    x0, 
                    method='lm', 
                    max_nfev=2000, 
                    ftol=1e-15, 
                    xtol=1e-15, 
                    gtol=1e-15
                )
                x_opt = res.x
                
                U_opt = x_opt[:size_u * R].reshape(size_u, R)
                V_opt = x_opt[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W_opt = x_opt[size_u * R + size_v * R :].reshape(size_w, R)
                
                # Check if close to half-integers and verify
                if verify_solution(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
                    
            except Exception:
                continue
                
        return None

    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit

    # Try ranks from 14 up to schoolbook (18)
    # 13 is a known lower bound for some matrix mult, but 14 is a safer start for 2x3x3?
    # Actually, let's start from 14. If 14 fails, try 15, etc.
    for R in [14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
            
        # Increase restarts for lower ranks?
        # 100 restarts should be fast enough for R <= 18 (vars ~ 18*21 = 378)
        res = solve_rank(R, max_restarts=100)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
