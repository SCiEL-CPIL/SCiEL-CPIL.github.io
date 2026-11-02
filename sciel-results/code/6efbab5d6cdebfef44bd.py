"""Rewritten program: linear-algebra search for a rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _try_rank_13_search(T, deadline):
    """
    Attempts to find a rank-13 decomposition using random restarts and local optimization.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 2, 4
    dim_U, dim_V, dim_W = 4, 8, 8
    r = 13
    
    T_flat = T.flatten()
    
    def objective(vars_flat):
        vars_reshaped = vars_flat.reshape(r, -1)
        U = vars_reshaped[:, :dim_U].T
        V = vars_reshaped[:, dim_U:dim_U+dim_V].T
        W = vars_reshaped[:, dim_U+dim_V:].T
        
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_calc.flatten() - T_flat
        return residual
        
    best_result = None
    best_residual = float('inf')
    
    for i in range(20): 
        if time.time() > deadline:
            break
            
        np.random.seed(42 + i)
        init = np.random.randn(r * (dim_U + dim_V + dim_W)) * 0.5
        
        try:
            res = minimize(objective, init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_residual:
                best_residual = res.fun
                best_result = res.x
            if best_residual < 1e-8:
                break
        except Exception:
            continue
            
    if best_result is not None and best_residual < 1e-8:
        vars_reshaped = best_result.reshape(r, -1)
        U = vars_reshaped[:, :dim_U].T
        V = vars_reshaped[:, dim_U:dim_U+dim_V].T
        W = vars_reshaped[:, dim_U+dim_V:].T
        
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.allclose(T_calc, T, atol=1e-5):
            return U_int, V_int, W_int
            
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 100.0

    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    solution = _try_rank_13_search(T, deadline)
    
    if solution:
        U, V, W = solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    rank = 14
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1
    
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
