import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = 115.0 # Safety margin

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # Try to find lower rank
    # Start from 20, go up to 44
    ranks_to_try = list(range(20, 45))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            U = np.random.uniform(-1, 1, (nm, R))
            V = np.random.uniform(-1, 1, (mp, R))
            W = np.random.uniform(-1, 1, (pn, R))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat = params[:nm*R]
                V_flat = params[nm*R:nm*R + mp*R]
                W_flat = params[nm*R + mp*R:]
                
                U_ = U_flat.reshape(nm, R)
                V_ = V_flat.reshape(mp, R)
                W_ = W_flat.reshape(pn, R)
                
                M = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
                
                data_err = np.sum((M - T)**2)
                return data_err
            
            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                U_opt = res.x[:nm*R].reshape(nm, R)
                V_opt = res.x[nm*R:nm*R + mp*R].reshape(mp, R)
                W_opt = res.x[nm*R + mp*R:].reshape(pn, R)
                
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                M_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                
                if np.allclose(M_rounded, T, atol=1e-9):
                    best_rank = R
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    found = True
                    break
                    
        if found:
            break

    return best_solution
