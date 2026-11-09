import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin

    # Schoolbook solution as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        # Check all zero entries? 11025 - 105 = 10920
        # It's fast enough
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_prime = b // p
                    k = b % p
                    k_prime = c // n
                    i_prime = c % n
                    if i == i_prime and j == j_prime and k == k_prime:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
        return True

    def objective(params, r):
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:].reshape(pn, r)
        
        error = 0.0
        # Non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    recon = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error += (1.0 - recon)**2
        
        # Penalty for norm
        # Norm^2 = sum_r ||u_r||^2 ||v_r||^2 ||w_r||^2
        norm_u = np.sum(U**2, axis=0)
        norm_v = np.sum(V**2, axis=0)
        norm_w = np.sum(W**2, axis=0)
        norm_sq = np.sum(norm_u * norm_v * norm_w)
        
        # Target norm is 105 (since T has 105 ones)
        # If recon is correct, norm_sq should be 105
        # But if recon has large values on zeros, norm_sq will be large
        error += 0.1 * (norm_sq - 105.0)**2
        
        return error

    # Try ranks from 104 down to 50
    for r in range(104, 49, -1):
        if time.time() > deadline:
            break
            
        for restart in range(5):
            if time.time() > deadline:
                break
                
            params = np.random.randn(nm*r + mp*r + pn*r) * 0.1
            
            from scipy.optimize import minimize
            res = minimize(objective, params, args=(r,), method='BFGS', options={'maxiter': 50})
            
            if res.fun < 1e-4:
                U = res.x[:nm*r].reshape(nm, r)
                V = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W = res.x[nm*r+mp*r:].reshape(pn, r)
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if verify(U_int, V_int, W_int):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
