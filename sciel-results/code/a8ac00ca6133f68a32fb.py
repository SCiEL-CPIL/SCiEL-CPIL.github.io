import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = 0.0
                            for c in range(r):
                                actual += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {'U': U, 'V': V, 'W': W}

def _try_rank(r, n, m, p, max_iter=200, tol=1e-8, seed=0):
    np.random.seed(seed)
    
    # Precompute target tensor
    T_target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, j, k, k, i] = 1.0
                
    def objective_vec(params):
        U_flat = params[:n*m*r].reshape(n, m, r)
        V_flat = params[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
        W_flat = params[n*m*r + m*p*r :].reshape(p, n, r)
        
        T_decomp = np.einsum('ijc,j2kc,k2ic->ijj2kk2i2', U_flat, V_flat, W_flat)
        return np.sum((T_decomp - T_target)**2)

    U_init = np.random.randn(n*m, r)
    V_init = np.random.randn(m*p, r)
    W_init = np.random.randn(p*n, r)
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    res = minimize(objective_vec, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': 1e-6})
    
    if res.fun < tol:
        U_sol = res.x[:n*m*r].reshape(n*m, r)
        V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_sol = res.x[n*m*r + m*p*r :].reshape(p*n, r)
        
        if _verify(U_sol, V_sol, W_sol, n, m, p):
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            if _verify(U_round, V_round, W_round, n, m, p):
                return {'U': U_round, 'V': V_round, 'W': W_round}
            else:
                return {'U': U_round, 'V': V_round, 'W': W_round}
                
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140
    
    ranks_to_try = [30, 35, 40, 45, 50, 55, 59]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        for seed in range(5):
            if time.time() > deadline:
                break
                
            res = _try_rank(r, n, m, p, max_iter=200, tol=1e-8, seed=seed)
            if res is not None:
                return res
                
    return _schoolbook(n, m, p)
