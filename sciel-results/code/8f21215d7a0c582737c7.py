import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    if V.shape != (m * p, R) or W.shape != (p * n, R):
        return False
    
    if not np.allclose(U * 2, np.round(U * 2)): return False
    if not np.allclose(V * 2, np.round(V * 2)): return False
    if not np.allclose(W * 2, np.round(W * 2)): return False

    # Reshape for einsum
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    S = np.einsum('kir,ir,jkr->ikij', W_reshaped, U_reshaped, V_reshaped)
    
    target = np.zeros((n, p, m, m))
    for j in range(m):
        target[:, :, j, j] = 1.0
        
    if not np.allclose(S, target, atol=1e-9):
        return False
        
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    target_rank = 32
    
    def objective(params):
        R = target_rank
        U_size = n * m * R
        V_size = m * p * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        S = np.einsum('kir,ir,jkr->ikij', W, U, V)
        
        target = np.zeros((n, p, m, m))
        for j in range(m):
            target[:, :, j, j] = 1.0
            
        diff = S - target
        return np.sum(diff ** 2)

    num_attempts = 20
    for attempt in range(num_attempts):
        if time.time() > deadline - 5:
            break
            
        R = target_rank
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        np.random.seed(attempt)
        # Initialize with small half-integers
        U_init = np.random.randint(-3, 4, (U_size)) * 0.5
        V_init = np.random.randint(-3, 4, (V_size)) * 0.5
        W_init = np.random.randint(-3, 4, (W_size)) * 0.5
        
        params_init = np.concatenate([U_init, V_init, W_init])
        
        try:
            res = minimize(objective, params_init, method='BFGS', options={'maxiter': 1000, 'disp': False})
        except Exception:
            continue
            
        if res.fun < 1e-8:
            params_opt = res.x
            U_opt = params_opt[:U_size].reshape(n * m, R)
            V_opt = params_opt[U_size:U_size + V_size].reshape(m * p, R)
            W_opt = params_opt[U_size + V_size:].reshape(p * n, R)
            
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify(U_round, V_round, W_round):
                return {"U": U_round, "V": V_round, "W": W_round}
    
    return _schoolbook()
