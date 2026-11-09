import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    def compute_error(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        error = np.sum((T - target) ** 2)
        return error

    err0 = compute_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_error = err0
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook

    def try_rank(r, num_restarts=5):
        nonlocal best_U, best_V, best_W, best_rank, best_error
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/halves
            # Using a wider range might help escape local minima
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Flatten for optimization
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_opt = x[:n*m*r].reshape(n, m, r)
                V_opt = x[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
                W_opt = x[n*m*r + m*p*r:].reshape(p, n, r)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_opt, V_opt, W_opt)
                target = np.ones((n, m, p))
                return np.sum((T - target) ** 2)
            
            # Use Nelder-Mead for local search on continuous space
            # This is faster than manual coordinate descent for small dimensions
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
                if time.time() > deadline - 5:
                    break
                
                x_opt = res.x
                U_opt = x_opt[:n*m*r].reshape(n, m, r)
                V_opt = x_opt[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
                W_opt = x_opt[n*m*r + m*p*r:].reshape(p, n, r)
                
                # Snap to nearest half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                final_err = compute_error(U_snap, V_snap, W_snap, r)
                
                if final_err < best_error:
                    best_error = final_err
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    best_rank = r
                    if best_error < 1e-9:
                        break
            except Exception:
                pass
    
    # Try rank 16 first
    try_rank(16, num_restarts=20)
    
    # If rank 16 not found, try rank 17
    if best_rank > 17 and time.time() < deadline - 10:
        try_rank(17, num_restarts=10)
    
    # Ensure we have a valid fallback
    if best_U is None:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                # Snap to nearest half-integer
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
