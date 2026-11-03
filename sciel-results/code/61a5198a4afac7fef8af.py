import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

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
    
    # Target Tensor T
    # Indices: u (0..8), v (0..8), w (0..8)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u,v,w] = 1 if consistent, else 0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def error(U, V, W):
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(S - T)

    def rationalize(M, tol=1e-5):
        M2 = 2 * M
        M_rounded = np.round(M2)
        if np.linalg.norm(M2 - M_rounded) < tol * (np.linalg.norm(M2) + 1e-9):
            return M_rounded / 2.0
        M_int = np.round(M)
        if np.linalg.norm(M - M_int) < tol * (np.linalg.norm(M) + 1e-9):
            return M_int
        return None

    start_time = time.time()
    deadline = start_time + 110

    # Try ranks from 26 down to 23
    # Rank 26 is easier to find than 23
    for rank in [26, 25, 24, 23]:
        if time.time() > deadline:
            break
        
        # More restarts for higher ranks (easier)
        num_restarts = 50 if rank >= 25 else 20
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 100 + rank)
            
            # Sparse random initialization: entries in {-1, 0, 1}
            # This is often closer to the integer/half-integer solution structure
            U = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Ensure not all zeros
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue

            # ALS
            for it in range(200):
                if time.time() > deadline:
                    break
                den_U = np.einsum('vr,wr->r', V**2, W**2)
                num_U = np.einsum('uvw,vr,wr->ur', T, V, W)
                U = num_U / (den_U + 1e-9)
                
                den_V = np.einsum('ur,wr->r', U**2, W**2)
                num_V = np.einsum('uvw,ur,wr->vr', T, U, W)
                V = num_V / (den_V + 1e-9)
                
                den_W = np.einsum('ur,vr->r', U**2, V**2)
                num_W = np.einsum('uvw,ur,vr->wr', T, U, V)
                W = num_W / (den_W + 1e-9)
            
            err = error(U, V, W)
            
            # If error is small, try to polish with gradient descent
            if err < 1e-4:
                # Flatten parameters
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def objective(x):
                    U_t = x[:n*m*rank].reshape(n*m, rank)
                    V_t = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_t = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    S = np.einsum('ur,vr,wr->uvw', U_t, V_t, W_t)
                    return np.sum((S - T)**2)
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
                    if res.success or res.fun < err:
                        U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                        V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        err_opt = error(U_opt, V_opt, W_opt)
                        
                        if err_opt < 1e-7:
                            Ur = rationalize(U_opt)
                            Vr = rationalize(V_opt)
                            Wr = rationalize(W_opt)
                            if Ur is not None and Vr is not None and Wr is not None:
                                if error(Ur, Vr, Wr) < 1e-7:
                                    return {
                                        "U": Ur.tolist(),
                                        "V": Vr.tolist(),
                                        "W": Wr.tolist()
                                    }
                except Exception:
                    pass

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
