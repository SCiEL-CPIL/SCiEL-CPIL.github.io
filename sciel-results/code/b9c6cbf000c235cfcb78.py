# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 100  # Safety margin for 120s limit
    
    # Define target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    def compute_error(params):
        R = len(params) // (n*m + m*p + p*n)
        size_u = n*m * R
        size_v = m*p * R
        
        U = params[:size_u].reshape(n*m, R)
        V = params[size_u:size_u+size_v].reshape(m*p, R)
        W = params[size_u+size_v:].reshape(p*n, R)
        
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        return (T_recon - T).ravel()

    def try_rank(R, max_restarts=5):
        np.random.seed(42)
        best_sol = None
        best_err = np.inf
        
        for _ in range(max_restarts):
            if time.time() - start_time > timeout:
                break
                
            U0 = np.random.randint(-2, 3, (n*m, R))
            V0 = np.random.randint(-2, 3, (m*p, R))
            W0 = np.random.randint(-2, 3, (p*n, R))
            
            x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
            
            try:
                res = least_squares(compute_error, x0, method='trf', max_nfev=2000)
                if res.cost < best_err:
                    best_err = res.cost
                    best_sol = res.x
            except Exception:
                continue
                
        if best_sol is None:
            return None
            
        size_u = n*m * R
        size_v = m*p * R
        U_opt = best_sol[:size_u].reshape(n*m, R)
        V_opt = best_sol[size_u:size_u+size_v].reshape(m*p, R)
        W_opt = best_sol[size_u+size_v:].reshape(p*n, R)
        
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U_round[:, r], V_round[:, r], W_round[:, r])
            
        if np.allclose(T_recon, T, atol=1e-6):
            return U_round, V_round, W_round
            
        return None

    # Try rank 16
    if time.time() - start_time < timeout:
        sol_16 = try_rank(16)
        if sol_16:
            U, V, W = sol_16
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 17
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
