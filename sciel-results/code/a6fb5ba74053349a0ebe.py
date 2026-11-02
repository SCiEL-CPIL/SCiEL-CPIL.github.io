import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    # Schoolbook rank 27
    rank_sb = 27
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = rank_sb

    def objective(vars):
        r = len(vars) // 3
        U_f = vars[:nm*r].reshape(nm, r)
        V_f = vars[nm*r:2*nm*r].reshape(mp, r)
        W_f = vars[2*nm*r:3*nm*r].reshape(pn, r)
        
        errors = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W_f[k*n+i, :] * U_f[i*m+j, :] * V_f[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        errors.append(s - expected)
        return np.array(errors)

    def try_rank(r, num_starts=10):
        for start_idx in range(num_starts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(42 + start_idx)
            
            # Initialize with schoolbook-like structure
            U_init = np.zeros((nm, r))
            V_init = np.zeros((mp, r))
            W_init = np.zeros((pn, r))
            
            # Fill with random values
            U_init += np.random.randn(nm, r) * 0.5
            V_init += np.random.randn(mp, r) * 0.5
            W_init += np.random.randn(pn, r) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(objective, x0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                sol = res.x
                U_sol = sol[:nm*r].reshape(nm, r)
                V_sol = sol[nm*r:2*nm*r].reshape(mp, r)
                W_sol = sol[2*nm*r:3*nm*r].reshape(pn, r)
                
                # Round to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-3):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
    if time.time() - start_time < deadline - 20:
        r_target = 23
        result = try_rank(r_target, num_starts=15)
        if result:
            best_U, best_V, best_W = result
            best_rank = r_target

    return {"U": best_U, "V": best_V, "W": best_W}
