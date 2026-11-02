import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = 145
    
    # Standard rank 36
    U_school = np.zeros((n * m, n * m * p), dtype=float)
    V_school = np.zeros((m * p, n * m * p), dtype=float)
    W_school = np.zeros((p * n, n * m * p), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
                c += 1
                
    def _try_rank(n, m, p, rank, deadline, init_U=None, init_V=None, init_W=None):
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        T_flat = T.flatten()
        
        def objective(x):
            U = x[:rank * n * m].reshape(rank, n * m)
            V = x[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p)
            W = x[rank * n * m + rank * m * p :].reshape(rank, p * n)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.linalg.norm(T_recon.flatten() - T_flat) ** 2

        np.random.seed(42)
        best_error = float('inf')
        best_x = None
        
        if init_U is not None:
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_error:
                best_error = res.fun
                best_x = res.x
            if best_error < 1e-8:
                return _finalize(n, m, p, rank, best_x, T)
                
        for restart in range(30):
            if time.time() > deadline:
                break
            x0 = np.random.randn(rank * (n * m + m * p + p * n))
            x0 = x0 * 0.1
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_error:
                best_error = res.fun
                best_x = res.x
            if best_error < 1e-8:
                return _finalize(n, m, p, rank, best_x, T)
                
        if best_error < 1e-6:
            return _finalize(n, m, p, rank, best_x, T)
            
        return None

    def _finalize(n, m, p, rank, x, T):
        U = x[:rank * n * m].reshape(rank, n * m)
        V = x[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p)
        W = x[rank * n * m + rank * m * p :].reshape(rank, p * n)
        
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        if np.allclose(T_recon, T, atol=1e-5):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
        return None

    # 1. Try to find rank 21 for 3x3x3
    # Use 60s for this
    decomp_333 = _try_rank(3, 3, 3, 21, start_time + 60)
    
    # 2. Try to find rank 30 for 3x3x4
    # If we have 333 rank 21, use it to initialize the first 21 terms.
    # The last 9 terms are standard 3x3x1.
    
    init_U = None
    init_V = None
    init_W = None
    
    if decomp_333 is not None:
        U_333 = np.array(decomp_333["U"])
        V_333 = np.array(decomp_333["V"])
        W_333 = np.array(decomp_333["W"])
        
        # Construct standard 3x3x1 decomposition
        U_331 = np.zeros((3 * 3, 9), dtype=float)
        V_331 = np.zeros((3 * 1, 9), dtype=float)
        W_331 = np.zeros((1 * 3, 9), dtype=float)
        c = 0
        for i in range(3):
            for j in range(3):
                U_331[i * 3 + j, c] = 1
                V_331[j * 1 + 0, c] = 1
                W_331[0 * 3 + i, c] = 1
                c += 1
                
        # Combine
        init_U = np.vstack([U_333, U_331])
        init_V = np.vstack([V_333, V_331])
        init_W = np.vstack([W_333, W_331])
        
        # Note: The indices for V and W in 3x3x1 are different from 3x3x4.
        # In 3x3x4, V has size 3x4, W has size 4x3.
        # In 3x3x1, V has size 3x1, W has size 1x3.
        # We need to embed them into the 3x3x4 structure.
        # V_331 is 3x1. We need to put it in the first column of V_334 (3x4).
        # W_331 is 1x3. We need to put it in the first row of W_334 (4x3).
        
        V_334_init = np.zeros((3 * 4, 9), dtype=float)
        W_334_init = np.zeros((4 * 3, 9), dtype=float)
        
        for r in range(9):
            for j in range(3):
                V_334_init[j * 4 + 0, r] = V_331[j, r]
            for i in range(3):
                W_334_init[0 * 3 + i, r] = W_331[0, i]
                
        # Also, the U_331 indices are correct (i*m+j).
        
        init_V = np.vstack([V_333, V_334_init])
        init_W = np.vstack([W_333, W_334_init])
        
    decomp_334 = _try_rank(3, 3, 4, 30, deadline, init_U, init_V, init_W)
    if decomp_334 is not None:
        return decomp_334
        
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
