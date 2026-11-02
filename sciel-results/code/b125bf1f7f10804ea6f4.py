import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 2, 15
    
    # Known rank-53 solution
    U0 = [[0] * 53 for _ in range(n * m)]
    V0 = [[0] * 53 for _ in range(m * p)]
    W0 = [[0] * 53 for _ in range(p * n)]
    
    A_s = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    B_s = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    C_s = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    A_b = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    B_b = [[1,0],[0,1],[1,0],[0,1]]
    C_b = [[1,0],[1,0],[0,1],[0,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = block * 2
        k1 = k0 + 1
        for r in range(7):
            c = col_offset + r
            for i in range(4):
                U0[i][c] = A_s[r][i]
            for j in range(2):
                V0[j*p + k0][c] = B_s[r][j*2 + 0]
                V0[j*p + k1][c] = B_s[r][j*2 + 1]
            W0[k0*n + 0][c] = C_s[r][0]
            W0[k1*n + 0][c] = C_s[r][1]
            W0[k0*n + 1][c] = C_s[r][2]
            W0[k1*n + 1][c] = C_s[r][3]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(4):
            U0[i][c] = A_b[r][i]
        V0[0*p + k_last][c] = B_b[r][0]
        V0[1*p + k_last][c] = B_b[r][1]
        W0[k_last*n + 0][c] = C_b[r][0]
        W0[k_last*n + 1][c] = C_b[r][1]
    
    best_result = {"U": U0, "V": V0, "W": W0}
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try to find rank-52 decomposition
    if time.time() < deadline - 10:
        try:
            rank = 52
            np.random.seed(123)
            
            # Start from rank-53 and try to merge
            U_init = np.array(U0, dtype=float)[:, :rank]
            V_init = np.array(V0, dtype=float)[:, :rank]
            W_init = np.array(W0, dtype=float)[:, :rank]
            
            # Add small random perturbation to escape local minima
            U_init += np.random.randn(*U_init.shape) * 0.01
            V_init += np.random.randn(*V_init.shape) * 0.01
            W_init += np.random.randn(*W_init.shape) * 0.01
            
            def objective(params):
                if time.time() > deadline - 5:
                    return 1e10
                U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
                U_mat = U_flat.reshape(n*m, rank)
                V_mat = V_flat.reshape(m*p, rank)
                W_mat = W_flat.reshape(p*n, rank)
                T_approx = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                return np.sum((T - T_approx)**2)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
            
            if result.fun < 1e-10:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_opt = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                # Verify
                T_check = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
                if np.linalg.norm(T - T_check) < 1e-10:
                    best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        except Exception:
            pass
    
    return best_result
