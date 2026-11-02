import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Target tensor T[i, k, j, j'] = 1 if j == j' else 0
    # i in 0..n-1, k in 0..p-1, j in 0..m-1, j' in 0..m-1
    T = np.zeros((n, p, m, m))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, k, j, j] = 1.0

    def compute_residuals(x, R):
        # x is flattened U, V, W
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        U = x[:n*m*R].reshape(n, m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
        W = x[n*m*R + m*p*R :].reshape(p, n, R)
        
        M = np.einsum('kir,ijr,jkr->ikjj', W, U, V)
        residuals = (M - T).flatten()
        return residuals

    def check_solution(U, V, W, R):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_int = np.round(U_arr * 2).astype(int)
        V_int = np.round(V_arr * 2).astype(int)
        W_int = np.round(W_arr * 2).astype(int)
        
        M_int = np.zeros((n, p, m, m), dtype=int)
        for r in range(R):
            w_vec = W_int[:, r].reshape(p, n)
            u_vec = U_int[:, r].reshape(n, m)
            v_vec = V_int[:, r].reshape(m, p)
            term = np.einsum('ki,ij,jk->ikjj', w_vec, u_vec, v_vec)
            M_int += term
        
        T_int = np.zeros((n, p, m, m), dtype=int)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    T_int[i, k, j, j] = 8
        
        if np.array_equal(M_int, T_int):
            return True
        return False

    def discrete_local_search(U, V, W, R, max_steps=50):
        U_int = (U * 2).astype(int)
        V_int = (V * 2).astype(int)
        W_int = (W * 2).astype(int)
        
        def compute_residual_int(U_i, V_i, W_i):
            M_int = np.zeros((n, p, m, m), dtype=int)
            for r in range(R):
                w_vec = W_i[:, r].reshape(p, n)
                u_vec = U_i[:, r].reshape(n, m)
                v_vec = V_i[:, r].reshape(m, p)
                term = np.einsum('ki,ij,jk->ikjj', w_vec, u_vec, v_vec)
                M_int += term
            
            T_int = np.zeros((n, p, m, m), dtype=int)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        T_int[i, k, j, j] = 8
            return np.sum((M_int - T_int)**2)

        current_err = compute_residual_int(U_int, V_int, W_int)
        if current_err == 0:
            return U_int, V_int, W_int, True

        for step in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # Try perturbing U
            for i in range(n*m):
                for r in range(R):
                    for delta in [-1, 1]:
                        U_trial = U_int.copy()
                        U_trial[i, r] += delta
                        err = compute_residual_int(U_trial, V_int, W_int)
                        if err < current_err:
                            U_int = U_trial
                            current_err = err
                            improved = True
            # Try perturbing V
            for i in range(m*p):
                for r in range(R):
                    for delta in [-1, 1]:
                        V_trial = V_int.copy()
                        V_trial[i, r] += delta
                        err = compute_residual_int(U_int, V_trial, W_int)
                        if err < current_err:
                            V_int = V_trial
                            current_err = err
                            improved = True
            # Try perturbing W
            for i in range(p*n):
                for r in range(R):
                    for delta in [-1, 1]:
                        W_trial = W_int.copy()
                        W_trial[i, r] += delta
                        err = compute_residual_int(U_int, V_int, W_trial)
                        if err < current_err:
                            W_int = W_trial
                            current_err = err
                            improved = True
            if not improved:
                break
        return U_int, V_int, W_int, current_err == 0

    def try_rank(R):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        # Try a few random starts
        for attempt in range(150):
            if time.time() > deadline:
                return
            
            # Random start with small integers
            # Bias towards sparsity
            U0 = np.random.choice([-1, 0, 1], size=(n*m, R))
            V0 = np.random.choice([-1, 0, 1], size=(m*p, R))
            W0 = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]).astype(float)
            
            # Optimize
            try:
                res = opt.least_squares(compute_residuals, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
            
            x_opt = res.x
            U = x_opt[:n*m*R].reshape(n*m, R)
            V = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Try rounding to integers and half-integers
            for scale in [1, 2]:
                U_round = np.round(U * scale) / scale
                V_round = np.round(V * scale) / scale
                W_round = np.round(W * scale) / scale
                
                if check_solution(U_round, V_round, W_round, R):
                    best_rank = R
                    best_U = U_round.tolist()
                    best_V = V_round.tolist()
                    best_W = W_round.tolist()
                    return
                
                # If not exact, try discrete local search
                U_int, V_int, W_int, found = discrete_local_search(U_round, V_round, W_round, R)
                if found:
                    best_rank = R
                    best_U = (U_int / 2.0).tolist()
                    best_V = (V_int / 2.0).tolist()
                    best_W = (W_int / 2.0).tolist()
                    return

    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize best with schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    # Try ranks lower than schoolbook
    for R in range(14, 18):
        try_rank(R)
        if best_rank <= R:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
