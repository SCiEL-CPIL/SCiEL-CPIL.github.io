import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Target tensor: T[i, k, j, j'] = 1 if j == j' else 0
    T = np.zeros((n, p, m, m))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, k, j, j] = 1.0

    T_flat = T.flatten()
    T_int = (T * 8).astype(int)

    def compute_residuals(x, R):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        # M[i,k,j,j'] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r]
        M = np.zeros((n, p, m, m))
        for r in range(R):
            w_col = W[:, r]  # (p*n,)
            u_col = U[:, r]  # (n*m,)
            v_col = V[:, r]  # (m*p,)
            for i in range(n):
                for k in range(p):
                    w_val = w_col[k * n + i]
                    if w_val == 0:
                        continue
                    for j in range(m):
                        u_val = u_col[i * m + j]
                        if u_val == 0:
                            continue
                        for jp in range(m):
                            v_val = v_col[jp * p + k]
                            M[i, k, j, jp] += w_val * u_val * v_val
        return (M - T).flatten()

    def compute_residuals_fast(x, R):
        """Vectorized version using einsum for speed."""
        U = x[:n*m*R].reshape(n, m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
        W = x[n*m*R + m*p*R :].reshape(p, n, R)
        
        # M[i,k,j,j'] = sum_r W[k,i,r] * U[i,j,r] * V[j',k,r]
        M = np.einsum('kir,ijr,jkr->ikjj', W, U, V)
        return (M - T).flatten()

    def check_solution_exact(U, V, W, R):
        """Exact integer check: multiply by 8 to clear half-integers."""
        U_int = np.round(np.array(U) * 2).astype(int)
        V_int = np.round(np.array(V) * 2).astype(int)
        W_int = np.round(np.array(W) * 2).astype(int)
        
        M_int = np.zeros((n, p, m, m), dtype=int)
        for r in range(R):
            w_vec = W_int[:, r].reshape(p, n)
            u_vec = U_int[:, r].reshape(n, m)
            v_vec = V_int[:, r].reshape(m, p)
            term = np.einsum('ki,ij,jk->ikjj', w_vec, u_vec, v_vec)
            M_int += term
        
        return np.array_equal(M_int, T_int)

    def discrete_local_search(U, V, W, R, max_steps=30):
        """Local search on half-integer lattice."""
        U_int = np.round(np.array(U) * 2).astype(int)
        V_int = np.round(np.array(V) * 2).astype(int)
        W_int = np.round(np.array(W) * 2).astype(int)
        
        def compute_err_int(U_i, V_i, W_i):
            M_int = np.zeros((n, p, m, m), dtype=int)
            for r in range(R):
                w_vec = W_i[:, r].reshape(p, n)
                u_vec = U_i[:, r].reshape(n, m)
                v_vec = V_i[:, r].reshape(m, p)
                M_int += np.einsum('ki,ij,jk->ikjj', w_vec, u_vec, v_vec)
            diff = M_int - T_int
            return np.sum(diff * diff)

        current_err = compute_err_int(U_int, V_int, W_int)
        if current_err == 0:
            return U_int, V_int, W_int, True

        for step in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            
            # Try all single-element perturbations in U
            for idx in range(n * m * R):
                row, col = idx // R, idx % R
                for delta in [-1, 1]:
                    U_trial = U_int.copy()
                    U_trial[row, col] += delta
                    err = compute_err_int(U_trial, V_int, W_int)
                    if err < current_err:
                        U_int = U_trial
                        current_err = err
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Try all single-element perturbations in V
            for idx in range(m * p * R):
                row, col = idx // R, idx % R
                for delta in [-1, 1]:
                    V_trial = V_int.copy()
                    V_trial[row, col] += delta
                    err = compute_err_int(U_int, V_trial, W_int)
                    if err < current_err:
                        V_int = V_trial
                        current_err = err
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Try all single-element perturbations in W
            for idx in range(p * n * R):
                row, col = idx // R, idx % R
                for delta in [-1, 1]:
                    W_trial = W_int.copy()
                    W_trial[row, col] += delta
                    err = compute_err_int(U_int, V_int, W_trial)
                    if err < current_err:
                        W_int = W_trial
                        current_err = err
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return U_int, V_int, W_int, current_err == 0

    def try_rank(R, max_attempts=20):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return
            
            # Generate diverse initializations
            if attempt % 3 == 0:
                # Random sparse
                U0 = np.random.choice([-1, 0, 1], size=(n*m, R))
                V0 = np.random.choice([-1, 0, 1], size=(m*p, R))
                W0 = np.random.choice([-1, 0, 1], size=(p*n, R))
            elif attempt % 3 == 1:
                # Random dense with small values
                U0 = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
                V0 = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
                W0 = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
            else:
                # Structured: start from schoolbook-like and perturb
                U0 = np.zeros((n*m, R))
                V0 = np.zeros((m*p, R))
                W0 = np.zeros((p*n, R))
                # Assign random subsets of schoolbook terms
                for r in range(R):
                    # Pick random (i,j,k)
                    i, j, k = np.random.randint(0, n), np.random.randint(0, m), np.random.randint(0, p)
                    sign = np.random.choice([-1, 1])
                    U0[i*m + j, r] = sign
                    V0[j*p + k, r] = sign
                    W0[k*n + i, r] = sign
                # Add some noise
                U0 += np.random.choice([-1, 0, 1], size=(n*m, R)) * 0.5
                V0 += np.random.choice([-1, 0, 1], size=(m*p, R)) * 0.5
                W0 += np.random.choice([-1, 0, 1], size=(p*n, R)) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]).astype(float)
            
            # Optimize with LM
            try:
                res = opt.least_squares(compute_residuals_fast, x0, args=(R,), method='lm', max_nfev=500)
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
                
                if check_solution_exact(U_round, V_round, W_round, R):
                    best_rank = R
                    best_U = U_round.tolist()
                    best_V = V_round.tolist()
                    best_W = W_round.tolist()
                    return
                
                # Discrete local search from rounded solution
                U_int, V_int, W_int, found = discrete_local_search(U_round, V_round, W_round, R)
                if found:
                    best_rank = R
                    best_U = (U_int / 2.0).tolist()
                    best_V = (V_int / 2.0).tolist()
                    best_W = (W_int / 2.0).tolist()
                    return

    def try_rank_with_polish(R, max_attempts=10):
        """Try rank R, then attempt to reduce by zeroing columns."""
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return
            
            U0 = np.random.choice([-1, 0, 1], size=(n*m, R))
            V0 = np.random.choice([-1, 0, 1], size=(m*p, R))
            W0 = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]).astype(float)
            
            try:
                res = opt.least_squares(compute_residuals_fast, x0, args=(R,), method='lm', max_nfev=400)
            except Exception:
                continue
            
            x_opt = res.x
            U = x_opt[:n*m*R].reshape(n*m, R)
            V = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if check_solution_exact(U_round, V_round, W_round, R):
                # Try to reduce rank by zeroing columns
                U_arr = np.array(U_round)
                V_arr = np.array(V_round)
                W_arr = np.array(W_round)
                
                for col in range(R - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    # Try zeroing column col
                    U_test = U_arr.copy()
                    V_test = V_arr.copy()
                    W_test = W_arr.copy()
                    U_test[:, col] = 0
                    V_test[:, col] = 0
                    W_test[:, col] = 0
                    
                    # Re-optimize with reduced rank
                    R_new = col
                    x_red = np.concatenate([U_test[:, :R_new].flatten(), 
                                             V_test[:, :R_new].flatten(), 
                                             W_test[:, :R_new].flatten()]).astype(float)
                    try:
                        res2 = opt.least_squares(compute_residuals_fast, x_red, args=(R_new,), method='lm', max_nfev=200)
                    except Exception:
                        continue
                    
                    U2 = res2.x[:n*m*R_new].reshape(n*m, R_new)
                    V2 = res2.x[n*m*R_new : n*m*R_new + m*p*R_new].reshape(m*p, R_new)
                    W2 = res2.x[n*m*R_new + m*p*R_new :].reshape(p*n, R_new)
                    
                    U2r = np.round(U2 * 2) / 2
                    V2r = np.round(V2 * 2) / 2
                    W2r = np.round(W2 * 2) / 2
                    
                    if check_solution_exact(U2r, V2r, W2r, R_new):
                        best_rank = R_new
                        best_U = U2r.tolist()
                        best_V = V2r.tolist()
                        best_W = W2r.tolist()
                        return
                
                best_rank = R
                best_U = U_round.tolist()
                best_V = V_round.tolist()
                best_W = W_round.tolist()
                return

    # Initialize best with schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    # Try ranks lower than schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # Try from 9 up to 17
    for R in range(9, 18):
        if time.time() > deadline - 5:
            break
        try_rank(R, max_attempts=25 if R <= 12 else 15)
        if best_rank <= R:
            break

    # If we found something, try to polish further
    if best_rank < 18 and time.time() < deadline - 10:
        for R in range(max(9, best_rank - 2), best_rank):
            if time.time() > deadline - 5:
                break
            try_rank_with_polish(R, max_attempts=8)
            if best_rank <= R:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
