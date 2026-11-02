import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

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

    def compute_residuals(x, R):
        # x is flattened U, V, W
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        # Residuals for each i, k, j, j'
        # M_ik[j, j'] = sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r]
        # Target is delta_{j, j'}
        
        residuals = []
        for i in range(n):
            for k in range(p):
                # w_vec is W[k*n+i, :] shape (R,)
                w_vec = W[k * n + i, :]
                for j in range(m):
                    # u_vec is U[i*m+j, :] shape (R,)
                    u_vec = U[i * m + j, :]
                    for jp in range(m):
                        # v_vec is V[jp*p+k, :] shape (R,)
                        v_vec = V[jp * p + k, :]
                        # term = sum_r w_vec[r] * u_vec[r] * v_vec[r]
                        val = np.dot(w_vec, u_vec * v_vec)
                        target = 1.0 if j == jp else 0.0
                        residuals.append(val - target)
        return np.array(residuals)

    def check_solution(U, V, W, R):
        # Exact check with integer arithmetic or high precision
        # Using the direct constraint
        for i in range(n):
            for k in range(p):
                w_vec = W[k * n + i, :]
                M = np.zeros((m, m))
                for j in range(m):
                    u_vec = U[i * m + j, :]
                    for jp in range(m):
                        v_vec = V[jp * p + k, :]
                        M[j, jp] = np.dot(w_vec, u_vec * v_vec)
                # Check if M is close to Identity
                if not np.allclose(M, np.eye(m), atol=1e-6):
                    return False
        return True

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank(R):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        # Try a few random starts
        for attempt in range(20):
            if time.time() > deadline:
                return
            
            # Random start with small integers
            # Scale to avoid huge values initially
            x0 = np.random.randint(-2, 3, size=(n*m*R + m*p*R + p*n*R)).astype(float)
            
            # Optimize
            try:
                # Use least_squares on the residuals
                # method='lm' is good for small problems
                res = opt.least_squares(compute_residuals, x0, args=(R,), method='lm', max_nfev=500)
            except Exception:
                continue
            
            x_opt = res.x
            U = x_opt[:n*m*R].reshape(n*m, R)
            V = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Round to half-integers
            U_int = round_to_half_int(U)
            V_int = round_to_half_int(V)
            W_int = round_to_half_int(W)
            
            if check_solution(U_int, V_int, W_int, R):
                best_rank = R
                best_U = U_int.tolist()
                best_V = V_int.tolist()
                best_W = W_int.tolist()
                return

    for R in range(11, 18):
        try_rank(R)
        if best_rank <= R:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
