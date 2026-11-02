import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize best with schoolbook (rank 18)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    def compute_residuals(x, R):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        residuals = []
        for i in range(n):
            for k in range(p):
                w_vec = W[k * n + i, :]
                for j in range(m):
                    u_vec = U[i * m + j, :]
                    for jp in range(m):
                        v_vec = V[jp * p + k, :]
                        val = np.dot(w_vec, u_vec * v_vec)
                        target = 1.0 if j == jp else 0.0
                        residuals.append(val - target)
        return np.array(residuals)

    def check_solution(U, V, W, R, atol=1e-6):
        for i in range(n):
            for k in range(p):
                w_vec = W[k * n + i, :]
                M = np.zeros((m, m))
                for j in range(m):
                    u_vec = U[i * m + j, :]
                    for jp in range(m):
                        v_vec = V[jp * p + k, :]
                        M[j, jp] = np.dot(w_vec, u_vec * v_vec)
                if not np.allclose(M, np.eye(m), atol=atol):
                    return False
        return True

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank(R, max_attempts=100):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return
            
            total_size = n*m*R + m*p*R + p*n*R
            
            # Diverse initialization
            if attempt % 3 == 0:
                x0 = np.random.randint(-1, 2, size=total_size).astype(float)
            elif attempt % 3 == 1:
                x0 = np.random.uniform(-1, 1, size=total_size)
            else:
                x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = opt.least_squares(compute_residuals, x0, args=(R,), 
                                       method='lm', max_nfev=1000)
            except Exception:
                continue
            
            x_opt = res.x
            U = x_opt[:n*m*R].reshape(n*m, R)
            V = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            
            U_int = round_to_half_int(U)
            V_int = round_to_half_int(V)
            W_int = round_to_half_int(W)
            
            if check_solution(U_int, V_int, W_int, R, atol=1e-5):
                best_rank = R
                best_U = U_int.tolist()
                best_V = V_int.tolist()
                best_W = W_int.tolist()
                return

    for R in range(14, 18):
        if time.time() > deadline:
            break
        try_rank(R, max_attempts=50)
        if best_rank <= R:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
