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

    def residuals(x, R):
        # x is flattened U, V, W
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        res = []
        # Use fixed random A, B for speed
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-2, 3, size=(n, m)).astype(float)
            B = np.random.randint(-2, 3, size=(m, p)).astype(float)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_val = np.dot(U[:, r], A_flat)
                v_val = np.dot(V[:, r], B_flat)
                w_vec = W[:, r]
                for i in range(n):
                    for k in range(p):
                        C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
            res.extend((C_true - C_approx).flatten())
        return np.array(res)

    def check_solution(U, V, W, R):
        np.random.seed(123)
        for _ in range(20):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_val = np.dot(U[:, r], A_flat)
                v_val = np.dot(V[:, r], B_flat)
                w_vec = W[:, r]
                for i in range(n):
                    for k in range(p):
                        C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
                        
            if not np.allclose(C_approx, C_true, atol=1e-4):
                return False
        return True

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank(R):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        # Try a few random starts
        for attempt in range(10):
            if time.time() > deadline:
                return
            
            x0 = np.random.randint(-2, 3, size=(n*m*R + m*p*R + p*n*R)).astype(float)
            
            # Optimize
            try:
                res = opt.least_squares(residuals, x0, args=(R,), method='lm', max_nfev=200)
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
