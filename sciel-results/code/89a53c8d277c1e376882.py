import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-9):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Residual T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1
        # We can compute this efficiently
        # T[i, j, k] = (U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])) - 1? No.
        # Sum is over c.
        # Let's loop over i, j, k for simplicity since n,m,p are small.
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j, :]
                for k in range(p):
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def round_to_half_integer(M):
        return np.round(M * 2) / 2.0

    def solve_rank(r, seed=0, max_iter=100):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    # B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j, :] @ B.T = 1
                    # U_vec = (B B.T)^-1 B 1
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # C[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    C = np.zeros((n, r))
                    for i in range(n):
                        C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(C.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    # D[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    D = np.zeros((m, r))
                    for j in range(m):
                        D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(D.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # Check convergence / validity
            # Simple residual check
            err = 0
            for i in range(n):
                for j in range(m):
                    u_row = U[i * m + j, :]
                    for k in range(p):
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        val = np.dot(u_row, v_row * w_row)
                        err += (val - 1.0)**2
            if err < 1e-12:
                # Try rounding
                U_r = round_to_half_integer(U)
                V_r = round_to_half_integer(V)
                W_r = round_to_half_integer(W)
                if check_decomposition(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r
            if err < 1e-8:
                # Close, maybe try rounding with tighter check
                U_r = round_to_half_integer(U)
                V_r = round_to_half_integer(V)
                W_r = round_to_half_integer(W)
                if check_decomposition(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r

        return None

    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Try ranks from 55 down to 50
    for r in range(55, 49, -1):
        if time.time() > deadline:
            break
        for seed in range(50):
            if time.time() > deadline:
                break
            res = solve_rank(r, seed=seed, max_iter=20)
            if res is not None:
                U, V, W = res
                # Verify strictly
                if check_decomposition(U, V, W, tol=1e-9):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return best_result
