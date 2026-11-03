import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8

    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0

    T_flat = T.flatten()

    # Theoretical min rank is 19. We try 19 to 23.
    for r in range(19, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 50
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            # Flatten parameters into a single vector for scipy
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_curr = x[:dim_U*r].reshape(dim_U, r)
                V_curr = x[dim_U*r:dim_U*r + dim_V*r].reshape(dim_V, r)
                W_curr = x[dim_U*r + dim_V*r:].reshape(dim_W, r)
                
                T_rec = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec += np.einsum('a,b,c->abc', W_curr[:, rr], U_curr[:, rr], V_curr[:, rr])
                
                return (T_flat - T_rec.flatten())
            
            try:
                result = opt.least_squares(objective, x0, method='lm', max_nfev=1000)
            except Exception:
                continue

            if result.success or result.cost < 1e-10:
                U_opt = result.x[:dim_U*r].reshape(dim_U, r)
                V_opt = result.x[dim_U*r:dim_U*r + dim_V*r].reshape(dim_V, r)
                W_opt = result.x[dim_U*r + dim_V*r:].reshape(dim_W, r)
                
                # Try to round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                T_rec_rounded = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec_rounded += np.einsum('a,b,c->abc', W_rounded[:, rr], U_rounded[:, rr], V_rounded[:, rr])
                
                error_rounded = np.linalg.norm(T - T_rec_rounded)
                
                if error_rounded < 1e-8:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

    # If no low rank found, return schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
