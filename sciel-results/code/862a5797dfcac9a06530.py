import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    min_rank = max(n * m, m * p, p * n)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Pre-compute indices
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # We try ranks starting from the lower bound upwards
    max_rank_to_try = min_rank + 10 # Try up to 35
    
    for R in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            # Initialize with random values
            # Scaling is important for stability
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each (i, j), solve M_ij * u_ij = 1
                # M_ij is p x R, M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                new_U = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        # Construct M_ij
                        # V_slice: p x R
                        # W_slice: p x R
                        V_slice = V[j * p:(j + 1) * p, :]
                        W_slice = W[np.arange(p) * n + i, :]
                        M = V_slice * W_slice # p x R
                        # Solve M * u = 1
                        # Use lstsq
                        try:
                            u, residuals, rank, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except np.linalg.LinAlgError:
                            u = np.zeros(R)
                        new_U[i * m + j, :] = u
                
                U = new_U
                
                # Update V
                # For each (j, k), solve N_jk * v_jk = 1
                # N_jk is n x R, N_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
                new_V = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        U_slice = U[np.arange(n) * m + j, :]
                        W_slice = W[k * n + np.arange(n), :]
                        N = U_slice * W_slice # n x R
                        try:
                            v, residuals, rank, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        except np.linalg.LinAlgError:
                            v = np.zeros(R)
                        new_V[j * p + k, :] = v
                V = new_V
                
                # Update W
                # For each (k, i), solve P_ki * w_ki = 1
                # P_ki is m x R, P_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
                new_W = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        U_slice = U[np.arange(n) * m + np.arange(m) * 0 + np.arange(m), :] # Wait, U index is i*m+j
                        # U_slice should be U[i*m + j, :] for j in range(m)
                        # So indices are [i*m, i*m+1, ..., i*m+m-1]
                        U_slice = U[i * m:(i + 1) * m, :]
                        V_slice = V[np.arange(m) * p + k, :]
                        P = U_slice * V_slice # m x R
                        try:
                            w, residuals, rank, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        except np.linalg.LinAlgError:
                            w = np.zeros(R)
                        new_W[k * n + i, :] = w
                W = new_W
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_error < 1e-10:
                    break
            
            if best_error < 1e-8:
                # Try rounding to half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                is_exact = True
                max_diff = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                            diff = abs(val - 1.0)
                            if diff > max_diff:
                                max_diff = diff
                            if diff > 1e-9:
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                
                if is_exact:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
