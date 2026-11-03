import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    min_rank = max(n * m, m * p, p * n)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140 # 140s to leave 10s margin for evaluation overhead
    
    # Try ranks from lower bound up to a reasonable limit
    # The trivial rank is 100.
    # We hope to find something much lower.
    max_rank_to_try = 60 # Trying up to 60
    
    for R in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank
        num_restarts = 5
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        V_slice = V[j * p:(j + 1) * p, :]
                        W_idx = np.array([k * n + i for k in range(p)])
                        W_slice = W[W_idx, :]
                        C_ij = V_slice * W_slice
                        try:
                            u, _, _, _ = np.linalg.lstsq(C_ij, np.ones(p), rcond=None)
                        except np.linalg.LinAlgError:
                            u = np.zeros(R)
                        U[i * m + j, :] = u

                # Solve for V
                for j in range(m):
                    for k in range(p):
                        U_idx = np.array([i * m + j for i in range(n)])
                        U_slice = U[U_idx, :]
                        W_idx = np.array([k * n + i for i in range(n)])
                        W_slice = W[W_idx, :]
                        D_jk = U_slice * W_slice
                        try:
                            v, _, _, _ = np.linalg.lstsq(D_jk, np.ones(n), rcond=None)
                        except np.linalg.LinAlgError:
                            v = np.zeros(R)
                        V[j * p + k, :] = v

                # Solve for W
                for k in range(p):
                    for i in range(n):
                        U_idx = np.array([i * m + j for j in range(m)])
                        U_slice = U[U_idx, :]
                        V_idx = np.array([j * p + k for j in range(m)])
                        V_slice = V[V_idx, :]
                        E_ki = U_slice * V_slice
                        try:
                            w, _, _, _ = np.linalg.lstsq(E_ki, np.ones(m), rcond=None)
                        except np.linalg.LinAlgError:
                            w = np.zeros(R)
                        W[k * n + i, :] = w

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
                    
                if best_error < 1e-12:
                    break

            # Check if we have a good solution
            if best_error < 1e-8:
                # Snap to half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                # Verify exactness
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                
                if is_exact:
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        # If no exact solution found for this rank, continue to next rank

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
