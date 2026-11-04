import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    target_rank = N * M * P  # 120

    # Precompute index arrays for fast tensor evaluation
    idx_i = np.repeat(np.arange(N), M * P)
    idx_j = np.tile(np.repeat(np.arange(M), P), N)
    idx_k = np.tile(np.arange(P), N * M)

    U_rows = idx_i * M + idx_j
    V_rows = idx_j * P + idx_k
    W_rows = idx_k * N + idx_i

    def schoolbook():
        U = [[0] * target_rank for _ in range(N * M)]
        V = [[0] * target_rank for _ in range(M * P)]
        W = [[0] * target_rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, R):
        """Exact verification using integer arithmetic."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        vals = np.zeros(N * M * P, dtype=np.int64)
        for r in range(R):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            prod = u_col[U_rows] * v_col[V_rows] * w_col[W_rows]
            vals += prod
        return np.all(vals == 1)

    def compute_sqr_err(U_np, V_np, W_np, R):
        """Compute squared error using NumPy for speed."""
        vals = np.zeros(N * M * P, dtype=np.float64)
        for r in range(R):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            prod = u_col[U_rows] * v_col[V_rows] * w_col[W_rows]
            vals += prod
        err = vals - 1.0
        return np.sum(err * err)

    def local_search_numpy(U, V, W, R, max_steps=50000):
        """
        Local search using NumPy for fast error evaluation.
        Tries to reduce squared error by modifying individual entries.
        """
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        cur_err = compute_sqr_err(U_np, V_np, W_np, R)
        if cur_err < 1e-12:
            if verify(U, V, W, R):
                return U, V, W, 0

        step = 0
        no_improve_count = 0
        while step < max_steps and time.time() < deadline - 5:
            step += 1
            improved = False

            mat_choice = random.random()
            if mat_choice < 0.33:
                mat_np = U_np
                mat_list = U
                n_rows = N * M
            elif mat_choice < 0.66:
                mat_np = V_np
                mat_list = V
                n_rows = M * P
            else:
                mat_np = W_np
                mat_list = W
                n_rows = P * N

            row = random.randrange(n_rows)
            col = random.randrange(R)
            old_val = mat_np[row, col]

            candidates = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5]
            if old_val in candidates:
                candidates = [c for c in candidates if c != old_val]

            best_candidate = None
            best_new_err = cur_err

            for new_val in candidates[:4]:
                mat_np[row, col] = new_val
                if mat_np is U_np:
                    new_err = compute_sqr_err(mat_np, V_np, W_np, R)
                elif mat_np is V_np:
                    new_err = compute_sqr_err(U_np, mat_np, W_np, R)
                else:
                    new_err = compute_sqr_err(U_np, V_np, mat_np, R)
                if new_err < best_new_err - 1e-15:
                    best_new_err = new_err
                    best_candidate = new_val
                mat_np[row, col] = old_val

            if best_candidate is not None:
                mat_np[row, col] = best_candidate
                if mat_np is U_np:
                    mat_list[row][col] = best_candidate
                elif mat_np is V_np:
                    mat_list[row][col] = best_candidate
                else:
                    mat_list[row][col] = best_candidate
                cur_err = best_new_err
                improved = True
                no_improve_count = 0

                if cur_err < 1e-12:
                    U_int = [[round(U_np[i, r] * 2) / 2 for r in range(R)] for i in range(N * M)]
                    V_int = [[round(V_np[i, r] * 2) / 2 for r in range(R)] for i in range(M * P)]
                    W_int = [[round(W_np[i, r] * 2) / 2 for r in range(R)] for i in range(P * N)]
                    if verify(U_int, V_int, W_int, R):
                        return U_int, V_int, W_int, 0
            else:
                no_improve_count += 1
                if no_improve_count > 5000:
                    break

        U_out = [[U_np[i, r] for r in range(R)] for i in range(N * M)]
        V_out = [[V_np[i, r] for r in range(R)] for i in range(M * P)]
        W_out = [[W_np[i, r] for r in range(R)] for i in range(P * N)]
        return U_out, V_out, W_out, int(cur_err)

    # Get the base schoolbook solution
    base_solution = schoolbook()
    U_base = base_solution["U"]
    V_base = base_solution["V"]
    W_base = base_solution["W"]

    # Try to find a rank 119 solution
    # Strategy: Zero out one column and run local search on the remaining 119
    # to see if they can compensate.
    
    # We iterate over a subset of columns to try to remove
    # To save time, we only try a few random columns
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        col_to_remove = random.randrange(target_rank)
        
        # Create new matrices with 119 columns
        U_new = []
        V_new = []
        W_new = []
        
        for i in range(N * M):
            U_new.append([U_base[i][r] for r in range(target_rank) if r != col_to_remove])
        for i in range(M * P):
            V_new.append([V_base[i][r] for r in range(target_rank) if r != col_to_remove])
        for i in range(P * N):
            W_new.append([W_base[i][r] for r in range(target_rank) if r != col_to_remove])
            
        R_new = target_rank - 1
        
        # Run local search to see if we can fix the error
        U_res, V_res, W_res, err = local_search_numpy(U_new, V_new, W_new, R_new, max_steps=10000)
        
        if err == 0:
            if verify(U_res, V_res, W_res, R_new):
                return {"U": U_res, "V": V_res, "W": W_res}

    # If no improvement found, return schoolbook
    return base_solution
# EVOLVE-BLOCK-END
