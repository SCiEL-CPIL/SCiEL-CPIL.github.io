import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110

    def compute_error_numpy(U, V, W, r, n, m, p):
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        T_recon = np.einsum('ijt,jkt,kit->ijk', U_t, V_t, W_t)
        T_target = np.ones((n, m, p))
        err = T_recon - T_target
        return np.sum(err ** 2)

    def check_integer_half(U, V, W, r, tol=1e-9):
        if not np.all(np.abs(2*U - np.round(2*U)) < tol): return False
        if not np.all(np.abs(2*V - np.round(2*V)) < tol): return False
        if not np.all(np.abs(2*W - np.round(2*W)) < tol): return False
        return True

    def snap_to_grid(U, V, W):
        U_snap = np.round(2 * U) / 2.0
        V_snap = np.round(2 * V) / 2.0
        W_snap = np.round(2 * W) / 2.0
        return U_snap, V_snap, W_snap

    def verify_solution(U, V, W, r, n, m, p):
        err = compute_error_numpy(U, V, W, r, n, m, p)
        if err < 1e-8:
            if check_integer_half(U, V, W, r):
                return True
        return False

    def local_discrete_search(U, V, W, r, n, m, p, max_steps=100):
        """
        Attempts to fix a near-miss solution by flipping individual entries
        by +-0.5 or +-1.0 if it reduces the error to 0.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        curr_err = compute_error_numpy(U, V, W, r, n, m, p)
        if curr_err < 1e-8:
            return U, V, W, True

        # We only consider entries that are currently non-zero or near the grid
        # To keep it fast, we just iterate through all entries and try small moves
        # But since we snap first, we are on the grid. We try to move to neighbors.
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
                
            improved = False
            # Try moving a random entry in U
            if np.random.rand() < 0.33:
                idx = np.random.randint(n * m)
                t = np.random.randint(r)
                old_val = U[idx, t]
                # Try +-0.5
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    U[idx, t] = new_val
                    new_err = compute_error_numpy(U, V, W, r, n, m, p)
                    if new_err < curr_err:
                        curr_err = new_err
                        improved = True
                        if new_err < 1e-8:
                            return U, V, W, True
                        break
                    elif new_err < curr_err + 1e-6: # allow small increases to escape local mins? No, strict for now
                        U[idx, t] = new_val
                        curr_err = new_err
                        improved = True
                        break
                    else:
                        U[idx, t] = old_val
            
            # Try moving a random entry in V
            elif np.random.rand() < 0.66:
                idx = np.random.randint(m * p)
                t = np.random.randint(r)
                old_val = V[idx, t]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    V[idx, t] = new_val
                    new_err = compute_error_numpy(U, V, W, r, n, m, p)
                    if new_err < curr_err:
                        curr_err = new_err
                        improved = True
                        if new_err < 1e-8:
                            return U, V, W, True
                        break
                    elif new_err < curr_err + 1e-6:
                        V[idx, t] = new_val
                        curr_err = new_err
                        improved = True
                        break
                    else:
                        V[idx, t] = old_val
            
            # Try moving a random entry in W
            else:
                idx = np.random.randint(p * n)
                t = np.random.randint(r)
                old_val = W[idx, t]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    W[idx, t] = new_val
                    new_err = compute_error_numpy(U, V, W, r, n, m, p)
                    if new_err < curr_err:
                        curr_err = new_err
                        improved = True
                        if new_err < 1e-8:
                            return U, V, W, True
                        break
                    elif new_err < curr_err + 1e-6:
                        W[idx, t] = new_val
                        curr_err = new_err
                        improved = True
                        break
                    else:
                        W[idx, t] = old_val
                        
            if not improved:
                break
                
        return U, V, W, False

    def search_rank(r, n, m, p, time_limit):
        best_err = float('inf')
        best_sol = None
        
        max_restarts = 500
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize with small random values
            # Using a mix of 0 and small numbers helps sparsity
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            curr_err = compute_error_numpy(U, V, W, r, n, m, p)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                    return U_snap, V_snap, W_snap

            # Gradient Descent Step (simple)
            # To speed up convergence, we do a few iterations of coordinate descent
            for _ in range(50):
                if time.time() > time_limit:
                    break
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    # Compute gradient approximation or just try small steps
                    # For speed, we just do random small steps
                    step = np.random.choice([-0.1, 0.1])
                    U[a, t] += step
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    step = np.random.choice([-0.1, 0.1])
                    V[b, t] += step
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    step = np.random.choice([-0.1, 0.1])
                    W[c, t] += step
                
                new_err = compute_error_numpy(U, V, W, r, n, m, p)
                
                if new_err < curr_err:
                    curr_err = new_err
                    if curr_err < 1e-10:
                        U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                        if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                            return U_snap, V_snap, W_snap
                else:
                    if var_type == 0:
                        U[a, t] = old_val
                    elif var_type == 1:
                        V[b, t] = old_val
                    else:
                        W[c, t] = old_val
            
            # Snap and Polish
            if curr_err < 1.0:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                snap_err = compute_error_numpy(U_snap, V_snap, W_snap, r, n, m, p)
                if snap_err < 1e-8:
                    if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                        return U_snap, V_snap, W_snap
                else:
                    # Try to fix it locally
                    U_fix, V_fix, W_fix, success = local_discrete_search(U_snap, V_snap, W_snap, r, n, m, p, max_steps=20)
                    if success:
                        return U_fix, V_fix, W_fix

        return None

    # Try to find a rank significantly lower than 56
    # 3x3x8 matrix multiplication tensor rank is at most 72.
    # Known bounds suggest it could be much lower. Let's try 40, 45, 50.
    ranks_to_try = [40, 45, 50, 55]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        sol = search_rank(r, n, m, p, deadline)
        if sol:
            U, V, W = sol
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
