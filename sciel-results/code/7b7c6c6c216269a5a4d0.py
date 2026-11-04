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
        err = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a], V[b] * W[c])
                    eq_idx = (i * m + j) * p + k
                    err[eq_idx] = s - 1.0
        return err

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
        if np.max(np.abs(err)) < 1e-8:
            if check_integer_half(U, V, W, r):
                return True
        return False

    def search_rank(r, n, m, p, time_limit):
        best_err = float('inf')
        best_sol = None
        
        max_restarts = 500
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            
            curr_err = np.sum(compute_error_numpy(U, V, W, r, n, m, p) ** 2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                    return U_snap, V_snap, W_snap

            no_improve = 0
            max_iters = 2000
            step_size = 0.3
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    U[a, t] += np.random.choice([-1, 1]) * step_size
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    V[b, t] += np.random.choice([-1, 1]) * step_size
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    W[c, t] += np.random.choice([-1, 1]) * step_size
                
                new_err = np.sum(compute_error_numpy(U, V, W, r, n, m, p) ** 2)
                
                if new_err < curr_err:
                    curr_err = new_err
                    no_improve = 0
                    if np.random.rand() < 0.1:
                        step_size *= 1.05
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
                    no_improve += 1
                    if no_improve % 20 == 0:
                        step_size *= 0.9
                    if no_improve > 100:
                        break
                        
            if curr_err < 0.5:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                snap_err = np.sum(compute_error_numpy(U_snap, V_snap, W_snap, r, n, m, p) ** 2)
                if snap_err < 1e-8:
                    if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                        return U_snap, V_snap, W_snap

        return None

    # Try to find a rank 32 decomposition for 3x3x4
    # If found, use it to construct a rank 64 decomposition for 3x3x8
    n4, m4, p4 = 3, 3, 4
    sol_4 = search_rank(32, n4, m4, p4, deadline)
    
    if sol_4:
        U4, V4, W4 = sol_4
        r4 = 32
        # Construct 3x3x8 solution
        # U is (9, 64), V is (24, 64), W is (24, 64)
        # U is [U4, U4]
        # V is [V4_block1, V4_block2] where V4_block1 corresponds to cols 0-3 of B, V4_block2 to cols 4-7
        # But V4 is for 3x3x4, so it has shape (12, 32).
        # We need to map the 12 rows of V4 to the 24 rows of V for 3x3x8.
        # For block 1 (cols 0-3), the rows are 0-11.
        # For block 2 (cols 4-7), the rows are 12-23.
        U8 = np.hstack([U4, U4])
        V8 = np.zeros((24, 64))
        W8 = np.zeros((24, 64))
        
        # Fill V8
        # Block 1: rows 0-11 of V8 correspond to rows 0-11 of V4
        V8[0:12, 0:32] = V4
        # Block 2: rows 12-23 of V8 correspond to rows 0-11 of V4
        V8[12:24, 32:64] = V4
        
        # Fill W8
        # W4 is for 3x3x4, so it has shape (12, 32).
        # For block 1, the output cols are 0-3. The rows of W correspond to (k, i).
        # In 3x3x4, k is 0-3.
        # In 3x3x8, k is 0-7.
        # For block 1, k is 0-3. So rows 0-11 of W8 correspond to rows 0-11 of W4.
        W8[0:12, 0:32] = W4
        # For block 2, k is 4-7. So rows 12-23 of W8 correspond to rows 0-11 of W4.
        W8[12:24, 32:64] = W4
        
        # Verify
        if verify_solution(U8, V8, W8, 64, n, m, p):
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U8]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V8]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W8]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Try rank 35 for 3x3x4
    sol_4 = search_rank(35, n4, m4, p4, deadline)
    if sol_4:
        U4, V4, W4 = sol_4
        r4 = 35
        U8 = np.hstack([U4, U4])
        V8 = np.zeros((24, 70))
        W8 = np.zeros((24, 70))
        V8[0:12, 0:35] = V4
        V8[12:24, 35:70] = V4
        W8[0:12, 0:35] = W4
        W8[12:24, 35:70] = W4
        
        if verify_solution(U8, V8, W8, 70, n, m, p):
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U8]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V8]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W8]
            return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": U_school, "V": V_school, "W": W_school}
