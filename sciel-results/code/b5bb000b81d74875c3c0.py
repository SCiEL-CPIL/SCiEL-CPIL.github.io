import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    # Build target tensor
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc, Target, atol=tol)

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def solve_for_W(U, V):
        """Given U and V, solve for W using least squares."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        R = U_arr.shape[1]
        
        W = np.zeros((dim_W, R))
        residual = 0.0
        
        # M[alpha, beta, r] = U[alpha, r] * V[beta, r]
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        
        for gamma in range(dim_W):
            A = M.reshape(dim_U * dim_V, R)
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            # Use lstsq to find W[gamma, :]
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            r = b - A @ x
            residual += np.sum(r**2)
        
        return W, residual

    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # Schoolbook solution (rank 75)
    best_result = None
    best_rank = n * m * p  # 75
    
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 25
    # Upper bound is current best
    for R in range(25, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Number of restarts decreases as R increases (higher R is easier)
        # But we start from low R which is hard.
        n_restarts = 20 if R < 40 else 10
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with sparse half-integers
            # Most entries should be 0
            U_init = np.zeros((dim_U, R))
            V_init = np.zeros((dim_V, R))
            
            # Fill with random half-integers with some probability
            # Probability of non-zero
            p_nonzero = 0.3
            
            for i in range(dim_U):
                for r in range(R):
                    if random.random() < p_nonzero:
                        U_init[i, r] = random.choice([-1, -0.5, 0.5, 1])
                        
            for i in range(dim_V):
                for r in range(R):
                    if random.random() < p_nonzero:
                        V_init[i, r] = random.choice([-1, -0.5, 0.5, 1])
            
            # Initial solve
            W_check, res_check = solve_for_W(U_init, V_init)
            W_snap = snap_to_half_int(W_check)
            
            if verify(U_init, V_init, W_snap):
                best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_snap.tolist()}
                best_rank = R
                found = True
                break
            
            # Local search
            # We will try to improve the residual by flipping entries in U or V
            # and re-solving W
            current_res = res_check
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            
            max_steps = 2000
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random variable to change
                # 50% chance U, 50% chance V
                if random.random() < 0.5:
                    idx_u = random.randint(0, dim_U - 1)
                    idx_r = random.randint(0, R - 1)
                    old_val = U_curr[idx_u, idx_r]
                    # Generate new value
                    # If old was 0, try small non-zero
                    # If old was non-zero, try 0 or flip
                    if old_val == 0:
                        new_val = random.choice([-0.5, 0.5])
                    else:
                        new_val = random.choice([0, -old_val, old_val + 0.5, old_val - 0.5])
                        # Ensure it's a half integer
                        new_val = snap_to_half_int(new_val)
                        
                    U_curr[idx_u, idx_r] = new_val
                else:
                    idx_v = random.randint(0, dim_V - 1)
                    idx_r = random.randint(0, R - 1)
                    old_val = V_curr[idx_v, idx_r]
                    if old_val == 0:
                        new_val = random.choice([-0.5, 0.5])
                    else:
                        new_val = random.choice([0, -old_val, old_val + 0.5, old_val - 0.5])
                        new_val = snap_to_half_int(new_val)
                        
                    V_curr[idx_v, idx_r] = new_val

                # Solve for W
                W_new, res_new = solve_for_W(U_curr, V_curr)
                W_new_snap = snap_to_half_int(W_new)
                
                # Check if we found a solution
                if verify(U_curr, V_curr, W_new_snap):
                    best_result = {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_new_snap.tolist()}
                    best_rank = R
                    found = True
                    break
                
                # Accept if residual decreased
                if res_new < current_res:
                    current_res = res_new
                    # Keep the change
                else:
                    # Revert
                    if random.random() < 0.5:
                        U_curr[idx_u, idx_r] = old_val
                    else:
                        V_curr[idx_v, idx_r] = old_val

            if found:
                break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
