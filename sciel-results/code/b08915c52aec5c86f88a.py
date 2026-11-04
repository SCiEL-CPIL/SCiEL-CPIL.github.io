import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
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
    
    def check_decomposition(U, V, W, R):
        # U: n*m x R, V: m*p x R, W: p*n x R
        # Check: sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] == delta(j,l)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                        target = 1 if j == l else 0
                        if s != target:
                            return False
        return True

    def compute_error(U, V, W, R):
        # U: (n*m, R) -> reshape (n, m, R)
        # V: (m*p, R) -> reshape (m, p, R)
        # W: (p*n, R) -> reshape (p, n, R)
        U_reshaped = np.array(U).reshape(n, m, R)
        V_reshaped = np.array(V).reshape(m, p, R)
        W_reshaped = np.array(W).reshape(p, n, R)
        
        error = 0.0
        for i in range(n):
            for k in range(p):
                u = U_reshaped[i, :, :] # (m, R)
                v = V_reshaped[:, k, :] # (m, R)
                w = W_reshaped[k, i, :] # (R,)
                
                # M = sum_r w[r] * u[:, r] * v[:, r]
                # M = (u * w) @ v.T
                M = (u * w) @ v.T
                # Target is I_m
                error += np.sum((M - np.eye(m))**2)
        return error

    def try_find_decomposition(R, time_limit=10):
        start_time = time.time()
        # Random initialization
        U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(p * n)]
        
        current_error = compute_error(U, V, W, R)
        
        # Hill Climbing
        while time.time() - start_time < time_limit:
            if current_error == 0:
                if check_decomposition(U, V, W, R):
                    return {"U": U, "V": V, "W": W}
                else:
                    # Should not happen if error is 0 and integers, but safety check
                    return None
            
            # Pick a random variable to change
            var_type = random.choice(['U', 'V', 'W'])
            if var_type == 'U':
                row = random.randint(0, n * m - 1)
                col = random.randint(0, R - 1)
                old_val = U[row][col]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                U[row][col] = new_val
                new_error = compute_error(U, V, W, R)
                if new_error < current_error:
                    current_error = new_error
                else:
                    U[row][col] = old_val
            elif var_type == 'V':
                row = random.randint(0, m * p - 1)
                col = random.randint(0, R - 1)
                old_val = V[row][col]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                V[row][col] = new_val
                new_error = compute_error(U, V, W, R)
                if new_error < current_error:
                    current_error = new_error
                else:
                    V[row][col] = old_val
            else: # W
                row = random.randint(0, p * n - 1)
                col = random.randint(0, R - 1)
                old_val = W[row][col]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                W[row][col] = new_val
                new_error = compute_error(U, V, W, R)
                if new_error < current_error:
                    current_error = new_error
                else:
                    W[row][col] = old_val

            # Occasional restart if stuck?
            # For now, just keep climbing.
            
        return None

    # Try ranks from 20 to 23
    for R in [20, 21, 22, 23]:
        result = try_find_decomposition(R, time_limit=10)
        if result:
            return result

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
