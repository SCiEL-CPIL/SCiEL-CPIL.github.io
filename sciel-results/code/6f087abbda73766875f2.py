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
                
    # Precompute indices for faster access
    # We need to check: sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = 8 * delta(j,l)
    # (Using scaled integers where value = int/2)
    
    # List of (i, j, l, k, target_scaled)
    # target_scaled is 8 if j==l else 0
    constraints = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    target = 8 if j == l else 0
                    constraints.append((i, j, l, k, target))
                    
    def calc_error(U, V, W, R):
        # U, V, W are lists of lists of integers (scaled by 2)
        # Returns sum of squared errors
        err = 0
        for (i, j, l, k, target) in constraints:
            s = 0
            ui_j = U[i * m + j]
            vl_k = V[l * p + k]
            wk_i = W[k * n + i]
            for r in range(R):
                s += ui_j[r] * vl_k[r] * wk_i[r]
            diff = s - target
            err += diff * diff
        return err

    def local_search(R, time_limit=3.0):
        start_time = time.time()
        
        # Initialize with random small integers
        # Range: -2 to 2 (i.e., -1.0 to 1.0)
        # Maybe wider? -4 to 4.
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Random init
            U = [[random.randint(-4, 4) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-4, 4) for _ in range(R)] for _ in range(m * p)]
            W = [[random.randint(-4, 4) for _ in range(R)] for _ in range(p * n)]
            
            current_err = calc_error(U, V, W, R)
            if current_err == 0:
                return U, V, W
                
            # Local search
            stuck_count = 0
            max_stuck = 100
            best_err = current_err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            
            while time.time() - start_time < time_limit:
                if stuck_count > max_stuck:
                    # Restart from best or random
                    if random.random() < 0.5:
                        U = [row[:] for row in best_U]
                        V = [row[:] for row in best_V]
                        W = [row[:] for row in best_W]
                        current_err = best_err
                    else:
                        U = [[random.randint(-4, 4) for _ in range(R)] for _ in range(n * m)]
                        V = [[random.randint(-4, 4) for _ in range(R)] for _ in range(m * p)]
                        W = [[random.randint(-4, 4) for _ in range(R)] for _ in range(p * n)]
                        current_err = calc_error(U, V, W, R)
                        if current_err < best_err:
                            best_err = current_err
                            best_U = [row[:] for row in U]
                            best_V = [row[:] for row in V]
                            best_W = [row[:] for row in W]
                    stuck_count = 0
                    continue
                    
                # Pick a random variable
                # Total vars: n*m*R + m*p*R + p*n*R
                total_vars = (n * m + m * p + p * n) * R
                idx = random.randint(0, total_vars - 1)
                
                if idx < n * m * R:
                    r = idx % R
                    row = idx // R
                    mat = U
                    old_val = mat[row][r]
                elif idx < (n * m + m * p) * R:
                    idx_v = idx - n * m * R
                    r = idx_v % R
                    row = idx_v // R
                    mat = V
                    old_val = mat[row][r]
                else:
                    idx_w = idx - (n * m + m * p) * R
                    r = idx_w % R
                    row = idx_w // R
                    mat = W
                    old_val = mat[row][r]
                    
                # Try changing by +1 or -1
                # We can try both and pick the best, or just one.
                # Let's try a few small changes
                new_val = old_val + random.choice([-1, 1])
                
                # Apply
                mat[row][r] = new_val
                new_err = calc_error(U, V, W, R)
                
                if new_err < current_err:
                    current_err = new_err
                    stuck_count = 0
                    if current_err < best_err:
                        best_err = current_err
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_err == 0:
                            return best_U, best_V, best_W
                else:
                    # Revert
                    mat[row][r] = old_val
                    stuck_count += 1
                    
            # End while time
            # End attempt
            
        return None

    # Try ranks from low to high
    # Lower bound for 2x3x4 is likely around 18-20.
    # Schoolbook is 24.
    for R in [18, 19, 20, 21, 22, 23]:
        result = local_search(R, time_limit=5.0)
        if result:
            U, V, W = result
            # Convert back to original scale (divide by 2)
            # The task requires integer/half-integer entries.
            # Our internal representation is 2*value.
            # So we need to return values / 2.
            # If value is odd, it's a half-integer.
            # We can return floats or Fractions. The task says "integer/half-integer entries".
            # Returning floats like 0.5, 1.0 is safe.
            U_out = [[u / 2.0 for u in row] for row in U]
            V_out = [[v / 2.0 for v in row] for row in V]
            W_out = [[w / 2.0 for w in row] for row in W]
            
            # Final verification with floats
            def check_decomposition(U, V, W, R):
                for (i, j, l, k, target) in constraints:
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                    target_val = 1.0 if target == 8 else 0.0
                    if abs(s - target_val) > 1e-9:
                        return False
                return True

            if check_decomposition(U_out, V_out, W_out, R):
                return {"U": U_out, "V": V_out, "W": W_out}
            else:
                # Should not happen if local search found 0 error, but just in case
                pass

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
