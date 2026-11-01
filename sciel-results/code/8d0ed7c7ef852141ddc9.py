import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 19
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find a decomposition of rank 19
    # Variables: U (n*m, r), V (m*p, r), W (p*n, r)
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # We can fix U and V to be sparse (e.g., 0 or 1) and solve for W?
    # Or use random search.
    
    # Since manual hardcoding is risky if I don't have the exact values, 
    # and random search might not converge to exact half-integers quickly,
    # I will implement a robust fallback to the schoolbook method (rank 24)
    # and attempt a quick optimization for rank 19.
    
    # However, without a guaranteed fast solver, the schoolbook is the safe bet for validity.
    # But the task asks for "rank-minimal". 
    # If I can't find < 24, I return 24.
    
    # Let's try a specific known pattern or a very fast random search.
    # Given the small size, we can try to solve for W given random U, V.
    # For each (i, k), we have equations for all j.
    # Let X_c = U[:, c] V[:, c] (vector of length m*p? No, U is n*m, V is m*p).
    # The term is sum_j U[i, j, c] V[j, k, c]. Let's call this M_{i, k, c}.
    # Then we need sum_c M_{i, k, c} W_{k, i, c} = 1.
    # This is a system of n*p equations for W? No, W has p*n elements.
    # For each (k, i), we have one equation: sum_c M_{i, k, c} W_{k, i, c} = 1.
    # But M depends on c. W_{k, i, c} is the variable.
    # So for each pair (k, i), we have 1 equation with r variables W_{k, i, c}.
    # This is underdetermined. We can pick r-1 variables freely and solve for the last one?
    # Yes! If we fix U and V, we can determine W for each (k, i) independently, provided the coefficient of the chosen variable is non-zero.
    
    # Algorithm:
    # 1. Randomly initialize U and V with small integers (e.g., -1, 0, 1).
    # 2. For each (k, i) in range(p*n):
    #    a. Compute coeffs[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    #    b. We need sum_c coeffs[c] * W[k*n+i, c] = 1.
    #    c. Pick a column c_last where coeffs[c_last] != 0.
    #    d. Set W[k*n+i, c] for c != c_last to random small integers (e.g., 0 or 1).
    #    e. Solve for W[k*n+i, c_last]:
    #       W[k*n+i, c_last] = (1 - sum_{c != c_last} coeffs[c] * W[k*n+i, c]) / coeffs[c_last]
    #    f. Check if W[k*n+i, c_last] is a half-integer (or integer).
    #       If not, this random U, V doesn't work easily. Retry with different U, V or different W choices.
    #    g. If it is a half-integer, accept.
    # 3. If all (k, i) pairs yield valid half-integer W, we have a solution.
    # 4. If time runs out, return schoolbook.
    
    # This search space is small enough to try many random U, V.
    
    import random
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find rank 19
    r = 19
    attempts = 0
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        # Random U, V with entries in {-1, 0, 1}
        U_try = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V_try = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        
        W_try = [[0] * r for _ in range(p * n)]
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Calculate coeffs
                coeffs = [0.0] * r
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U_try[i * m + j][c] * V_try[j * p + k][c]
                    coeffs[c] = s
                
                # Find a pivot
                pivot = -1
                for c in range(r):
                    if coeffs[c] != 0:
                        pivot = c
                        break
                
                if pivot == -1:
                    valid = False
                    break
                
                # Randomly assign other W values to small integers
                # To keep numbers small, use 0 or 1
                for c in range(r):
                    if c == pivot:
                        continue
                    W_try[k * n + i][c] = random.choice([0, 1, -1])
                
                # Solve for pivot
                sum_other = sum(coeffs[c] * W_try[k * n + i][c] for c in range(r) if c != pivot)
                val = (1.0 - sum_other) / coeffs[pivot]
                
                # Check if val is half-integer
                # val * 2 should be integer
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    valid = False
                    break
                
                W_try[k * n + i][pivot] = val
        
        if valid:
            # Verify exactness
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_try[i * m + j][c] * V_try[j * p + k][c] * W_try[k * n + i][c]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if ok:
                best_rank = r
                best_U, best_V, best_W = U_try, V_try, W_try
                break # Found rank 19, stop searching for better (unlikely to find <19 quickly)
        
        if time.time() > deadline:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
