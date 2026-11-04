import numpy as np

def construct():
    n, m, p = 2, 3, 7
    
    # Schoolbook rank
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
    
    # Try to find lower rank
    # Variables: W has n*p*r elements
    # Equations: m^2 * n * p elements
    # For each j, j' in range(m), i in range(n), k in range(p):
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = 1 if j==j' else 0
    
    import random
    import time
    
    start_time = time.time()
    max_time = 100 # Leave buffer for 120s limit
    
    # Try ranks from theoretical min to schoolbook - 1
    # Theoretical min is not known, but we can start from something reasonable.
    # Let's try from 20 to 41.
    
    for r in range(20, rank_school):
        if time.time() - start_time > max_time:
            break
            
        num_trials = 100
        for trial in range(num_trials):
            if time.time() - start_time > max_time:
                break
            
            # Random U and V with integer entries
            # Using small integers to keep numbers manageable
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            # Build linear system A W_flat = b
            # A is (m^2 * n * p) x (n * p * r)
            # b is (m^2 * n * p)
            
            num_eqs = m * m * n * p
            num_vars = n * p * r
            
            A = np.zeros((num_eqs, num_vars))
            b = np.zeros(num_eqs)
            
            idx = 0
            for jj in range(m):
                for jj_prime in range(m):
                    target = 1.0 if jj == jj_prime else 0.0
                    for i in range(n):
                        for k in range(p):
                            # Equation: sum_c W[k*n+i, c] * U[i*m+jj, c] * V[jj_prime*p+k, c] = target
                            # Variables are W[k*n+i, c]
                            # Index in W_flat for W[k*n+i, c] is (k*n+i)*r + c
                            
                            u_row = U[i * m + jj]
                            v_row = V[jj_prime * p + k]
                            
                            # The coefficient for W[k*n+i, c] is u_row[c] * v_row[c]
                            # We add this to the row idx of A, at columns corresponding to W[k*n+i, :]
                            
                            base_col = (k * n + i) * r
                            A[idx, base_col:base_col+r] = u_row * v_row
                            b[idx] = target
                            idx += 1
            
            # Solve A W_flat = b
            # Use least squares or direct solve if square/invertible
            # Since we want exact decomposition, we check residual later.
            try:
                # np.linalg.lstsq gives a solution that minimizes error
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is close to zero residual
                # Also check if W_flat entries are close to integers or half-integers?
                # The problem says "integer/half-integer entries".
                # Let's round to nearest 0.5 and verify.
                
                W_rounded_flat = np.round(W_flat * 2) / 2.0
                
                # Verify the rounded solution
                # Reconstruct W
                W = W_rounded_flat.reshape((n * p, r))
                
                # Check condition
                valid = True
                for jj in range(m):
                    for jj_prime in range(m):
                        target = 1.0 if jj == jj_prime else 0.0
                        for i in range(n):
                            for k in range(p):
                                # sum_c W[k*n+i, c] * U[i*m+jj, c] * V[jj_prime*p+k, c]
                                val = 0.0
                                for c in range(r):
                                    val += W[k * n + i, c] * U[i * m + jj, c] * V[jj_prime * p + k, c]
                                if abs(val - target) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists for output
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_rounded_flat.reshape((n * p, r)).tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
            except np.linalg.LinAlgError:
                continue
        
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
