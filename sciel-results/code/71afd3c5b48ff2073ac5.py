import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
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
    
    # Try to find a lower rank decomposition
    # We will try ranks from 60 down to 50
    # If we find one, we return it.
    
    # The linear system for W given U, V:
    # For each j, i, k: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    # Let x_r = W[r, k*n+i]
    # This is a linear system A x = b
    # A has shape (m*n*p, R*n*p)
    # Each row corresponds to (j, i, k)
    # Each column corresponds to (r, i', k')
    # A[(j,i,k), (r, i', k')] = U[r, i*m+j] * V[r, j*p+k] if (i', k') == (i, k) else 0
    # Wait, W[r, k*n+i] is the coefficient for the term (i,k) in rank r.
    # So for a fixed (j,i,k), the equation is:
    # sum_{r} U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    # The variables are W[r, k*n+i] for all r and all (i', k').
    # But for a specific (j,i,k), only the variables W[r, k*n+i] appear.
    # So the system is block diagonal with respect to (i,k)?
    # No, W[r, k*n+i] is the same variable for all j.
    # So for a fixed (i,k), we have m equations (one for each j) and R variables (W[r, k*n+i] for r=1..R).
    # Equation for j: sum_{r} U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    # This is a system of m equations in R unknowns for each (i,k).
    # Since m=4 and R>=50, the system is underdetermined.
    # We can solve it using least squares or by picking a basis.
    
    # To ensure integer/half-integer entries, we can try to find a solution with small integers.
    # We can use a random sparse U, V and solve for W.
    
    best_result = None
    best_rank = rank_school
    
    for target_rank in [60, 58, 56, 54, 52, 50]:
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for trial in range(20):
            if time.time() > deadline:
                break
                
            R = target_rank
            # Initialize U, V with random small integers
            # Use numpy for speed
            U_np = np.zeros((R, n * m), dtype=int)
            V_np = np.zeros((R, m * p), dtype=int)
            
            # Random sparse initialization
            for r in range(R):
                # Pick a few non-zero entries
                num_nonzero_u = np.random.randint(2, 5)
                idx_u = np.random.choice(n * m, num_nonzero_u, replace=False)
                U_np[r, idx_u] = np.random.choice([-1, 1], num_nonzero_u)
                
                num_nonzero_v = np.random.randint(2, 5)
                idx_v = np.random.choice(m * p, num_nonzero_v, replace=False)
                V_np[r, idx_v] = np.random.choice([-1, 1], num_nonzero_v)
            
            # Solve for W
            # For each (i, k), we have a system:
            # A_ik * w_ik = 1
            # where A_ik is m x R
            # A_ik[j, r] = U_np[r, i*m+j] * V_np[r, j*p+k]
            
            W_np = np.zeros((R, p * n), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A_ik
                    A_ik = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A_ik[j, r] = U_np[r, i * m + j] * V_np[r, j * p + k]
                    
                    # Solve A_ik * w_ik = ones(m)
                    # Use least squares
                    try:
                        w_ik, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if solution is close to half-integers
                    # w_ik should be close to multiples of 0.5
                    # Let's check the distance to the nearest half-integer
                    rounded_w = np.round(w_ik * 2) / 2
                    if not np.allclose(w_ik, rounded_w, atol=1e-6):
                        # Try to see if a different solution in the null space works?
                        # For now, just reject
                        valid = False
                        break
                    
                    W_np[:, k * n + i] = rounded_w
            
            if not valid:
                continue
            
            # Verify the decomposition
            # Reconstruct and check
            U_list = U_np.tolist()
            V_list = V_np.tolist()
            W_list = W_np.tolist()
            
            # Check exactness
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_list[r][i * m + j] * V_list[r][j * p + k] * W_list[r][k * n + i]
                        if abs(val - 1) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
            
            if is_exact:
                return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
