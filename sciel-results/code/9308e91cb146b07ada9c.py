import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def is_half_integer(matrix, tol=1e-9):
        # Check if all elements are close to k/2
        # Return rounded matrix if valid, else None
        rounded = np.round(matrix * 2) / 2
        if np.all(np.abs(matrix - rounded) < tol):
            # Convert to list of lists of int/float
            res = []
            for row in rounded:
                res_row = []
                for val in row:
                    if val == int(val):
                        res_row.append(int(val))
                    else:
                        res_row.append(val)
                res.append(res_row)
            return res
        return None

    def verify_decomposition(U, V, W, rank):
        # U, V, W are lists of lists
        # Convert to numpy arrays for fast dot products
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    start_time = time.time()
    time_limit = 110
    
    best_rank = rank_school
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # We want to find a rank < 60.
    # Let's try ranks from 35 to 59.
    ranks_to_try = list(range(35, 60))
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
        
        # Number of restarts. We have ~110s / 25 ranks = 4.4s per rank.
        # Let's do 500 restarts per rank if time permits.
        num_restarts = 500
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random U, V with entries in {-1, 0, 1}
            # Sparse: 50% zeros
            U_rand = np.random.choice([-1, 0, 1], size=(nm, R), p=[0.25, 0.5, 0.25])
            V_rand = np.random.choice([-1, 0, 1], size=(mp, R), p=[0.25, 0.5, 0.25])
            
            # Build A and b
            # A is (n*m*p, R*n*p)
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        coeffs = U_rand[i * m + j] * V_rand[j * p + k]
                        start_col = (k * n + i) * R
                        end_col = start_col + R
                        A[eq_idx, start_col:end_col] = coeffs
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            # Check for half-integers
            W_nice = is_half_integer(W_sol)
            if W_nice is not None:
                # Verify with exact arithmetic (using floats is fine for 1e-9)
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                if verify_decomposition(U_list, V_list, W_nice, R):
                    best_rank = R
                    best_U = U_list
                    best_V = V_list
                    best_W = W_nice
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
