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
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Heuristic search for lower rank
    # We try to find a decomposition with rank R < rank_school
    # Strategy: Randomize U, V, solve for W, check if W is "nice" (integer/half-integer)
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Try ranks from lower bounds up to schoolbook - 1
    # We will iterate ranks in increasing order to find the minimal one.
    
    def is_half_integer(matrix, tol=1e-9):
        for row in matrix:
            for val in row:
                # Check if val is close to k/2
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
        # Convert to list of lists of float (or int if integer)
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        # Check if sum_c U[i,j,c] V[j,k,c] W[i,k,c] == 1 for all i,j,k
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Search loop
    # We try a few ranks. If we find a valid one, we return it.
    # To save time, we don't iterate all ranks from 1 to 60.
    # We try specific ranks known to be promising or just a range.
    # Let's try ranks: 50, 55.
    # If we find one, we return it.
    
    ranks_to_try = [55, 50]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Number of restarts per rank
        num_restarts = 500
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random values from {-1, 0, 1}
            # Use sparse initialization to increase chance of simple solutions
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
            
            # Build the linear system for W
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U_rand[i * m + j][c] * V_rand[j * p + k][c]
                            w_idx = (k * n + i) * R + c
                            A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is close to half-integers
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    return {"U": best_U, "V": best_V, "W": best_W}

        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
