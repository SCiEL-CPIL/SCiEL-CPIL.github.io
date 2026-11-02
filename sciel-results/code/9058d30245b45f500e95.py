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
    
    start_time = time.time()
    time_limit = 115  # Leave 5s for return
    
    # Coefficients to try: Integers and Half-Integers
    # Using a set of small rationals increases the chance of finding a valid solution
    coeffs = [-1, -0.5, 0, 0.5, 1]
    
    # Ranks to try. We go from low to high.
    # Lower bound for 3x4x5 is not 30, but let's start reasonably.
    # If we find a lower rank, we return it immediately.
    ranks_to_try = [35, 40, 45, 50, 55, 58, 59]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Number of restarts per rank
        # We will adjust based on time
        num_restarts = 5000 # High number, we will break on time
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random U and V with coefficients from `coeffs`
            # Using numpy for speed
            U_rand = np.random.choice(coeffs, size=(nm, R)).tolist()
            V_rand = np.random.choice(coeffs, size=(mp, R)).tolist()
            
            # Build the linear system for W
            # A.shape = (n*m*p, R*n*p)
            # b.shape = (n*m*p,)
            # We can construct A more efficiently using broadcasting or pre-computed indices?
            # Given the small size (60x15R), construction is fast enough.
            
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Variable index for W is k * n + i
                        # For each c, coeff is U[i*m+j, c] * V[j*p+k, c]
                        # w_idx = (k*n+i) * R + c
                        # We can vectorize the inner loop over c
                        # u_row = U_rand[i*m+j]
                        # v_row = V_rand[j*p+k]
                        # coeffs_c = np.array(u_row) * np.array(v_row)
                        # A[eq_idx, (k*n+i)*R : (k*n+i)*R + R] = coeffs_c
                        
                        u_row = np.array(U_rand[i * m + j])
                        v_row = np.array(V_rand[j * p + k])
                        coeff_vec = u_row * v_row
                        start_idx = (k * n + i) * R
                        A[eq_idx, start_idx:start_idx + R] = coeff_vec
                        eq_idx += 1
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is close to half-integers
            # We want W to be in {-1, -0.5, 0, 0.5, 1} or similar small rationals
            # Let's check if it's close to k/2
            W_nice = np.round(W_sol * 2) / 2.0
            if np.allclose(W_sol, W_nice, atol=1e-9):
                # Verify with the nice values
                # Convert to lists for the verify function
                W_nice_list = W_nice.tolist()
                
                # Quick verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(R):
                                s += U_rand[i * m + j][c] * V_rand[j * p + k][c] * W_nice_list[k * n + i][c]
                            if abs(s - 1) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                    
                if valid:
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice_list
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
