import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor M
    # M[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    M = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1
    
    # Indices where M is 1
    nz_indices = np.argwhere(M == 1)
    
    # Schoolbook solution
    U_school = np.zeros((n*m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=int)
    col = 0
    for idx in nz_indices:
        a, b, c = idx
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1
        col += 1
        
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = 110  # 110 seconds to be safe
    
    # Try to find a solution with rank R < 105
    # We know R <= 87, so we try down to 80
    for R in range(rank_schoolbook - 1, 79, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with small random integers
            # Using -1, 0, 1 is a good start for tensor decompositions
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
            
            # Local search
            # We want to minimize || M_calc - M ||^2
            # M_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # Precompute partial sums to speed up error calculation?
            # For small R, direct calculation is fine.
            
            def calc_error(U, V, W):
                err = 0.0
                for a, b, c in nz_indices:
                    s = 0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    err += (s - 1) ** 2
                # Check zero constraints (sampled)
                # If we don't check all zeros, we might get a false positive
                # But we can check all zeros at the end if error on non-zeros is 0
                # For speed, we only check non-zeros during search
                return err
            
            current_err = calc_error(U, V, W)
            
            # Local search loop
            while time.time() < deadline - 1:
                # Pick a random variable
                mat = np.random.choice([U, V, W])
                row = np.random.randint(mat.shape[0])
                col = np.random.randint(mat.shape[1])
                
                old_val = mat[row, col]
                # Try changing to old_val +/- 1 or 0
                new_val = old_val + np.random.choice([-1, 0, 1])
                
                mat[row, col] = new_val
                new_err = calc_error(U, V, W)
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err == 0:
                        # Verify against all zeros
                        valid = True
                        for a in range(n*m):
                            for b in range(m*p):
                                for c in range(p*n):
                                    if M[a,b,c] == 0:
                                        s = 0
                                        for r in range(R):
                                            s += U[a, r] * V[b, r] * W[c, r]
                                        if s != 0:
                                            valid = False
                                            break
                            if not valid: break
                        if valid:
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            best_rank = R
                            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                else:
                    mat[row, col] = old_val
                    
            # End local search for this restart
            
    # If no better solution found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
