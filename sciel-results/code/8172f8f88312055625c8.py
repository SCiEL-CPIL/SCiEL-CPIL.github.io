import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def check_solution(U, V, W, n, m, p):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_solution = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Try to find better solution
    for R in range(30, rank_school):
        if time.time() - start_time > 110:
            break
            
        for _ in range(50):
            if time.time() - start_time > 110:
                break
                
            # Random sparse U, V
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Solve for W
            W = np.zeros((p * n, R))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build system for W[k*n+i, :]
                    # Equations: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for j in 0..m-1
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A x = 1
                    # Least squares
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if x is close to half-integers
                    x_rounded = np.round(x * 2) / 2
                    if np.allclose(x, x_rounded, atol=1e-6):
                        W[k * n + i, :] = x_rounded
                    else:
                        # Allow some tolerance? Or fail?
                        # If we fail, we can try to adjust?
                        # For now, if not close, mark as invalid for this trial
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify full solution
                if check_solution(U, V, W, n, m, p):
                    best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R
                    break # Found a solution for this R, try smaller R? No, we are iterating R up.
                    # Actually, we want the MINIMAL rank. So we should iterate R from low to high.
                    # If we find one, we can break and return it?
                    # But maybe a smaller R exists? We are iterating 30, 31, ...
                    # So if we find one at R=35, we should continue to check if R=34 works?
                    # But we are iterating R in range(30, 42).
                    # So we should break the inner loop and continue the outer loop?
                    # No, if we find a solution at R, we should record it and continue to see if a smaller R works?
                    # But we are increasing R. So the first one found is the smallest in this range?
                    # No, we start at 30. If 30 fails, we try 31.
                    # So if we find one at 35, we should stop and return it?
                    # Yes, because we are looking for rank-minimal.
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_solution
