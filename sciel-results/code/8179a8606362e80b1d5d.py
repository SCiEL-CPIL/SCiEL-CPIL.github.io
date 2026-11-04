import time
import numpy as np
from fractions import Fraction

def construct():
    t_start = time.time()
    limit = t_start + 110.0
    
    n, m, p = 2, 3, 8
    
    def _schoolbook():
        rb = n * m * p
        U_f = [[0] * rb for _ in range(n * m)]
        V_f = [[0] * rb for _ in range(m * p)]
        W_f = [[0] * rb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_f[i * m + a][c] = 1
                    V_f[a * p + k][c] = 1
                    W_f[k * n + i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    def _validate(U, V, W, rank):
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    for r in range(rank):
                        acc += U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
                    if acc != 1:
                        return False
        return True

    # Search for lower rank
    # Try R from 24 to 43
    import random
    random.seed(42)
    
    best_result = None
    
    for R in range(24, 44):
        if time.time() > limit - 5:
            break
            
        # Try a few random instances for this R
        trials = 100
        for _ in range(trials):
            if time.time() > limit - 5:
                break
                
            # Generate random sparse U, V
            # Each row of U (size m) and V (size p) should have a few non-zeros
            # We want the system M W = 1 to be solvable.
            # M is m x R. We need rank(M) >= m.
            
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            
            # Fill U and V with random -1, 0, 1
            # Ensure each row has at least 2 non-zeros to allow solving
            for idx in range(n * m):
                # Pick 2-3 random columns
                cols = random.sample(range(R), min(3, R))
                for c in cols:
                    U[idx][c] = random.choice([-1, 1])
                    
            for idx in range(m * p):
                cols = random.sample(range(R), min(3, R))
                for c in cols:
                    V[idx][c] = random.choice([-1, 1])
            
            # Solve for W
            W = [[0] * R for _ in range(p * n)]
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build system M W_ki = 1
                    # M is m x R
                    M = np.zeros((m, R))
                    for a in range(m):
                        for r in range(R):
                            M[a, r] = U[i*m+a][r] * V[a*p+k][r]
                    
                    # Solve M x = 1
                    try:
                        # Use least squares or direct solve if square?
                        # M is m x R. m=3, R>=24. Underdetermined.
                        # We want an integer solution.
                        # We can use np.linalg.lstsq and check if it's close to integer.
                        # Or we can iterate through possible integer solutions?
                        # Since R is large, lstsq is better.
                        sol, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        
                        # Check if sol is close to integer
                        # If not, try to adjust?
                        # For now, just check if it's close.
                        diff = np.abs(sol - np.round(sol))
                        if np.max(diff) < 1e-6:
                            W_sol = np.round(sol).astype(int)
                            # Verify
                            if np.allclose(M.dot(W_sol), np.ones(m)):
                                for r in range(R):
                                    W[k*n+i][r] = int(W_sol[r])
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                        
                if not valid:
                    break
            
            if valid:
                if _validate(U, V, W, R):
                    return {"U": U, "V": V, "W": W}
    
    # If no lower rank found, return the known rank 44 solution
    # Re-use the existing rank 44 logic or just return schoolbook?
    # The current program has a rank 44 solution. We should keep it.
    
    # Refactor the rank 44 solution into a helper or just copy it
    # To save space and time, we just return the rank 44 solution if search fails.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    su = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    sv = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    sw = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    for b_idx in range(4):
        k_offset = b_idx * 2
        for s_idx in range(7):
            for i in range(2):
                for a in range(2):
                    val = su[s_idx][i * 2 + a]
                    if val:
                        U[i * m + a][col] += val
            for a in range(2):
                for k in range(2):
                    val = sv[s_idx][a * 2 + k]
                    if val:
                        V[a * p + (k_offset + k)][col] += val
            for k in range(2):
                for i in range(2):
                    val = sw[s_idx][k * 2 + i]
                    if val:
                        W[(k_offset + k) * n + i][col] += val
            col += 1
            
    for i in range(2):
        for k in range(8):
            u_row = i * m + 2
            v_row = 2 * p + k
            w_row = k * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
