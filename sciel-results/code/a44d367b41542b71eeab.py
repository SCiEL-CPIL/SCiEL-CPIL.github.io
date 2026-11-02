import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        # U: nm x rank, V: mp x rank, W: pn x rank
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(x, tol=0.01):
        # Snap to nearest 0.5
        snapped = np.round(x * 2) / 2.0
        # Check if close
        if np.all(np.abs(x - snapped) < tol):
            return snapped, True
        return x, False

    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Search ranks from lowest known possible (13) up to schoolbook (18)
    # Theoretical lower bound is max(n*m, m*p, n*p) = 9, but 13 is likely min.
    for r in range(13, 18):
        if time.time() > deadline:
            break
            
        # Number of attempts per rank
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random V and W with half-integer entries
            # Using a small set of values to increase density of solutions
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            V = np.array([[random.choice(vals) for _ in range(r)] for _ in range(m * p)])
            W = np.array([[random.choice(vals) for _ in range(r)] for _ in range(p * n)])
            
            # Check if V or W are all zeros in any row? 
            # If a row of V is zero, then for any i,k where V[j*p+k, r]=0 for all r, 
            # the term is 0. But we need sum to be 1.
            # So we need to ensure the system is solvable.
            
            U_real = np.zeros((n * m, r))
            solvable = True
            
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Construct matrix M (p x r)
                    # M[k, r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    M = np.zeros((p, r))
                    for k in range(p):
                        v_row = V[j * p + k]
                        w_row = W[k * n + i]
                        M[k] = v_row * w_row
                    
                    # Solve M @ U_col = 1
                    # U_col is vector of length r
                    try:
                        # Use least squares
                        U_col, residuals, rank_M, s = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_real[u_idx] = U_col
                    except np.linalg.LinAlgError:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if not solvable:
                continue
                
            # Snap U to half-integers
            U_snapped, is_close = snap_to_half_integers(U_real)
            
            if is_close:
                # Convert to lists for verification
                U_list = U_snapped.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no lower rank found, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
