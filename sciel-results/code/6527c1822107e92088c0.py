"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    np.random.seed(42)
    
    # Strategy: fix U and V, solve for W using linear algebra
    for target_rank in range(24, 96, 1):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random U and V
            U_rand = np.random.choice([-1.0, 1.0], size=(n * m, target_rank))
            V_rand = np.random.choice([-1.0, 1.0], size=(m * p, target_rank))
            
            # Solve for W
            W_rand = np.zeros((p * n, target_rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build m x target_rank matrix
                    A_block = np.zeros((m, target_rank))
                    for j in range(m):
                        A_block[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                    
                    # Solve A_block * w = ones(m)
                    try:
                        w_sol, residuals, rank_val, sv = np.linalg.lstsq(A_block, np.ones(m), rcond=None)
                        W_rand[k * n + i, :] = w_sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify
            if verify(U_rand, V_rand, W_rand):
                best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
