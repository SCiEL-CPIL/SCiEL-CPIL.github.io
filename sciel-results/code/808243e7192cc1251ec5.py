import time
import numpy as np
import random

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared residuals."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _local_search(U, V, W, n, m, p, max_iters=20000):
    """
    Perform local search with incremental error update.
    """
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute current sums S[i,j,k]
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                
    current_err = np.sum((S - 1.0)**2)
    
    candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        if current_err < 1e-12:
            break
            
        # Randomly select a matrix, row, col
        mat_choice = random.randint(0, 2)
        
        if mat_choice == 0:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, r - 1)
            i = row // m
            j = row % m
            old_val = U[row, col]
            
            # Try changing U[i, m+j, col]
            # Affects S[i, j, k] for all k
            best_val = old_val
            best_err = current_err
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Calculate delta_err
                d_err = 0.0
                for k in range(p):
                    v_val = V[j * p + k, col]
                    w_val = W[k * n + i, col]
                    if v_val == 0 or w_val == 0:
                        continue
                    delta_s = delta * v_val * w_val
                    s_old = S[i, j, k]
                    s_new = s_old + delta_s
                    d_err += (s_new - 1.0)**2 - (s_old - 1.0)**2
                new_err = current_err + d_err
                if new_err < best_err:
                    best_err = new_err
                    best_val = new_val
            
            if best_val != old_val:
                delta = best_val - old_val
                U[row, col] = best_val
                # Update S
                for k in range(p):
                    v_val = V[j * p + k, col]
                    w_val = W[k * n + i, col]
                    if v_val != 0 and w_val != 0:
                        S[i, j, k] += delta * v_val * w_val
                current_err = best_err
                
        elif mat_choice == 1:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, r - 1)
            j = row // p
            k = row % p
            old_val = V[row, col]
            
            # Affects S[i, j, k] for all i
            best_val = old_val
            best_err = current_err
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                d_err = 0.0
                for i in range(n):
                    u_val = U[i * m + j, col]
                    w_val = W[k * n + i, col]
                    if u_val == 0 or w_val == 0:
                        continue
                    delta_s = delta * u_val * w_val
                    s_old = S[i, j, k]
                    s_new = s_old + delta_s
                    d_err += (s_new - 1.0)**2 - (s_old - 1.0)**2
                new_err = current_err + d_err
                if new_err < best_err:
                    best_err = new_err
                    best_val = new_val
            
            if best_val != old_val:
                delta = best_val - old_val
                V[row, col] = best_val
                for i in range(n):
                    u_val = U[i * m + j, col]
                    w_val = W[k * n + i, col]
                    if u_val != 0 and w_val != 0:
                        S[i, j, k] += delta * u_val * w_val
                current_err = best_err
                
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, r - 1)
            k = row // n
            i = row % n
            old_val = W[row, col]
            
            # Affects S[i, j, k] for all j
            best_val = old_val
            best_err = current_err
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                d_err = 0.0
                for j in range(m):
                    u_val = U[i * m + j, col]
                    v_val = V[j * p + k, col]
                    if u_val == 0 or v_val == 0:
                        continue
                    delta_s = delta * u_val * v_val
                    s_old = S[i, j, k]
                    s_new = s_old + delta_s
                    d_err += (s_new - 1.0)**2 - (s_old - 1.0)**2
                new_err = current_err + d_err
                if new_err < best_err:
                    best_err = new_err
                    best_val = new_val
            
            if best_val != old_val:
                delta = best_val - old_val
                W[row, col] = best_val
                for j in range(m):
                    u_val = U[i * m + j, col]
                    v_val = V[j * p + k, col]
                    if u_val != 0 and v_val != 0:
                        S[i, j, k] += delta * u_val * v_val
                current_err = best_err
    
    return U, V, W, current_err

def _try_rank_r(n, m, p, r, max_restarts=20):
    """
    Try to find a rank-r decomposition.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize with random sparse half-integers
        # Bias towards 0
        probs = [0.1, 0.1, 0.6, 0.1, 0.1] # -1, -0.5, 0, 0.5, 1
        U0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, r), p=probs)
        V0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, r), p=probs)
        W0 = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, r), p=probs)
        
        U_sol, V_sol, W_sol, err = _local_search(U0, V0, W0, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            
            if err < 1e-12:
                return U_sol, V_sol, W_sol
                
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try rank 18
    if time.time() - START_TIME < TIME_LIMIT - 20:
        result = _try_rank_r(n, m, p, 18, max_restarts=20)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = 18
    
    # Try rank 19 if 18 failed and time permits
    if best_rank > 19 and time.time() - START_TIME < TIME_LIMIT - 15:
        result = _try_rank_r(n, m, p, 19, max_restarts=10)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = 19
            
    # Verify
    ok = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = n * m * p

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
