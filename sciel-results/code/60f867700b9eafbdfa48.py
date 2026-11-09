import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school), dtype=int)
    V_school = np.zeros((m * p, R_school), dtype=int)
    W_school = np.zeros((p * n, R_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def is_valid(U, V, W):
        for i in range(n):
            U_i = U[i*m:(i+1)*m, :]
            for k in range(p):
                V_k = V[np.arange(m)*p + k, :]
                W_ki = W[k*n + i, :]
                M = (U_i * W_ki) @ V_k.T
                if not np.allclose(M, np.eye(m)):
                    return False
        return True

    def error(U, V, W):
        err = 0
        for i in range(n):
            U_i = U[i*m:(i+1)*m, :]
            for k in range(p):
                V_k = V[np.arange(m)*p + k, :]
                W_ki = W[k*n + i, :]
                M = (U_i * W_ki) @ V_k.T
                err += np.sum((M - np.eye(m))**2)
        return err

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = start_time + 145
    
    # Try ranks from lower bound to best_rank - 1
    for R in range(28, best_rank):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        # Increase attempts for lower ranks
        max_attempts = 50 if R < 35 else 20
        
        while not found and time.time() < deadline - 5 and attempts < max_attempts:
            attempts += 1
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            W = np.zeros((p*n, R), dtype=int)
            
            # Initialization: Random sparse
            for r in range(R):
                # Ensure at least one non-zero in each matrix for this column
                U[random.randint(0, n*m-1), r] = 1
                V[random.randint(0, m*p-1), r] = 1
                W[random.randint(0, p*n-1), r] = 1
                
                # Add more
                for _ in range(2):
                    if random.random() < 0.7:
                        U[random.randint(0, n*m-1), r] = random.choice([-1, 1])
                    if random.random() < 0.7:
                        V[random.randint(0, m*p-1), r] = random.choice([-1, 1])
                    if random.random() < 0.7:
                        W[random.randint(0, p*n-1), r] = random.choice([-1, 1])
            
            current_err = error(U, V, W)
            if current_err == 0:
                found = True
                break
                
            # Local search
            improved = True
            stagnation = 0
            while improved and time.time() < deadline - 5 and stagnation < 1000:
                improved = False
                stagnation += 1
                
                # Try a random flip
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, R-1)
                    old_val = U[idx, r]
                    new_vals = [v for v in [-1, 0, 1] if v != old_val]
                    random.shuffle(new_vals)
                    for new_val in new_vals:
                        U[idx, r] = new_val
                        new_err = error(U, V, W)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                            stagnation = 0
                            if current_err == 0:
                                found = True
                                break
                        else:
                            U[idx, r] = old_val
                elif mat == 'V':
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, R-1)
                    old_val = V[idx, r]
                    new_vals = [v for v in [-1, 0, 1] if v != old_val]
                    random.shuffle(new_vals)
                    for new_val in new_vals:
                        V[idx, r] = new_val
                        new_err = error(U, V, W)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                            stagnation = 0
                            if current_err == 0:
                                found = True
                                break
                        else:
                            V[idx, r] = old_val
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, R-1)
                    old_val = W[idx, r]
                    new_vals = [v for v in [-1, 0, 1] if v != old_val]
                    random.shuffle(new_vals)
                    for new_val in new_vals:
                        W[idx, r] = new_val
                        new_err = error(U, V, W)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                            stagnation = 0
                            if current_err == 0:
                                found = True
                                break
                        else:
                            W[idx, r] = old_val
                if found:
                    break

            if found:
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break # Found minimal rank for this search, but maybe lower exists?
                # Since we iterate upwards, this is the first found.
                # We can stop here as we want the minimal rank found.
                # But if we failed at 28, we don't know if 28 is impossible.
                # However, if we find 30, it's better than 56.
                # We can return it.
                pass

    # Final check
    if not is_valid(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
