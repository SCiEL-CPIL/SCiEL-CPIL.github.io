import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Trivial decomposition as fallback (Rank 36)
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    random.seed(42)
    
    # Target ranks to try, starting from the known improvement (34)
    # We try 34 first, then 33, then 32
    ranks_to_try = [34, 33, 32]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # We allow a certain number of attempts per rank
        # For rank 34, we expect it to be findable
        max_attempts = 200000 if target_rank == 34 else 50000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            
            # Random entries in {-1, 0, 1}
            # We can vary density
            for i in range(n * m):
                for t in range(target_rank):
                    r_val = random.random()
                    if r_val < 0.33:
                        U[i, t] = 1
                    elif r_val < 0.66:
                        U[i, t] = -1
                    else:
                        U[i, t] = 0
            
            for i in range(m * p):
                for t in range(target_rank):
                    r_val = random.random()
                    if r_val < 0.33:
                        V[i, t] = 1
                    elif r_val < 0.66:
                        V[i, t] = -1
                    else:
                        V[i, t] = 0
            
            # Check if W can be solved
            W_valid = True
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            for k in range(p):
                for i in range(n):
                    # Build A (m x r)
                    # A[j, t] = U[i*m+j, t] * V[j*p+k, t]
                    # We can vectorize this construction
                    # U_row = U[i*m:(i+1)*m, :]
                    # V_col = V[k::p, :] # No, V is indexed by j*p+k
                    # V indices for fixed k are j*p+k.
                    # So V_slice = V[np.arange(m)*p + k, :]
                    
                    U_slice = U[i*m:(i+1)*m, :]
                    V_slice = V[np.arange(m)*p + k, :]
                    
                    A = U_slice * V_slice # Element-wise product, shape (m, r)
                    b = np.ones(m, dtype=np.float64)
                    
                    # Solve A x = b
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if np.any(residuals > 1e-9):
                            W_valid = False
                            break
                        # Check for nice numbers (integers or half-integers)
                        scaled = sol * 2.0
                        if not np.allclose(scaled, np.round(scaled), atol=1e-6):
                            W_valid = False
                            break
                        W[k * n + i, :] = sol
                    except np.linalg.LinAlgError:
                        W_valid = False
                        break
                if not W_valid:
                    break
            
            if W_valid:
                # Final verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for t in range(target_rank):
                                s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_result = (target_rank, U, V, W)
                    return {"U": best_result[1].tolist(), "V": best_result[2].tolist(), "W": best_result[3].tolist()}
        
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
