import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute S_ijk = sum_t U_ijt V_jkt W_kit
    # Using einsum
    S = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as default
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try ranks from a lower bound upward
    # We know rank < 144 is possible (e.g., 114)
    # Search from 100 to 143
    for r in range(100, best_rank):
        if time.time() > deadline:
            break
        
        # Try many random attempts with diverse entry sets
        # Use numpy for speed
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Generate random integer/half-integer matrices
            # Using small integers to avoid overflow/precision issues
            U = np.random.choice([-1, 0, 1, 2, -2], size=(n*m, r))
            V = np.random.choice([-1, 0, 1, 2, -2], size=(m*p, r))
            W = np.random.choice([-1, 0, 1, 2, -2], size=(p*n, r))
            
            if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best
