import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify_333(U, V, W, rank):
    """Verify 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _search_333(rank, max_attempts=100):
    """Search for a rank-r decomposition of 3x3x3."""
    n, m, p = 3, 3, 3
    for attempt in range(max_attempts):
        rng = np.random.RandomState(attempt)
        # Use sparse random matrices
        density = 0.4
        U = rng.choice([-1, 0, 1], size=(n*m, rank))
        V = rng.choice([-1, 0, 1], size=(m*p, rank))
        W = rng.choice([-1, 0, 1], size=(p*n, rank))
        
        mask_u = rng.random((n*m, rank)) < density
        mask_v = rng.random((m*p, rank)) < density
        mask_w = rng.random((p*n, rank)) < density
        U = U * mask_u
        V = V * mask_v
        W = W * mask_w
        
        if _verify_333(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Strategy: Find a low-rank decomposition for 3x3x3 and double it
    # Try ranks from 26 down to 23
    for r3 in [26, 25, 24, 23]:
        if time.time() > deadline - 20:
            break
            
        # Try to find a 3x3x3 decomposition of rank r3
        result_333 = _search_333(r3, max_attempts=50)
        
        if result_333:
            U3, V3, W3 = result_333
            # Construct 3x3x6 decomposition
            U6 = np.zeros((n*m, 2*r3))
            V6 = np.zeros((m*p, 2*r3))
            W6 = np.zeros((p*n, 2*r3))
            
            # First half: for k in 0..2
            U6[:, :r3] = np.array(U3)
            for j in range(m):
                for k in range(3):
                    V6[j*6+k, :r3] = V3[j*3+k]
            for k in range(3):
                for i in range(n):
                    W6[k*3+i, :r3] = W3[k*3+i]
            
            # Second half: for k in 3..5
            U6[:, r3:] = np.array(U3)
            for j in range(m):
                for k in range(3, 6):
                    V6[j*6+k, r3:] = V3[j*3+(k-3)]
            for k in range(3, 6):
                for i in range(n):
                    W6[k*3+i, r3:] = W3[(k-3)*3+i]
            
            # Verify the 3x3x6 decomposition
            if _verify_336(U6.tolist(), V6.tolist(), W6.tolist(), 2*r3):
                if 2*r3 < best_rank:
                    best_rank = 2*r3
                    best_result = {
                        "U": U6.tolist(),
                        "V": V6.tolist(),
                        "W": W6.tolist()
                    }
                break # Found a valid one, no need to try lower r3 for this strategy
    
    return best_result

def _verify_336(U, V, W, rank):
    """Verify 3x3x6 decomposition."""
    n, m, p = 3, 3, 6
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
