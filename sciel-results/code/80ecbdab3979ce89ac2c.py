import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline to be safe
    
    # Trivial decomposition
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    
    # Verify a tensor decomposition
    def verify(U, V, W, tol=1e-6):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decomposition using structured random search
    # We focus on small integer/half-integer values
    
    # Try ranks from 49 down to some minimum
    for rank in range(rank_trivial - 1, max(20, rank_trivial - 20), -1):
        if time.time() > deadline:
            break
        
        # Try random initializations with small integer values
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization with small integer/half-integer values
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            if verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition for this rank
    
    if best_rank < rank_trivial:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
