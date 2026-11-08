import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_60():
    """
    Naive rank-60 decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _is_exact(U, V, W):
    """
    Check if the decomposition is exact.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T_calc[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                    return False
    return True

def _local_improve(U, V, W, max_time=60.0):
    """
    Attempt to reduce rank by eliminating redundant columns.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if not _is_exact(U, V, W):
        return U, V, W
        
    improved_U, improved_V, improved_W = U.copy(), V.copy(), W.copy()
    current_rank = improved_U.shape[1]
    
    # Try removing columns one at a time
    for attempt in range(10):
        if time.time() - start_time > max_time:
            break
        
        removed_any = False
        cols_to_try = list(range(current_rank))
        np.random.shuffle(cols_to_try)
        
        for r in cols_to_try[:]:
            if time.time() - start_time > max_time:
                break
            
            mask = [i for i in range(current_rank) if i != r]
            if len(mask) == 0:
                continue
                
            U_test = improved_U[:, mask]
            V_test = improved_V[:, mask]
            W_test = improved_W[:, mask]
            
            if _is_exact(U_test, V_test, W_test):
                improved_U = U_test
                improved_V = V_test
                improved_W = W_test
                current_rank = len(mask)
                removed_any = True
        
        if not removed_any:
            break
    
    return improved_U, improved_V, improved_W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Start with naive rank 60
    U, V, W = _construct_rank_60()
    
    # Perform local improvement
    if time.time() < DEADLINE - 10:
        U, V, W = _local_improve(U, V, W, max_time=DEADLINE - time.time() - 5)
        
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
