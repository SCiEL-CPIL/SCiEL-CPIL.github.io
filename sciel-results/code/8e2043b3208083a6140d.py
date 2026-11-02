"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(rank, time_limit=20.0):
    """Try to find a rank-`rank` decomposition using structured patterns."""
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    rng = np.random.RandomState(42)
    
    best_loss = float('inf')
    best_solution = None
    
    max_attempts = 10000
    for attempt in range(max_attempts):
        if time.time() - t0 > time_limit:
            break
        
        # Random initialization with small integer/half-integer values
        U = rng.randint(-2, 3, size=(dim_u, rank)) * 0.5
        V = rng.randint(-2, 3, size=(dim_v, rank)) * 0.5
        W = rng.randint(-2, 3, size=(dim_w, rank)) * 0.5
        
        # Compute the tensor
        T = np.einsum('ijr,jkr,kir->ijk', 
                       U.reshape(n, m, rank),
                       V.reshape(m, p, rank),
                       W.reshape(p, n, rank))
        
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(dim_u)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(dim_v)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(dim_w)]
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    # Strategy: Try ranks from 53 down to 40 with time limits
    ranks_to_try = list(range(53, 39, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        time_remaining = deadline - time.time()
        time_per_attempt = min(15.0, time_remaining / max(1, len(ranks_to_try)))
        
        result = _try_structured_decomposition(rank, time_limit=time_per_attempt)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
