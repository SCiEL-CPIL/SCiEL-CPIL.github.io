import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all (i,j,k) entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_decomposition(n, m, p, rank, seed, max_iters=5000):
    """Try to find a valid decomposition at given rank using coordinate descent."""
    rng = np.random.RandomState(seed)
    
    # Initialize with structured random values
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute error tensor
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        err = np.mean((result - target) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-12:
            break
        
        # Coordinate descent: update one matrix at a time
        # For U: fix V, W, solve for each U[u, c]
        # The contribution of U[u, c] is: U[u,c] * (V * W)[u, c] where (V*W)[u,c] = sum over j,k of V[j*p+k,c]*W[k*n+i,c]
        # This is complex, so use simple gradient step
        lr = 0.005
        U += lr * rng.randn(n * m, rank) * (1.0 + it * 0.001)
        V += lr * rng.randn(m * p, rank) * (1.0 + it * 0.001)
        W += lr * rng.randn(p * n, rank) * (1.0 + it * 0.001)
        
        # Occasionally try a larger step
        if it % 100 == 0:
            U += 0.02 * rng.randn(n * m, rank)
            V += 0.02 * rng.randn(m * p, rank)
            W += 0.02 * rng.randn(p * n, rank)
    
    if best_err < 1e-10:
        # Round to nearest half-integer if close
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        
        # Check if rounding to half-integers still works
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        else:
            return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Theoretical lower bound
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound + 1 upward
    # Focus on a reasonable range given time constraints
    ranks_to_try = list(range(lower_bound + 1, min(lower_bound + 30, best_rank)))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds for this rank
        max_seeds = 20 if rank < 35 else 10
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            result = _try_decomposition(n, m, p, rank, seed, max_iters=3000)
            if result is not None:
                U_new, V_new, W_new = result
                
                # Verify the result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = rank
                    # Found a valid decomposition, return immediately
                    return best_result
    
    # If no better found, return schoolbook
    return best_result
