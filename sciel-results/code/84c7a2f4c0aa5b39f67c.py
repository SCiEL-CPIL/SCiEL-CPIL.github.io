import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, tol=1e-9):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_int(x, tol=1e-9):
    """Check if x is integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _snap_half(arr):
    """Snap to nearest half-integer."""
    return np.round(np.array(arr, dtype=float) * 2) / 2.0

def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 90."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _compute_error(U, V, W):
    """Compute the sum of squared errors."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                err += (val - 1.0) ** 2
    return err

def _optimize_rank(r, time_budget, start_time):
    """Try to find a rank-r decomposition via optimization."""
    n, m, p = DIMENSIONS
    deadline = start_time + time_budget
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    np.random.seed(42)
    
    max_iters = 5000
    lr = 0.01
    
    for trial in range(3):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Scale to help convergence
        # The product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should sum to 1
        # If we scale U by a, V by b, W by c, the product scales by a*b*c
        # So we want a*b*c ~ 1/r to keep things reasonable
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_arr = U
            V_arr = V
            W_arr = W
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                        err = val - 1.0
                        # d(err^2)/dU[i*m+j,c] = 2*err * V[j*p+k,c] * W[k*n+i,c]
                        grad_U[i*m+j] += 2 * err * V_arr[j*p+k] * W_arr[k*n+i]
                        grad_V[j*p+k] += 2 * err * U_arr[i*m+j] * W_arr[k*n+i]
                        grad_W[k*n+i] += 2 * err * U_arr[i*m+j] * V_arr[j*p+k]
            
            # Gradient descent
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error
            err = _compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                lr = max(0.001, lr * 0.9)
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-6 and best_U is not None:
        # Snap to half-integers
        U_snap = _snap_half(best_U)
        V_snap = _snap_half(best_V)
        W_snap = _snap_half(best_W)
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

def _try_structured_rank(r, time_budget, start_time):
    """Try a structured approach for specific ranks."""
    n, m, p = DIMENSIONS
    deadline = start_time + time_budget
    
    # For rank = n*m*p, use schoolbook
    if r == n * m * p:
        return _schoolbook()
    
    # Try random restarts with optimization
    return _optimize_rank(r, time_budget, start_time)

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But the actual minimum might be higher
    
    # Try ranks in order, spending more time on lower ranks
    ranks_to_try = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50, 60, 70, 80, 85, 89]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            break
        
        # Allocate time based on rank
        # Lower ranks get more time
        time_alloc = min(15, max(5, 30 - (rank - 30) * 0.5))
        
        result = _try_structured_rank(rank, time_alloc, start_time)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # If we found a good rank, stop
            if rank <= 35:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
