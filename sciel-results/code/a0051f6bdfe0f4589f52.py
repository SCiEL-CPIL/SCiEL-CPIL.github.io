import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, with 10s safety margin

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-6):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[k*n+i, i*m+j, j*p+k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Build the target tensor T
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
                
    # Compute the decomposition
    T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
    
    return np.allclose(T_decomp, T, atol=tol)

def _make_schoolbook(n=3, m=3, p=6):
    """Create the schoolbook decomposition of rank n*m*p."""
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

def _try_random_decomposition(n, m, p, target_rank, rng, max_iters=500, deadline=None):
    """Try to find a decomposition of given rank using random integer/half-integer initialization."""
    nm, mp, pn = n * m, m * p, p * n
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
            
        # Random integer/half-integer entries in {-1, -0.5, 0, 0.5, 1}
        vals = [-1, -0.5, 0, 0.5, 1]
        U = np.array([[rng.choice(vals) for _ in range(target_rank)] for _ in range(nm)], dtype=float)
        V = np.array([[rng.choice(vals) for _ in range(target_rank)] for _ in range(mp)], dtype=float)
        W = np.array([[rng.choice(vals) for _ in range(target_rank)] for _ in range(pn)], dtype=float)
        
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def _gradient_descent(U, V, W, n, m, p, max_iters=5000, lr=0.01, deadline=None, tol=1e-8):
    """Refine a decomposition using gradient descent to minimize ||T_decomp - T||."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]
    
    # Build target tensor
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
                
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        R = T_decomp - T
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
        grad_U = np.einsum('abc,cr,ar->br', R, V, W)
        grad_V = np.einsum('abc,br,ar->cr', R, U, W)
        grad_W = np.einsum('abc,br,cr->ar', R, U, V)
        
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 1e-12:
            step_lr = lr * err / grad_norm
        else:
            step_lr = lr
            
        U -= step_lr * grad_U
        V -= step_lr * grad_V
        W -= step_lr * grad_W
        
    if best_err < tol:
        return best_U, best_V, best_W
    return None

def _greedy_remove_columns(U, V, W, n, m, p, deadline=None):
    """Try to reduce rank by removing columns that don't contribute."""
    rank = U.shape[1]
    current_rank = rank
    
    for _ in range(100):
        if deadline is not None and time.time() > deadline:
            break
        if current_rank <= max(n * m, m * p, p * n):
            break
            
        removed = False
        for c in range(current_rank):
            U_new = np.delete(U, c, axis=1)
            V_new = np.delete(V, c, axis=1)
            W_new = np.delete(W, c, axis=1)
            
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-9):
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                removed = True
                break
                
        if not removed:
            break
            
    return U, V, W

def construct():
    global START_TIME
    START_TIME = time.time()
    deadline = START_TIME + TIME_LIMIT
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    lower_bound = max(nm, mp, pn)
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    rng = random.Random(42)
    
    # Strategy 1: Try random search for lower ranks
    # Search from lower bound upward
    search_start = lower_bound
    search_end = min(schoolbook_rank - 1, lower_bound + 20)
    
    for rank in range(search_start, search_end + 1):
        if time.time() > deadline - 20:
            break
            
        result = _try_random_decomposition(n, m, p, rank, rng, max_iters=200, deadline=deadline - 20)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break
    
    # Strategy 2: Gradient descent refinement starting from schoolbook
    if time.time() < deadline - 10:
        # Try to refine the schoolbook decomposition
        refined = _gradient_descent(U_fb, V_fb, W_fb, n, m, p, max_iters=1000, lr=0.05, deadline=deadline - 10)
        if refined is not None:
            U, V, W = refined
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
                
                # Try to reduce rank further by removing columns
                U, V, W = _greedy_remove_columns(U, V, W, n, m, p, deadline=deadline - 5)
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
    
    # Strategy 3: If we have a decomposition, try to reduce its rank
    if best_rank < schoolbook_rank and time.time() < deadline - 5:
        U, V, W = _greedy_remove_columns(best_U, best_V, best_W, n, m, p, deadline=deadline - 5)
        if _verify(U, V, W, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
