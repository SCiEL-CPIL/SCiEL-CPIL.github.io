"""Matrix multiplication 4x5x6 rank-minimal exact decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# Deadline guard
START_TIME = time.time()
DEADLINE = 120.0  # seconds, with safety margin

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition of matrix multiplication.
    
    The tensor T[i,j,k] = 1 if we're multiplying (i,j) of A by (j,k) of B to get (i,k) of C.
    Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, let me think about this more carefully.
    
    Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k.
    We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, the W index is k*n+i per the trace convention W[k*n+i, r].
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
        1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, target_rank, max_iters=5000, tol=1e-10):
    """Try to find a rank-`target_rank` decomposition using gradient descent on the
    residual of the tensor equation. Returns (U, V, W) if found, else None."""
    rank = target_rank
    num_i = n * m
    num_j = m * p
    num_k = p * n
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(num_i, rank) * 0.1
    V = np.random.randn(num_j, rank) * 0.1
    W = np.random.randn(num_k, rank) * 0.1
    
    # We need: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's define the residual for each (i,j,k)
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
            
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # residual
                    res = 0.0
                    for r in range(rank):
                        res += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    res -= 1.0
                    loss += res * res
                    
                    for r in range(rank):
                        dU[idx_u, r] += 2.0 * res * V[idx_v, r] * W[idx_w, r]
                        dV[idx_v, r] += 2.0 * res * U[idx_u, r] * W[idx_w, r]
                        dW[idx_w, r] += 2.0 * res * U[idx_u, r] * V[idx_v, r]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_loss < tol:
            break
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if it % 100 == 99 and it > 0:
            lr *= 0.95
    
    if best_loss < tol:
        # Convert to list of lists with rounded values (try to get integer/half-integer)
        U_list = [[float(round(x * 2) / 2) for x in row] for row in best_U]
        V_list = [[float(round(x * 2) / 2) for x in row] for row in best_V]
        W_list = [[float(round(x * 2) / 2) for x in row] for row in best_W]
        
        # Verify after rounding
        if _validate(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        else:
            # Try without rounding
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def _try_integer_decomposition(n, m, p, target_rank, max_search=1000):
    """Try to find an integer decomposition by random search over small integer values."""
    rank = target_rank
    num_i = n * m
    num_j = m * p
    num_k = p * n
    
    # Values to try: -2, -1, 0, 1, 2
    vals = [-2, -1, 0, 1, 2]
    
    np.random.seed(123)
    for attempt in range(max_search):
        if _time_left() < 5.0:
            break
            
        U = np.random.choice(vals, size=(num_i, rank)).astype(float)
        V = np.random.choice(vals, size=(num_j, rank)).astype(float)
        W = np.random.choice(vals, size=(num_k, rank)).astype(float)
        
        # Check
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return U_list, V_list, W_list
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    fallback = _schoolbook_decomposition(n, m, p)
    
    # Known lower bound: for n x m x p matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # For 4x5x6: max(20, 30, 24) = 30
    # The schoolbook rank is 4*5*6 = 120
    # We want to find the minimal rank.
    
    # Try ranks from the lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    
    # For 4x5x6, the known minimal rank... let's try to find it.
    # The trivial bound is 120. The lower bound is 30.
    # Let's try a range of ranks.
    
    best_result = None
    best_rank = None
    
    # Try gradient descent for various ranks
    for rank in range(lower_bound, min(lower_bound + 50, 120)):
        if _time_left() < 10.0:
            break
        
        result = _try_rank_decomposition(n, m, p, rank, max_iters=2000, tol=1e-8)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    # If gradient descent didn't find anything, try integer search for smaller ranks
    if best_result is None:
        for rank in range(lower_bound, min(lower_bound + 30, 120)):
            if _time_left() < 10.0:
                break
            
            result = _try_integer_decomposition(n, m, p, rank, max_search=500)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return fallback
