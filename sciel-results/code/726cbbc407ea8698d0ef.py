"""Find a low-rank exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
_START = time.time()
_DEADLINE = 135.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p, tol=1e-8):
    """Check if the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _compute_loss(U_arr, V_arr, W_arr, n, m, p):
    """Compute sum of squared errors for the tensor decomposition."""
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                loss += (val - 1.0) ** 2
    return loss

def _try_reduce_rank(n, m, p, target_rank, max_iter=2000, seed=42):
    """Try to find a decomposition with the given target rank using gradient descent."""
    np.random.seed(seed)
    r = target_rank
    
    # Initialize with structured random values
    U_arr = np.random.randn(n*m, r) * 0.5
    V_arr = np.random.randn(m*p, r) * 0.5
    W_arr = np.random.randn(p*n, r) * 0.5
    
    # Scale to help with convergence
    scale = 1.0 / np.sqrt(r)
    U_arr *= scale
    V_arr *= scale
    W_arr *= scale
    
    best_loss = float('inf')
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    lr = 0.05
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Compute loss
        loss = _compute_loss(U_arr, V_arr, W_arr, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        if loss < 1e-10:
            break
        
        # Compute gradients
        dU = np.zeros_like(U_arr)
        dV = np.zeros_like(V_arr)
        dW = np.zeros_like(W_arr)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m+j
                    idx_v = j*p+k
                    idx_w = k*n+i
                    val = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                    err = val - 1.0
                    
                    # Gradient w.r.t. U
                    dU[idx_u] += 2 * err * V_arr[idx_v] * W_arr[idx_w]
                    # Gradient w.r.t. V
                    dV[idx_v] += 2 * err * U_arr[idx_u] * W_arr[idx_w]
                    # Gradient w.r.t. W
                    dW[idx_w] += 2 * err * U_arr[idx_u] * V_arr[idx_v]
        
        # Update
        U_arr -= lr * dU
        V_arr -= lr * dV
        W_arr -= lr * dW
        
        # Adaptive learning rate
        if it % 100 == 0 and loss > 1e-6:
            lr *= 0.95
    
    # Final check
    final_loss = _compute_loss(best_U, best_V, best_W, n, m, p)
    if final_loss < 1e-8:
        # Convert to lists with rounded values
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        # Validate
        if _validate(U_list, V_list, W_list, n, m, p, tol=1e-7):
            return U_list, V_list, W_list
    
    return None

def _polish(U, V, W, n, m, p, max_iter=500):
    """High-precision polish pass with tighter tolerance."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    lr = 0.01
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 2:
            break
        
        loss = _compute_loss(U_arr, V_arr, W_arr, n, m, p)
        if loss < 1e-14:
            break
        
        dU = np.zeros_like(U_arr)
        dV = np.zeros_like(V_arr)
        dW = np.zeros_like(W_arr)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m+j
                    idx_v = j*p+k
                    idx_w = k*n+i
                    val = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                    err = val - 1.0
                    
                    dU[idx_u] += 2 * err * V_arr[idx_v] * W_arr[idx_w]
                    dV[idx_v] += 2 * err * U_arr[idx_u] * W_arr[idx_w]
                    dW[idx_w] += 2 * err * U_arr[idx_u] * V_arr[idx_v]
        
        U_arr -= lr * dU
        V_arr -= lr * dV
        W_arr -= lr * dW
    
    U_list = [[round(float(x), 10) for x in row] for row in U_arr]
    V_list = [[round(float(x), 10) for x in row] for row in V_arr]
    W_list = [[round(float(x), 10) for x in row] for row in W_arr]
    
    if _validate(U_list, V_list, W_list, n, m, p, tol=1e-9):
        return U_list, V_list, W_list
    
    return None

def construct():
    global _START
    _START = time.time()
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Try a range of ranks with multiple random seeds
    
    candidate_ranks = [40, 50, 60, 70, 80, 90, 100, 104]
    
    for r in candidate_ranks:
        if _elapsed() > _DEADLINE - 10:
            break
        
        found = False
        for seed in [42, 123, 456, 789, 1024]:
            if _elapsed() > _DEADLINE - 10:
                break
            
            result = _try_reduce_rank(n, m, p, r, max_iter=1500, seed=seed)
            if result is not None:
                U_new, V_new, W_new = result
                
                # Try to polish
                polished = _polish(U_new, V_new, W_new, n, m, p, max_iter=300)
                if polished is not None:
                    U_new, V_new, W_new = polished
                
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
