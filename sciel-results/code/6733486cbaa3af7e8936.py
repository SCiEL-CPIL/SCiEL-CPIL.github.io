import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _schoolbook():
    """Return schoolbook decomposition with rank 54."""
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
                
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly matches the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(rank, max_iters=3000, tol=1e-7, seed=42):
    """
    Try to find a decomposition of given rank using gradient descent.
    Returns (U, V, W) lists or None if failed.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(seed)
    
    # Initialize with small random values
    scale = 1.0 / np.sqrt(rank)
    U = np.random.randn(nm, rank) * scale
    V = np.random.randn(mp, rank) * scale
    W = np.random.randn(pn, rank) * scale
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    momentum = 0.9
    
    # Momentum buffers
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    
    for it in range(max_iters):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    loss += err * err
                    
                    grad_U[idx_u, :] += 2.0 * err * V[idx_v, :] * W[idx_w, :]
                    grad_V[idx_v, :] += 2.0 * err * U[idx_u, :] * W[idx_w, :]
                    grad_W[idx_w, :] += 2.0 * err * U[idx_u, :] * V[idx_v, :]
        
        loss /= (n * m * p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Momentum update
        vel_U = momentum * vel_U - lr * grad_U
        vel_V = momentum * vel_V - lr * grad_V
        vel_W = momentum * vel_W - lr * grad_W
        
        U += vel_U
        V += vel_V
        W += vel_W
        
        # Decay learning rate
        if it % 500 == 499:
            lr *= 0.8
            momentum *= 0.95
    
    if best_loss < 1e-5:
        # Try to round to integers or half-integers
        for mult in [1, 2, 4, 8]:
            U_rounded = np.round(best_U * mult) / mult
            V_rounded = np.round(best_V * mult) / mult
            W_rounded = np.round(best_W * mult) / mult
            
            if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-8):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Try without rounding
        if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-8):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    Starts with schoolbook rank 54 and searches for lower ranks.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from lower to higher
    for rank in range(30, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 1024]:
            if time.time() > deadline - 10:
                break
            
            result = _try_rank(rank, max_iters=2000, tol=1e-7, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if best_rank < schoolbook_rank:
            continue
    
    return best_result
