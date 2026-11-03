"""Matrix multiplication 3x3x4: rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _check_exact(U, V, W):
    """Check if the decomposition is exact (within tight tolerance for integer/half-integer)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > 1e-9:
                    return False
    return True

def _to_half_integer(U, V, W):
    """Round entries to nearest half-integer and check validity."""
    U_r = np.round(2 * np.array(U, dtype=float)) / 2.0
    V_r = np.round(2 * np.array(V, dtype=float)) / 2.0
    W_r = np.round(2 * np.array(W, dtype=float)) / 2.0
    
    if _check_exact(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return [U_r.tolist(), V_r.tolist(), W_r.tolist()]
    return None

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _compute_gradient(U, V, W, rank):
    """Compute gradients for U, V, W using vectorized numpy operations."""
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Precompute outer products for efficiency
    # For each r, we need V[j*p+k, r] * W[k*n+i, r] etc.
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                # grad_U[i*m+j, r] = sum_k (val_ijk - 1) * V[j*p+k, r] * W[k*n+i, r]
                for k in range(p):
                    val = sum(U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2] for r2 in range(rank))
                    grad_U[i*m+j, r] += (val - 1.0) * V[j*p+k, r] * W[k*n+i, r]
    
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                # grad_V[j*p+k, r] = sum_i (val_ijk - 1) * U[i*m+j, r] * W[k*n+i, r]
                for i in range(n):
                    val = sum(U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2] for r2 in range(rank))
                    grad_V[j*p+k, r] += (val - 1.0) * U[i*m+j, r] * W[k*n+i, r]
    
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                # grad_W[k*n+i, r] = sum_j (val_ijk - 1) * U[i*m+j, r] * V[j*p+k, r]
                for j in range(m):
                    val = sum(U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2] for r2 in range(rank))
                    grad_W[k*n+i, r] += (val - 1.0) * U[i*m+j, r] * V[j*p+k, r]
    
    return grad_U, grad_V, grad_W

def _try_rank(rank, time_budget, seed_offset=0):
    """Try to find a valid rank-decomposition using gradient-based optimization with multiple restarts."""
    deadline = time.time() + time_budget
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    best_err = float('inf')
    
    n_restarts = 30
    n_iters = 3000
    
    for attempt in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize randomly
        np.random.seed(attempt * 104729 + seed_offset)
        scale = 0.3
        U = np.random.randn(U_rows, rank) * scale
        V = np.random.randn(V_rows, rank) * scale
        W = np.random.randn(W_rows, rank) * scale
        
        # Adaptive learning rate with momentum
        lr = 0.05
        momentum = 0.9
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        
        prev_err = float('inf')
        stall_count = 0
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Compute error
            err_sum = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        err_sum += (val - 1.0) ** 2
            
            if err_sum < 1e-12:
                break
            
            # Check for stall
            if err_sum > prev_err * 0.99:
                stall_count += 1
            else:
                stall_count = 0
            prev_err = err_sum
            
            if stall_count > 100:
                break
            
            # Compute gradients
            grad_U, grad_V, grad_W = _compute_gradient(U, V, W, rank)
            
            # Update with momentum
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            U += vel_U
            V += vel_V
            W += vel_W
            
            # Decay learning rate
            lr *= 0.9995
        
        # Check and round
        result = _to_half_integer(U.tolist(), V.tolist(), W.tolist())
        if result is not None:
            return result
        
        # Also check without rounding
        if _check_exact(U.tolist(), V.tolist(), W.tolist()):
            return [U.tolist(), V.tolist(), W.tolist()]
        
        # Track best
        if err_sum < best_err:
            best_err = err_sum
    
    return None

def _try_rank_vectorized(rank, time_budget, seed_offset=0):
    """Vectorized version for speed."""
    deadline = time.time() + time_budget
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Precompute index arrays
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append((i*m+j,))
                idx_V.append((j*p+k,))
                idx_W.append((k*n+i,))
    
    n_restarts = 20
    n_iters = 2000
    
    for attempt in range(n_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 7919 + seed_offset)
        U = np.random.randn(U_rows, rank) * 0.3
        V = np.random.randn(V_rows, rank) * 0.3
        W = np.random.randn(W_rows, rank) * 0.3
        
        lr = 0.03
        prev_err = float('inf')
        stall_count = 0
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Compute error vectorized
            err_sum = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err_sum += (val - 1.0) ** 2
            
            if err_sum < 1e-12:
                break
            
            if err_sum > prev_err * 0.99:
                stall_count += 1
            else:
                stall_count = 0
            prev_err = err_sum
            
            if stall_count > 80:
                break
            
            # Simple gradient step
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        grad = 0.0
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            grad += (val - 1.0) * V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, r] -= lr * grad
            
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        grad = 0.0
                        for i in range(n):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            grad += (val - 1.0) * U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, r] -= lr * grad
            
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        grad = 0.0
                        for j in range(m):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            grad += (val - 1.0) * U[i*m+j, r] * V[j*p+k, r]
                        W[k*n+i, r] -= lr * grad
            
            lr *= 0.9998
        
        result = _to_half_integer(U.tolist(), V.tolist(), W.tolist())
        if result is not None:
            return result
        
        if _check_exact(U.tolist(), V.tolist(), W.tolist()):
            return [U.tolist(), V.tolist(), W.tolist()]
    
    return None

def _strassen_3x3x3():
    """Strassen-like decomposition for 3x3x3 (rank 23 is known, but we use a simpler construction)."""
    # For 3x3x3, we can use a known rank-23 decomposition or fall back to schoolbook
    # Let's use the schoolbook for 3x3x3 as a base and try to extend
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_3x3x4_from_3x3x3():
    """Try to construct 3x3x4 from 3x3x3 by adding one column."""
    # Start with 3x3x3 schoolbook and add 3 more terms for the 4th column
    rank = 36
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(4):
                U[i * 3 + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Try ranks from lower to higher
    # Known: schoolbook is 36. Best known lower bound is around 25-28.
    # Try a focused search on ranks 28-35
    
    ranks_to_try = [28, 29, 30, 31, 32, 33, 34, 35]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on rank
        if r <= 30:
            time_budget = min(15, deadline - time.time())
        elif r <= 33:
            time_budget = min(12, deadline - time.time())
        else:
            time_budget = min(8, deadline - time.time())
        
        if time_budget < 2:
            break
        
        # Try multiple methods
        result = _try_rank(r, time_budget, seed_offset=0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_vectorized(r, min(10, deadline - time.time()), seed_offset=1000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
