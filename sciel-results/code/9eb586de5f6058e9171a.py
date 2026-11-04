"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            s = np.sum(W_arr[:, k*n+i] * U_arr[:, a*m+b] * V_arr[:, c*p+d])
                            if abs(s - expected) > tol:
                                return False
    return True

def _make_schoolbook(n, m, p):
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

def _try_rank_reduction(n, m, p, target_rank, time_budget=30.0):
    """Try to find a decomposition of the given target_rank.
    
    Uses a structured approach: start from schoolbook and try to merge terms
    by exploiting linear dependencies.
    """
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < time_budget
    
    if target_rank >= n * m * p:
        return None
    
    # Try random initialization and gradient descent
    np.random.seed(42)
    
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Compute current tensor
    def compute_tensor(U, V, W):
        result = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            result += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return result
    
    # Gradient descent
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(5000):
        if not still_going():
            break
        
        # Compute loss
        T_current = compute_tensor(U, V, W)
        current_loss = np.sum((T_current - T)**2)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Compute gradients using einsum to avoid broadcasting issues
        diff = T_current - T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(target_rank):
            # grad_U[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
            grad_U[:,r] = 2 * np.einsum('ijk,j,k->i', diff, V[:,r], W[:,r])
            
            # grad_V[j,r] = 2 * sum_{i,k} diff[i,j,k] * U[i,r] * W[k,r]
            grad_V[:,r] = 2 * np.einsum('ijk,i,k->j', diff, U[:,r], W[:,r])
            
            # grad_W[k,r] = 2 * sum_{i,j} diff[i,j,k] * U[i,r] * V[j,r]
            grad_W[:,r] = 2 * np.einsum('ijk,i,j->k', diff, U[:,r], V[:,r])
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        lr *= 0.9999
    
    if best_loss < 1e-10:
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to reduce rank
    # Note: For 2x5x5, rank 50 is schoolbook. 
    # Trying to go below 50 is very hard. 
    # We attempt a few ranks but fall back to schoolbook if not found quickly.
    for rank in range(best_rank - 1, max(best_rank - 3, 40), -1):
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 15:
            break
        
        budget = min(remaining * 0.5, 10.0) # Tighter budget to ensure fallback
        
        result = _try_rank_reduction(n, m, p, rank, time_budget=budget)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
