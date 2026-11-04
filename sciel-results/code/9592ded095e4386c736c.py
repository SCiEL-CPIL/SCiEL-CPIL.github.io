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
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Compute decomposed tensor
    T_comp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_comp += np.einsum('i,j,k->ijk', U_arr[r], V_arr[r], W_arr[r])
    
    if np.max(np.abs(T_comp - T)) > tol:
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
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Precompute indices for efficient computation
    T_flat = T.flatten()
    
    # Multiple random restarts
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    n_restarts = min(5, max(1, int(time_budget / 5)))
    
    for restart in range(n_restarts):
        if not still_going():
            break
        
        np.random.seed(42 + restart)
        
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Adaptive learning rate
        lr = 0.05
        best_loss_local = float('inf')
        best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
        
        # Use momentum
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        momentum = 0.9
        
        for step in range(3000):
            if not still_going():
                break
            
            # Compute tensor efficiently
            T_current = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_current += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            diff = T_current - T
            current_loss = np.sum(diff**2)
            
            if current_loss < best_loss_local:
                best_loss_local = current_loss
                best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
            
            if best_loss_local < 1e-12:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(target_rank):
                grad_U[:,r] = 2 * np.einsum('ijk,j,k->i', diff, V[:,r], W[:,r])
                grad_V[:,r] = 2 * np.einsum('ijk,i,k->j', diff, U[:,r], W[:,r])
                grad_W[:,r] = 2 * np.einsum('ijk,i,j->k', diff, U[:,r], V[:,r])
            
            # Momentum update
            vU = momentum * vU + (1 - momentum) * grad_U
            vV = momentum * vV + (1 - momentum) * grad_V
            vW = momentum * vW + (1 - momentum) * grad_W
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            # Adaptive learning rate
            if step % 100 == 0 and step > 0:
                lr *= 0.95
        
        if best_loss_local < best_loss:
            best_loss = best_loss_local
            best_U, best_V, best_W = best_U_local, best_V_local, best_W_local
    
    if best_loss < 1e-10 and best_U is not None:
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
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
    for rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 20:
            break
        
        budget = min(remaining * 0.6, 15.0) # More budget for search
        
        result = _try_rank_reduction(n, m, p, rank, time_budget=budget)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
