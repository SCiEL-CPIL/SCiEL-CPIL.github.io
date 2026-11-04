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

def _try_structured_reduction(n, m, p, target_rank, time_budget=60.0):
    """Try to find a decomposition of the given target_rank using structured approach.
    
    For 2×5×5, exploit the structure: for each middle index j, we have terms
    (0,j,k) and (1,j,k) that share V[j*p+k]. Try to combine them.
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
    
    # Strategy: Start from schoolbook and try to eliminate terms by finding
    # linear dependencies. For n=2, terms (0,j,k) and (1,j,k) share V factor.
    # We can try to replace pairs with a single term using a linear combination.
    
    # For each j, the two terms are:
    # Term A: U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    # Term B: U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    # 
    # If we use: a*TermA + b*TermB, we get:
    # U: a at position j, b at position m+j
    # V: (a+b) at position j*p+k
    # W: a at position k*n, b at position k*n+1
    #
    # This equals the sum of the two original terms iff a=1, b=1, which gives
    # the same two terms. So simple linear combination doesn't help.
    #
    # Instead, we need to find a more clever combination across different j,k values.
    
    # Alternative: Use the fact that the tensor is sparse and try to find a
    # low-rank approximation that is exact.
    
    # Let's try a random search with better initialization
    np.random.seed(42)
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    n_restarts = min(10, max(1, int(time_budget / 3)))
    
    for restart in range(n_restarts):
        if not still_going():
            break
        
        np.random.seed(1000 + restart * 7919)
        
        # Initialize with structured random values
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Use Adam-like optimizer
        lr = 0.1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        best_loss_local = float('inf')
        best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
        
        for step in range(5000):
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
            
            # Adam update
            t = step + 1
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Decay learning rate
            if step % 500 == 0 and step > 0:
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
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to reduce rank from 49 down
    for rank in range(best_rank - 1, max(best_rank - 10, 35), -1):
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 15:
            break
        
        budget = min(remaining * 0.7, 25.0)
        
        result = _try_structured_reduction(n, m, p, rank, time_budget=budget)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
