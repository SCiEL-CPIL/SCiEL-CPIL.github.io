"""Matrix multiplication tensor decomposition for 2x3 by 3x10."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W reconstruct T exactly."""
    recon = np.einsum('ak,bk,ck->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _try_rank_decomposition(n, m, p, rank, T, num_trials=200, time_budget=30):
    """Try to find a rank-r decomposition using random initialization and Newton's method.
    
    For a given rank r, we want to find U (nm x r), V (mp x r), W (pn x r) such that
    T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
    
    We use a random initialization and then refine using gradient-based optimization.
    """
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(trial * 12345 + rank)
        
        # Initialize with small random values
        scale = 1.0 / np.sqrt(rank)
        U = rng.randn(nm, rank) * scale
        V = rng.randn(mp, rank) * scale
        W = rng.randn(pn, rank) * scale
        
        # Run Adam-like optimization
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        best_loss = np.inf
        
        for step in range(2000):
            t += 1
            
            # Compute reconstruction
            recon = np.einsum('ak,bk,ck->abc', U, V, W)
            diff = recon - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-10:
                return U, V, W, loss
            
            if loss < best_loss:
                best_loss = loss
            
            # Early termination if stuck
            if step > 500 and loss > 1e-3:
                break
            
            # Compute gradients
            grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
            grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
            grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
            
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Decay learning rate
            lr = 0.01 / (1 + step / 500)
        
        if best_loss < 1e-8:
            return U, V, W, best_loss
    
    return None, None, None, np.inf

def _try_structured_decomposition(n, m, p, T):
    """Try to find a decomposition with rank < n*m*p using structure.
    
    For 2x3 by 3x10, the schoolbook rank is 60.
    We try ranks from 30 (lower bound) up to 59.
    """
    nm, mp, pn = n * m, m * p, p * n
    max_rank = n * m * p  # 60
    
    # Lower bound: max(nm, mp, pn) = max(6, 30, 20) = 30
    min_rank = max(nm, mp, pn)
    
    # Try ranks from min_rank to max_rank-1
    # Focus on ranks close to max_rank first since they're more likely to work
    ranks_to_try = list(range(max_rank - 1, min_rank - 1, -1))
    
    # Limit total time
    start_time = time.time()
    total_budget = 100  # seconds
    
    for rank in ranks_to_try:
        if time.time() - start_time > total_budget:
            break
        
        # Give more trials to lower ranks (harder) but less time each
        remaining = total_budget - (time.time() - start_time)
        trials_per_rank = min(50, max(10, int(remaining / len(ranks_to_try))))
        
        result = _try_rank_decomposition(n, m, p, rank, T, 
                                         num_trials=trials_per_rank,
                                         time_budget=min(20, remaining))
        
        if result[0] is not None:
            U, V, W, loss = result
            print(f"Found rank-{rank} decomposition with loss {loss}")
            return U, V, W, rank
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    T = _build_tensor(n, m, p)
    
    # Try to find a better decomposition
    try:
        U, V, W, rank = _try_structured_decomposition(n, m, p, T)
        if U is not None and rank < n * m * p:
            # Verify the decomposition
            if _verify_decomposition(U, V, W, T, tol=1e-9):
                print(f"Verified rank-{rank} decomposition")
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            else:
                print(f"Rank-{rank} decomposition failed verification")
    except Exception as e:
        print(f"Exception in structured decomposition: {e}")
    
    # Fallback to schoolbook
    sb = _schoolbook_decomposition(n, m, p)
    return {'U': sb['U'].tolist(), 'V': sb['V'].tolist(), 'W': sb['W'].tolist()}

# EVOLVE-BLOCK-END
