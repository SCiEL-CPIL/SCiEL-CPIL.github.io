"""Search for low-rank decomposition of 4x5x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
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

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _try_rank_random(r, n, m, p, max_iters=2000, timeout=8.0, seed_base=0):
    """Try random {-1,0,1} entries for rank r."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        rng = np.random.RandomState(seed_base + it)
        U = rng.randint(-1, 2, size=(n * m, r))
        V = rng.randint(-1, 2, size=(m * p, r))
        W = rng.randint(-1, 2, size=(p * n, r))
        
        if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_rank_gaussian(r, n, m, p, max_iters=50, timeout=5.0, seed_base=1000):
    """Try Gaussian random initialization with rounding."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        rng = np.random.RandomState(seed_base + it)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _verify_numpy(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return None

def _try_rank_structured(r, n, m, p, timeout=10.0):
    """
    Structured approach: start with schoolbook and try to merge/compress.
    For 4x5x5, we can try to use the structure of the 5-dim inner product.
    """
    start_time = time.time()
    
    # Strategy: use the fact that for each (i,k) pair, we have a 5-dim dot product.
    # We can try to represent each dot product using fewer than 5 rank-1 terms
    # by sharing across different (i,k) pairs.
    
    # For now, try a block-structured approach
    # Each "block" handles a subset of the inner dimension
    
    for attempt in range(20):
        if time.time() - start_time > timeout:
            break
        
        rng = np.random.RandomState(42 + attempt)
        
        # Initialize with sparse structure
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Assign random subsets of the 100 schoolbook terms to r columns
        # Each column gets ~100/r terms
        terms_per_col = 100 // r
        extra = 100 % r
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
                    # If col >= r, this term is dropped (incomplete)
        
        # This is just a partial schoolbook, won't work. Need different approach.
        
        # Alternative: random sparse initialization
        density = max(0.1, 3.0 / r)
        U = (rng.rand(n * m, r) < density).astype(float)
        V = (rng.rand(m * p, r) < density).astype(float)
        W = (rng.rand(p * n, r) < density).astype(float)
        
        # Try to fix by setting some entries to ±1
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
            if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Random perturbation
            idx = rng.randint(0, n * m)
            col = rng.randint(0, r)
            U[idx, col] = rng.choice([-1, 0, 1])
            
            idx = rng.randint(0, m * p)
            col = rng.randint(0, r)
            V[idx, col] = rng.choice([-1, 0, 1])
            
            idx = rng.randint(0, p * n)
            col = rng.randint(0, r)
            W[idx, col] = rng.choice([-1, 0, 1])
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try progressively lower ranks
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Try from reasonable lower bounds upward
    
    # First, try some known good ranks with random search
    # For 4x5x5, let's try ranks from 50 down to 25
    for rank in range(50, 24, -1):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Try random {-1,0,1} search
        t_budget = min(8.0, remaining * 0.4)
        result = _try_rank_random(rank, n, m, p, max_iters=3000, timeout=t_budget, seed_base=rank * 100)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        if time.time() > deadline - 15:
            break
        
        # Try Gaussian with rounding
        t_budget = min(5.0, remaining * 0.3)
        result = _try_rank_gaussian(rank, n, m, p, max_iters=30, timeout=t_budget, seed_base=rank * 200)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        if time.time() > deadline - 10:
            break
        
        # Try structured
        t_budget = min(5.0, remaining * 0.3)
        result = _try_rank_structured(rank, n, m, p, timeout=t_budget)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
