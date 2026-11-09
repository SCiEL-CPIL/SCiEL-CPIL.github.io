import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0  # safety margin before 150s kill

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    rank = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_rank(r, n, m, p, time_limit=30.0, seed=0):
    """Try to find a decomposition of rank r. Returns (U,V,W) or None."""
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Number of (k,i) pairs
    num_ki = p * n
    num_ij = n * m
    num_jk = m * p
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Initialize U and V randomly
        U = rng.randn(num_ij, r)
        V = rng.randn(num_jk, r)
        
        # For each (k,i), we need to solve A_{k,i} @ w_{k,i} = 1
        # where A_{k,i}[j, c] = U[i*m+j, c] * V[j*p+k, c]
        W = np.zeros((num_ki, r))
        valid = True
        
        for ki in range(num_ki):
            k = ki // n
            i = ki % n
            # Build A_{k,i}: m x r matrix
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            
            # Solve A @ w = 1 (m equations, r unknowns)
            if r >= m:
                # Use least squares, check residual
                w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                if len(residuals) > 0:
                    if residuals[0] > 1e-6:
                        valid = False
                        break
                W[ki, :] = w
            else:
                valid = False
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        if _verify(W.tolist(), V.tolist(), U.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _structured_search(r, n, m, p, time_limit=20.0):
    """Try structured initializations for U and V."""
    start = time.time()
    rng = np.random.RandomState(42)
    
    # Try sparse U and V
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        
        density = max(0.1, 0.5 - 0.3 * (r - 50) / 10) if r < 60 else 0.3
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        
        for i in range(n*m):
            for c in range(r):
                if rng.random() < density:
                    U[i, c] = rng.choice([-1, 1])
        
        for j in range(m*p):
            for c in range(r):
                if rng.random() < density:
                    V[j, c] = rng.choice([-1, 1])
        
        # Try to solve for W
        W = np.zeros((p*n, r))
        valid = True
        
        for ki in range(p*n):
            k = ki // n
            i = ki % n
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            
            if r >= m:
                w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
                W[ki, :] = w
            else:
                valid = False
                break
        
        if not valid:
            continue
        
        if _verify(W.tolist(), V.tolist(), U.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # Realistic lower bound for 3x4x5 is likely around 50-55
    
    # Try ranks from 55 down to 20, but prioritize higher ranks first
    # (more likely to find a solution)
    
    # Time budget allocation
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Try a few promising ranks
    ranks_to_try = []
    for r in range(59, 19, -1):
        if r < best_rank:
            ranks_to_try.append(r)
    
    # Limit attempts based on time
    max_attempts = min(len(ranks_to_try), 15)
    ranks_to_try = ranks_to_try[:max_attempts]
    
    for r in ranks_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Allocate time per rank
        time_per_rank = min(15.0, (DEADLINE - 10 - (time.time() - start_time)) / max(1, len(ranks_to_try) - ranks_to_try.index(r)))
        
        result = _try_rank(r, n, m, p, time_limit=time_per_rank, seed=r)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found a valid decomposition, stop
        
        # Also try structured search
        if time.time() - start_time < DEADLINE - 10:
            result2 = _structured_search(r, n, m, p, time_limit=min(5.0, time_per_rank/3))
            if result2 is not None:
                U, V, W = result2
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
    
    # Final safety check
    if time.time() - start_time > DEADLINE - 5:
        return _schoolbook()
    
    return best
