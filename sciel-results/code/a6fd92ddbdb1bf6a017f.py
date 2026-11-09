"""Program for matrix multiplication 2,3,7 rank decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Build the target tensor
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) match, else 0
    # We need: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
    
    # Check all non-zero entries of T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all zero entries of T (this is expensive, so we check a sample)
    # Total entries: (n*m)*(m*p)*(p*n) = 42*21*14 = 12348
    # Non-zero entries: n*m*p = 42
    # We need to verify the rest are ~0
    
    # For efficiency, check a random sample plus structured checks
    all_abc = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_set.add((a, b, c))
    
    zero_entries = [(a, b, c) for (a, b, c) in all_abc if (a, b, c) not in nonzero_set]
    
    # Check all zero entries (there are 12348 - 42 = 12306)
    for a, b, c in zero_entries:
        val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
        if abs(val) > tol:
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 128.0  # 128s safety margin (120s limit + buffer)
    
    # Trivial solution: rank = 42
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial), dtype=float)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=float)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=float)
    
    r = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U_trivial[i * m + j, r] = 1.0
                V_trivial[j * p + k, r] = 1.0
                W_trivial[k * n + i, r] = 1.0
                r += 1
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Try to find lower rank solutions using random initialization + verification
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n, n*p) = max(6, 21, 14, 14) = 21
    # But actually for matrix multiplication, the lower bound is more nuanced.
    # For 2x3 * 3x7, the output is 2x7 = 14 entries.
    # Strassen-like bounds suggest rank could be around 14-21.
    
    # Let's try ranks from 14 up to 41
    # For each rank, use multiple random restarts
    
    np.random.seed(123)
    
    for test_rank in range(14, 42):
        if time.time() > deadline:
            break
        
        # Determine number of attempts based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        max_attempts = min(50, int(remaining / 0.5))  # ~0.5s per attempt
        max_attempts = max(5, max_attempts)
        
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, test_rank)
            V = np.random.randn(m * p, test_rank)
            W = np.random.randn(p * n, test_rank)
            
            # Scale to improve chances
            U /= np.sqrt(test_rank)
            V /= np.sqrt(test_rank)
            W /= np.sqrt(test_rank)
            
            # Quick check on non-zero entries
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = test_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break  # Found solution at this rank
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
