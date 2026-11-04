import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Compute the tensor T[u, v, w] = sum_c U[u, c] * V[v, c] * W[w, c]
    T = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
    
    # Construct the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    # Check if T matches T_target
    if not np.allclose(T, T_target, atol=tol, rtol=0):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_strassen_block(n, m, p):
    """
    Try to use Strassen-like decomposition for sub-blocks.
    """
    # For 3x4 and 4x6, we can try to partition into smaller blocks
    # This is a heuristic search for lower rank decompositions.
    pass

def _random_search(n, m, p, max_rank, time_limit):
    """
    Randomly search for a decomposition with rank <= max_rank.
    Uses random integer/half-integer entries and verifies.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_indices.append((u, v, w))
    
    # Try decreasing rank
    for rank in range(min(max_rank, n*m*p), 10, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random decompositions at this rank
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(nm, rank))
            V = np.random.randint(-2, 3, size=(mp, rank))
            W = np.random.randint(-2, 3, size=(pn, rank))
            
            # Quick check: does it produce the right support?
            T = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
            
            # Check non-zero pattern
            valid = True
            for (u, v, w) in target_indices:
                if T[u, v, w] != 1.0:
                    valid = False
                    break
            
            if valid:
                # Also check no other entries are non-zero
                T_target = np.zeros((nm, mp, pn), dtype=float)
                for (u, v, w) in target_indices:
                    T_target[u, v, w] = 1.0
                if np.allclose(T, T_target, atol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist(), rank
            
            attempts += 1
    
    return None

def _structured_search(n, m, p, time_limit):
    """
    Structured search using Kronecker products and known identities.
    For 3x4x6, we can exploit the factorization 6 = 2*3 and 4 = 2*2.
    """
    start_time = time.time()
    
    # Use the schoolbook as a starting point and try to merge terms.
    U, V, W = _schoolbook(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def _greedy_merge(n, m, p, time_limit):
    """
    Try to merge schoolbook terms by finding linear dependencies.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    U, V, W = _schoolbook(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Try to find if any columns are linearly dependent
    # For each pair of columns, check if one can be expressed as a linear combination
    # of the others. This is a complex problem, so we'll use a heuristic.
    
    # For now, just return schoolbook
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try random search for a lower rank decomposition
    result = _random_search(n, m, p, n*m*p-1, DEADLINE - 10)
    
    if result is not None:
        U, V, W, rank = result
        # Verify the result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
