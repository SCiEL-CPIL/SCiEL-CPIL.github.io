import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if rank == 0:
        return False
        
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

def _random_search(n, m, p, target_rank, time_limit):
    """
    Search for a decomposition with exactly target_rank.
    """
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_indices.append((u, v, w))
                
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for (u, v, w) in target_indices:
        T_target[u, v, w] = 1.0
        
    # Precompute the target values for quick check
    # We only need to check the 72 non-zero entries
    
    attempts = 0
    max_attempts = 5000 # Increase attempts
    
    while attempts < max_attempts:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random integers
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for c in range(target_rank):
            # Pick a random target index to "aim" at
            idx = np.random.randint(0, len(target_indices))
            u, v, w = target_indices[idx]
            
            # Set the primary contribution
            U[u, c] = 1
            V[v, c] = 1
            W[w, c] = 1
            
            # Add some random noise to allow for cancellation/merging
            if np.random.random() < 0.3:
                # Add a random small integer to another position
                u2 = np.random.randint(0, nm)
                v2 = np.random.randint(0, mp)
                w2 = np.random.randint(0, pn)
                U[u2, c] += np.random.choice([-1, 1])
                V[v2, c] += np.random.choice([-1, 1])
                W[w2, c] += np.random.choice([-1, 1])
                
        # Quick check: compute T and compare only at target indices
        T = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
        
        # Check non-zero pattern
        valid = True
        for (u, v, w) in target_indices:
            if T[u, v, w] != 1.0:
                valid = False
                break
                
        if valid:
            # Also check no other entries are non-zero
            # This is expensive, so do it only if the first check passes
            if np.allclose(T, T_target, atol=1e-9):
                return U.tolist(), V.tolist(), W.tolist()
            
        attempts += 1
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a rank 71 decomposition
    search_time = DEADLINE - 10
    
    # Try rank 71
    result = _random_search(n, m, p, 71, search_time / 2)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Try rank 70
    remaining_time = DEADLINE - 10 - (time.time() - start_time)
    if remaining_time > 5:
        result = _random_search(n, m, p, 70, remaining_time)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
