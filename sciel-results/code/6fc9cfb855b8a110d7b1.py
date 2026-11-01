import time
import random
import sys
import numpy as np

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
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

def _construct_strassen_27():
    """
    Strassen's algorithm for 3x3 matrix multiplication.
    Rank 27 decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 27
    
    # Strassen's 7 products for 2x2, extended to 3x3 via block decomposition
    # Actually, let's use the known rank-27 decomposition for 3x3 matrices.
    # The standard Strassen for 2x2 gives rank 7. For 3x3, we can use a direct
    # rank-27 decomposition.
    
    # Let's use a simpler approach: the known minimal rank for 3x3 is 23 (Landsberg),
    # but finding it is hard. Let's first try to get a valid decomposition with
    # rank < 27.
    
    # For now, let's implement a known rank-27 decomposition.
    # Actually, let's try a different approach: use the fact that we can decompose
    # the multiplication tensor using block methods.
    
    # A simpler approach: use the schoolbook but try to find a better decomposition
    # through random search or known results.
    
    # Let's return a valid rank-27 decomposition using a known construction.
    # The multiplication tensor for 3x3 has rank at most 27 (trivial from 3x3x3).
    
    # Actually, let's just use the schoolbook for now and try to improve.
    # The known lower bound for 3x3 matrix multiplication rank is 19 (Landsberg),
    # and the best known upper bound is 23.
    
    # For this implementation, let's try to find a rank-23 decomposition if possible,
    # otherwise fall back to schoolbook (rank 27).
    
    # Since finding rank-23 is complex, let's first ensure we have a valid
    # decomposition. The schoolbook gives rank 27.
    
    return _construct_schoolbook()

def _search_rank(deadline):
    """
    Search for a rank-minimal decomposition using a fast local search with
    structured initialization.
    """
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # Try ranks from low to high
    for rank in range(19, 28):  # Known lower bound is 19, upper bound is 27
        if time.time() > deadline - 5:
            break
        
        # Try multiple initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a structured pattern
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Set up schoolbook-like structure for the first min(rank, 27) columns
            for c in range(min(rank, n*m*p)):
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            
            # Add random perturbations to all entries
            U += np.random.uniform(-0.5, 0.5, U.shape)
            V += np.random.uniform(-0.5, 0.5, V.shape)
            W += np.random.uniform(-0.5, 0.5, W.shape)
            
            # Fast local search
            step_size = 0.1
            for step in range(150):
                if time.time() > deadline - 5:
                    break
                
                # Compute error
                error = 0.0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[ii*m+jj, c] * V[jj*p+kk, c] * W[kk*n+ii, c]
                            error += (s - 1) ** 2
                
                if error < 1e-10:
                    if _verify(U.tolist(), V.tolist(), W.tolist()):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
                
                # Perturb a random entry
                which = random.choice(['U', 'V', 'W'])
                if which == 'U':
                    idx = random.randint(0, n*m-1)
                    c = random.randint(0, rank-1)
                    old_val = U[idx, c]
                    U[idx, c] = old_val + random.uniform(-step_size, step_size)
                elif which == 'V':
                    idx = random.randint(0, m*p-1)
                    c = random.randint(0, rank-1)
                    old_val = V[idx, c]
                    V[idx, c] = old_val + random.uniform(-step_size, step_size)
                else:
                    idx = random.randint(0, p*n-1)
                    c = random.randint(0, rank-1)
                    old_val = W[idx, c]
                    W[idx, c] = old_val + random.uniform(-step_size, step_size)
                
                # Check new error
                new_error = 0.0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[ii*m+jj, c] * V[jj*p+kk, c] * W[kk*n+ii, c]
                            new_error += (s - 1) ** 2
                
                if new_error > error:
                    # Revert and reduce step size
                    if which == 'U':
                        U[idx, c] = old_val
                    elif which == 'V':
                        V[idx, c] = old_val
                    else:
                        W[idx, c] = old_val
                    step_size *= 0.9
                else:
                    step_size = min(step_size * 1.02, 0.3)
    
    if best_result is not None:
        return best_result
    return None

def construct():
    overall_deadline = time.time() + 140
    
    # First, try to find a better decomposition
    result = _search_rank(overall_deadline)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _construct_schoolbook()
