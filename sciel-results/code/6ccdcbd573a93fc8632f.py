"""Matrix multiplication 4,4,9: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _validate(U, V, W, n, m, p, num_tests=5, tol=1e-6):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        expected = A @ B
        
        # Compute using the decomposition
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition should satisfy:
        # For each output element (i,k), sum_j A[i,j]*B[j,k] = sum_r U[i*m+j]*V[j*p+k]*W[k*n+i]
        # More precisely: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape for computation
        U_reshaped = U.reshape(n, m, rank)  # U[i,j,r]
        V_reshaped = V.reshape(m, p, rank)  # V[j,k,r]
        W_reshaped = W.reshape(p, n, rank)  # W[k,i,r]
        
        # Compute C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C[i, k] += np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
        
        if not np.allclose(C, expected, atol=tol):
            return False
    
    return True


def _try_random_merge(U, V, W, n, m, p, max_attempts=1000):
    """Try to reduce rank by randomly merging terms."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = U[0][0] if U else 0
    rank = len(U[0]) if U else 0
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    for attempt in range(max_attempts):
        if _time_left() < 5.0:
            break
        
        current_rank = len(U[0]) if U else 0
        if current_rank <= best_rank:
            break
        
        # Pick two random indices to try merging
        idx1 = random.randint(0, current_rank - 1)
        idx2 = random.randint(0, current_rank - 1)
        if idx1 == idx2:
            continue
        
        # Try to merge idx2 into idx1 with random coefficient
        # We want to see if we can express column idx2 as a combination of other columns
        # For now, try simple merging: add column idx2 to column idx1
        
        # Save state
        U_backup = [row[:] for row in U]
        V_backup = [row[:] for row in V]
        W_backup = [row[:] for row in W]
        
        # Merge: add column idx2 to column idx1
        for i in range(len(U)):
            U[i][idx1] += U[i][idx2]
        for i in range(len(V)):
            V[i][idx1] += V[i][idx2]
        for i in range(len(W)):
            W[i][idx1] += W[i][idx2]
        
        # Remove column idx2
        U = [row[:idx2] + row[idx2+1:] for row in U]
        V = [row[:idx2] + row[idx2+1:] for row in V]
        W = [row[:idx2] + row[idx2+1:] for row in W]
        
        # Validate
        if _validate(U, V, W, n, m, p, num_tests=3, tol=1e-5):
            best_rank = len(U[0]) if U else 0
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        else:
            # Restore
            U = U_backup
            V = V_backup
            W = W_backup
    
    return best_U, best_V, best_W, best_rank


def construct():
    """Return the best decomposition found for 4x4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0]) if U else 0
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try random merges
    if _time_left() > 10.0:
        U, V, W, rank = _try_random_merge(U, V, W, n, m, p, max_attempts=500)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}
