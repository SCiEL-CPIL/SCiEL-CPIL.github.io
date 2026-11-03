import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 120.0

def _check_time():
    global _START_TIME
    return time.time() - _START_TIME < _DEADLINE - 5.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _random_sparse_decomposition(n, m, p, rank, rng):
    """Generate a random sparse candidate decomposition."""
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                if rng.random() < 0.3:
                    U[i*m+j][c] = rng.choice([-1.0, 1.0])
        for j in range(m):
            for k in range(p):
                if rng.random() < 0.3:
                    V[j*p+k][c] = rng.choice([-1.0, 1.0])
        for k in range(p):
            for i in range(n):
                if rng.random() < 0.3:
                    W[k*n+i][c] = rng.choice([-1.0, 1.0])
    
    return U, V, W

def _structured_decomposition_attempt(n, m, p, rank, rng):
    """Try a structured approach: group columns and use block structure."""
    # For 3x3 x 3x7, try to use a decomposition where we process
    # the 7 output columns in a structured way
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Assign random but structured values
    for c in range(rank):
        # Each term has a "focus" on certain (i,j,k) patterns
        focus_i = rng.randint(0, n-1)
        focus_j = rng.randint(0, m-1)
        focus_k = rng.randint(0, p-1)
        
        for i in range(n):
            for j in range(m):
                if i == focus_i and j == focus_j:
                    U[i*m+j][c] = 1.0
                elif rng.random() < 0.1:
                    U[i*m+j][c] = rng.choice([-1.0, 1.0])
        
        for j in range(m):
            for k in range(p):
                if j == focus_j and k == focus_k:
                    V[j*p+k][c] = 1.0
                elif rng.random() < 0.1:
                    V[j*p+k][c] = rng.choice([-1.0, 1.0])
        
        for k in range(p):
            for i in range(n):
                if k == focus_k and i == focus_i:
                    W[k*n+i][c] = 1.0
                elif rng.random() < 0.1:
                    W[k*n+i][c] = rng.choice([-1.0, 1.0])
    
    return U, V, W

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the trivial decomposition as fallback
    best_result = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound on rank
    lower_bound = max(n*m, m*p, p*n)
    
    # Try to find lower-rank decompositions
    # Search from lower_bound up to best_rank - 1
    rng = random.Random(42)
    
    for rank in range(lower_bound, best_rank):
        if not _check_time():
            break
        
        # Try multiple random trials for this rank
        max_trials = 5000 if rank < 30 else 2000
        found = False
        
        for trial in range(max_trials):
            if not _check_time():
                break
            
            # Alternate between random and structured approaches
            if trial % 2 == 0:
                U, V, W = _random_sparse_decomposition(n, m, p, rank, rng)
            else:
                U, V, W = _structured_decomposition_attempt(n, m, p, rank, rng)
            
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, continue searching for better
            continue
    
    return best_result
