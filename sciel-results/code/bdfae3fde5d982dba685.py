"""Matrix multiplication 2,4,7 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # safety margin before 150s wall-clock limit

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy operations."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _try_strassen_like(n, m, p, rank, max_iters=5000, tol=1e-9):
    """
    Try to find a Strassen-like decomposition with the given rank.
    Uses a structured approach: assume a pattern and solve for coefficients.
    """
    # For now, this is a placeholder for a more sophisticated approach
    # We'll use random search with structure
    return None

def _random_search(n, m, p, rank, n_trials=100, seed=42):
    """Try random rank-r decompositions with entries from {-1, 0, 1}."""
    rng = np.random.RandomState(seed)
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for trial in range(n_trials):
        # Generate random sparse matrices with entries in {-1, 0, 1}
        U = rng.choice([-1, 0, 1], size=(U_rows, rank))
        V = rng.choice([-1, 0, 1], size=(V_rows, rank))
        W = rng.choice([-1, 0, 1], size=(W_rows, rank))
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _structured_search(n, m, p, rank, n_trials=100):
    """Try structured random decompositions that respect the tensor structure."""
    rng = np.random.RandomState(123)
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for trial in range(n_trials):
        # Generate with some structure: each row has a few non-zero entries
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        
        for i in range(U_rows):
            nnz = max(1, int(rank * 0.3))
            cols = rng.choice(rank, size=min(nnz, rank), replace=False)
            for c in cols:
                U[i, c] = rng.choice([-1, 1])
        
        for j in range(V_rows):
            nnz = max(1, int(rank * 0.3))
            cols = rng.choice(rank, size=min(nnz, rank), replace=False)
            for c in cols:
                V[j, c] = rng.choice([-1, 1])
        
        for k in range(W_rows):
            nnz = max(1, int(rank * 0.3))
            cols = rng.choice(rank, size=min(nnz, rank), replace=False)
            for c in cols:
                W[k, c] = rng.choice([-1, 1])
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _greedy_search(n, m, p, rank, max_trials=2000):
    """
    Greedy approach: start with a base decomposition and try to modify it.
    """
    # This is a placeholder for a more sophisticated greedy approach
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook as baseline
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_rank = n * m * p  # 56
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds:
    # For 2x4x7: lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    
    # Try ranks from just below schoolbook downward
    ranks_to_try = list(range(55, 27, -1))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Random search
        result = _random_search(n, m, p, rank, n_trials=50, seed=rank * 1000)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
        
        # Structured search
        if time.time() - start_time > DEADLINE - 15:
            break
        result = _structured_search(n, m, p, rank, n_trials=30)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists for the expected interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
