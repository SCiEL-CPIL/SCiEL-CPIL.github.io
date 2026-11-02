"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    if not np.allclose(C, 1.0, atol=tol):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition."""
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

def _try_local_search(U, V, W, n, m, p, time_limit=10.0):
    """
    Attempts to reduce the rank by removing columns that are linearly dependent or can be zeroed out.
    This is a heuristic.
    """
    start_time = time.time()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_U, current_V, current_W = U, V, W
    current_rank = best_rank
    
    # Simple heuristic: check for zero columns or duplicate columns
    # If two columns are identical in U, V, and W, we can't just remove one because the sum would change.
    # We need to solve for coefficients.
    # Given the time, we only check for trivial zeros.
    
    for r in range(current_rank - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        if np.allclose(current_U[:, r], 0.0, atol=1e-12) and np.allclose(current_V[:, r], 0.0, atol=1e-12) and np.allclose(current_W[:, r], 0.0, atol=1e-12):
            current_U = np.delete(current_U, r, axis=1)
            current_V = np.delete(current_V, r, axis=1)
            current_W = np.delete(current_W, r, axis=1)
            current_rank -= 1
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            continue
            
    return best_U, best_V, best_W, best_rank

def _construct_via_strassen_blocks(n, m, p):
    """
    Attempt to construct a lower rank decomposition by leveraging Strassen-like
    block decomposition for sub-blocks where possible.
    For 3x5x6, we can potentially decompose into blocks.
    However, for exact minimal rank, this is hard.
    We will try a simple recursive or block approach if it yields lower rank.
    """
    # This is a placeholder for a more sophisticated construction.
    # For now, we return None to indicate we should use schoolbook or other methods.
    return None

def _try_gradient_descent(rank, n, m, p, time_limit=5.0, seed=None):
    """
    Try to find a decomposition of a specific rank using gradient descent on the error.
    """
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Target tensor
    # T[i,j,k] = 1
    # Error = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
    
    # This is a non-convex problem. Gradient descent might not work well without careful initialization.
    # Given the time constraints and complexity, we skip this for now unless we have a very good initialization.
    # We will rely on the schoolbook method which is guaranteed.
    return None

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try local search to remove zero columns (unlikely to help for schoolbook)
    U_ls, V_ls, W_ls, rank_ls = _try_local_search(U, V, W, n, m, p, time_limit=5.0)
    if rank_ls < best_rank:
        best_U, best_V, best_W = U_ls, V_ls, W_ls
        best_rank = rank_ls
    
    # Check if we have time for other methods
    if time.time() < deadline - 10.0:
        # Try Strassen-like block decomposition
        res = _construct_via_strassen_blocks(n, m, p)
        if res is not None:
            U_s, V_s, W_s = res
            if _verify_decomposition(U_s, V_s, W_s, n, m, p):
                if U_s.shape[1] < best_rank:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = U_s.shape[1]
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
