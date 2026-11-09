"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import itertools
import numpy as np
from numpy.linalg import matrix_rank, solve, lstsq

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal (or best found) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, provide the schoolbook decomposition as a guaranteed valid result
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Strategy 1: Try ALS-based search for lower rank
    if time.time() < deadline:
        for target_rank in range(best_rank - 1, max(best_rank - 20, 55), -1):
            if time.time() > deadline - 10:
                break
            result = _try_als_decomposition(n, m, p, target_rank, deadline)
            if result is not None:
                rank = len(result['U'][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
                break  # Found a valid decomposition, stop
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _try_structured_decomposition(n, m, p):
    """
    Try a structured decomposition that exploits the block structure.
    """
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The matrix multiplication tensor has entries T[i,j,k] = 1 for all valid
    i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition means:
    T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This must equal 1 for all valid i, j, k.
    """
    rank = len(U[0])
    # Build numpy arrays for efficient verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over all rank terms
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_rank_reduction_search(n, m, p, base_result, base_rank):
    """
    Try to reduce the rank of the base decomposition by finding linear dependencies.
    """
    return None


def _try_randomized_search(n, m, p, current_best_rank):
    """
    Try randomized search for a lower-rank decomposition.
    """
    return None


def _try_als_decomposition(n, m, p, target_rank, deadline):
    """
    Try to find a rank-r decomposition using Alternating Least Squares.
    """
    import numpy as np
    
    r = target_rank
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute index arrays
    idx_i_U = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    idx_j_U = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    idx_k_U = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    for trial in range(15):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + trial)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        for iteration in range(400):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    V_block = V[j*p:(j+1)*p, :]  # (p, r)
                    W_idx = np.array([k*n+i for k in range(p)])
                    W_block = W[W_idx, :]  # (p, r)
                    M = V_block * W_block  # (p, r)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    U_idx = np.array([i*m+j for i in range(n)])
                    U_block = U[U_idx, :]  # (n, r)
                    W_block = W[k*n:(k+1)*n, :]  # (n, r)
                    M = U_block * W_block  # (n, r)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    U_idx = np.array([i*m+j for j in range(m)])
                    U_block = U[U_idx, :]  # (m, r)
                    V_idx = np.array([j*p+k for j in range(m)])
                    V_block = V[V_idx, :]  # (m, r)
                    M = U_block * V_block  # (m, r)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-7:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


# EVOLVE-BLOCK-END
