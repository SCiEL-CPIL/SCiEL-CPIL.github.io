"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and check it equals the canonical multiplication tensor (which is 1 for all i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_ranked_decomposition(n, m, p, target_rank, max_attempts=50, timeout=10):
    """
    Try to find an exact decomposition of rank target_rank.
    Uses random initialization with structured constraints.
    Returns (U, V, W) as lists or None if not found.
    """
    start_time = time.time()
    
    # The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: use rank-1 tensors that are "separable" in a structured way.
    # A common approach for small matrix multiplication: use tensor rank decompositions
    # based on the Strassen-like patterns or random search.
    
    # For 4x4 x 4x6, the known upper bound is 96 (schoolbook).
    # Let's try to find lower rank decompositions.
    
    best_result = None
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with random values
        U_arr = np.random.randn(n * m, target_rank) * 0.1
        V_arr = np.random.randn(m * p, target_rank) * 0.1
        W_arr = np.random.randn(p * n, target_rank) * 0.1
        
        # Try a few iterations of coordinate descent / Newton-like steps
        # to minimize the error sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        for iteration in range(100):
            if time.time() - start_time > timeout:
                break
            
            # Update U
            for idx in range(n * m):
                i, j = divmod(idx, m)
                # Gradient w.r.t. U[idx, c]: V[j*p+k, c] * W[k*n+i, c]
                for k in range(p):
                    v_row = V_arr[j * p + k]
                    w_row = W_arr[k * n + i]
                    # Current contribution
                    contrib = np.sum(U_arr[idx] * v_row * w_row)
                    # We want contrib to be 1, so gradient = (contrib - 1) * v_row * w_row
                    grad = (contrib - 1.0) * v_row * w_row
                    U_arr[idx] -= 0.01 * grad
            
            # Update V
            for idx in range(m * p):
                j, k = divmod(idx, p)
                for i in range(n):
                    u_row = U_arr[i * m + j]
                    w_row = W_arr[k * n + i]
                    contrib = np.sum(V_arr[idx] * u_row * w_row)
                    grad = (contrib - 1.0) * u_row * w_row
                    V_arr[idx] -= 0.01 * grad
            
            # Update W
            for idx in range(p * n):
                k, i = divmod(idx, n)
                for j in range(m):
                    u_row = U_arr[i * m + j]
                    v_row = V_arr[j * p + k]
                    contrib = np.sum(W_arr[idx] * u_row * v_row)
                    grad = (contrib - 1.0) * u_row * v_row
                    W_arr[idx] -= 0.01 * grad
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-6:
                # Found a solution! Convert to lists
                U_list = [[float(x) for x in row] for row in U_arr]
                V_list = [[float(x) for x in row] for row in V_arr]
                W_list = [[float(x) for x in row] for row in W_arr]
                return U_list, V_list, W_list
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try to find lower rank decompositions
    # Theoretical lower bound for 4x4 x 4x6 matrix multiplication
    # is at least max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But the actual minimum is likely higher.
    
    # Try ranks from a reasonable lower bound up
    # For 4x4 x 4x6, let's try ranks from 30 to 95
    for target_rank in range(30, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_ranked_decomposition(n, m, p, target_rank, 
                                           max_attempts=20, 
                                           timeout=min(5, max(1, deadline - time.time() - 1)))
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                # Continue trying to find even lower rank
                # But we can't go below current best, so break if we want to save time
                # Actually, let's continue to see if we can find even lower
    
    return best_result
# EVOLVE-BLOCK-END
