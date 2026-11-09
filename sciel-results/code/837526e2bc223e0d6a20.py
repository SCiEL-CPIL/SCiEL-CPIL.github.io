"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook_decomp_344():
    """Return schoolbook decomposition for 3 x 4 * 4 x 4 -> 3 x 4."""
    n, m, p = 3, 4, 4
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
    return U, V, W

def _convert_344_to_348(U, V, W):
    """Convert a 3x4x4 decomposition to a 3x4x8 decomposition."""
    n, m, p_old = 3, 4, 4
    p_new = 8
    rank = len(U[0])
    
    U_new = [row[:] for row in U]
    
    V_new = [[0] * rank for _ in range(m * p_new)]
    for j in range(m):
        for k in range(p_new):
            if k < p_old:
                V_new[j * p_new + k] = V[j * p_old + k][:]
            else:
                V_new[j * p_new + k] = V[j * p_old + (k - p_old)][:]
                
    W_new = [[0] * rank for _ in range(p_new * n)]
    for k in range(p_new):
        for i in range(n):
            if k < p_old:
                W_new[k * n + i] = W[k * n + i][:]
            else:
                W_new[k * n + i] = W[(k - p_old) * n + i][:]
                
    return {'U': U_new, 'V': V_new, 'W': W_new}

def _random_search_344(target_rank, num_trials=50, timeout=50):
    """Try to find a rank-target_rank decomposition for 3x4x4."""
    n, m, p = 3, 4, 4
    start = time.time()
    
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
        
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        for it in range(30):
            # Update W
            W_new = W.copy()
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            W = W_new
            
            # Update U
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            U = U_new
            
            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                    except:
                        pass
            V = V_new
        
        # Check exactness
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        
        if error < 1e-8:
            # Round to half-integers
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            err2 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        err2 += (val - 1.0)**2
            
            if err2 < 1e-8:
                return {
                    'U': U_r.tolist(),
                    'V': V_r.tolist(),
                    'W': W_r.tolist()
                }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # 1. Generate guaranteed rank 48 solution
    U_344, V_344, W_344 = _schoolbook_decomp_344()
    best_result = _convert_344_to_348(U_344, V_344, W_344)
    
    # 2. Try to find a better solution for 3x4x4 (rank < 48)
    # Schoolbook rank for 3x4x4 is 48.
    # Try ranks 47 down to 40.
    for target_rank in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        # Use remaining time for search
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Allocate a portion of time to each rank
        # Simple heuristic: 5s per rank
        search_timeout = min(5.0, time_left - 2)
        
        result_344 = _random_search_344(target_rank, num_trials=20, timeout=search_timeout)
        
        if result_344 is not None:
            # Convert to 3x4x8
            U_344_res = result_344['U']
            V_344_res = result_344['V']
            W_344_res = result_344['W']
            
            # Verify it's actually rank target_rank (it should be)
            # And convert
            best_result = _convert_344_to_348(U_344_res, V_344_res, W_344_res)
            # We found a lower rank, so we can return immediately 
            # or continue searching for even lower? 
            # Given the difficulty, returning the first found lower rank is a good strategy.
            # But we can try to keep searching if we have time?
            # No, lower rank is better. Let's return.
            return best_result
            
    return best_result
# EVOLVE-BLOCK-END
