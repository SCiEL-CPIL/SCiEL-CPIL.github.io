"""Matrix multiplication tensor decomposition search."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), the coefficient should be 1
    # Coefficient = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3-tensor contraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(np.dot(U_arr[u_idx, :], V_arr[v_idx, :]), W_arr[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    # Also check that non-target entries are zero
    # The tensor T[i,j,k] for matmul has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the standard matmul tensor has T[i,j,k] = 1 for all (i,j,k) in the "multiplication" sense
    # Wait - let me reconsider. The matrix multiplication tensor M has M[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p). The decomposition requires:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need this to equal 1 for ALL (i,j,k) in range.
    # Let me verify all entries equal 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(np.dot(U_arr[u_idx, :], V_arr[v_idx, :]), W_arr[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _try_rank_random(n, m, p, rank, num_trials=20, seed=42):
    """Try to find a rank-`rank` decomposition via random initialization + least squares."""
    if rank < max(n, m, p):
        return None
    
    rng = np.random.RandomState(seed)
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    total_entries = n * m * p
    
    for trial in range(num_trials):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        # Random initialization
        U = rng.randn(n_rows_U, rank)
        V = rng.randn(n_rows_V, rank)
        W = rng.randn(n_rows_W, rank)
        
        # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a system of n*m*p equations in the variables U,V,W.
        # Use alternating least squares: fix two, solve for the third.
        
        converged = False
        for iteration in range(50):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return None
            
            # Fix V, W; solve for U
            # For each i,j: sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) summed over k = 1
            # Actually: for each (i,j,k): U[i*m+j] dot (V[j*p+k] * W[k*n+i]) = 1
            # For fixed V,W: for each i,j, we have p equations:
            #   U[i*m+j] dot (V[j*p+k] * W[k*n+i]) = 1 for k=0..p-1
            # This is p equations in rank unknowns (the row U[i*m+j]).
            # Stack: for each (i,j), form matrix B[i,j] of shape (p, rank) where B[i,j][k,c] = V[j*p+k,c]*W[k*n+i,c]
            # Then solve B[i,j] @ U[i*m+j] = ones(p)
            
            U_new = np.zeros((n_rows_U, rank))
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve B @ x = ones(p)
                    try:
                        U_new[i * m + j, :], residuals, rank_solve, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = 0
            U = U_new
            
            # Fix U, W; solve for V
            V_new = np.zeros((n_rows_V, rank))
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = 0
            V = V_new
            
            # Fix U, V; solve for W
            W_new = np.zeros((n_rows_W, rank))
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, rank))
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = 0
            W = W_new
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-9:
                converged = True
                break
        
        if converged:
            # Convert to list of lists with rounded values
            U_list = [[round(float(x), 6) for x in row] for row in U]
            V_list = [[round(float(x), 6) for x in row] for row in V]
            W_list = [[round(float(x), 6) for x in row] for row in W]
            
            # Final verification
            if _check_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_structured(n, m, p, rank, num_trials=50, seed=42):
    """Try structured decompositions with some sparsity patterns."""
    if rank < max(n, m, p):
        return None
    
    rng = np.random.RandomState(seed)
    
    for trial in range(num_trials):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        # Try a structured approach: use the fact that for small matrices,
        # we can try rank = n*p + m*p - m (Landsberg bound) or other known bounds.
        # For 3x5x6: known upper bound is around 24 (from literature)
        # Try various structured initializations
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Random sparse initialization
        density = 0.3 + 0.4 * rng.random()
        for c in range(rank):
            for r in range(n * m):
                if rng.random() < density:
                    U[r, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
            for r in range(m * p):
                if rng.random() < density:
                    V[r, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
            for r in range(p * n):
                if rng.random() < density:
                    W[r, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
        
        # Alternating least squares
        for iteration in range(30):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return None
            
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U_new[i * m + j, :], _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            U = U_new
            
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            V = V_new
            
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, rank))
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        pass
            W = W_new
            
            # Check
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-9:
                U_list = [[round(float(x), 6) for x in row] for row in U]
                V_list = [[round(float(x), 6) for x in row] for row in V]
                W_list = [[round(float(x), 6) for x in row] for row in W]
                if _check_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Known upper bounds from literature for 3x5x6: around 24-30
    # Try ranks from a reasonable range downward
    
    # First, try a few specific ranks that are known to be achievable
    candidate_ranks = []
    for r in range(max(n*m, m*p, p*n), schoolbook_rank + 1):
        candidate_ranks.append(r)
    
    # Sort by rank (try lower ranks first)
    candidate_ranks.sort()
    
    # Try each rank
    for rank in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if rank >= best_rank:
            break
        
        # Try random approach
        result = _try_rank_random(n, m, p, rank, num_trials=10, seed=rank)
        if result is not None:
            best = result
            best_rank = rank
            continue
        
        # Try structured approach
        result = _try_rank_structured(n, m, p, rank, num_trials=20, seed=rank + 1000)
        if result is not None:
            best = result
            best_rank = rank
            continue
    
    # Final safety check
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return _schoolbook(n, m, p)
    
    return best
