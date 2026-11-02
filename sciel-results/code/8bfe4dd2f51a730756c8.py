"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.

Uses a greedy constructive approach to build a low-rank decomposition:
  - Start with all 96 tensor entries needing to equal 1
  - Iteratively add rank-1 terms (u, v, w) that satisfy many unsatisfied constraints
  - Use integer/half-integer entries
  - Verify exactness at the end

Falls back to schoolbook (rank 96) if no better solution is found.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify exact tensor decomposition.
    For all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    """
    if n is None:
        n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _greedy_construct(n=3, m=4, p=8, max_rank=95, time_budget=120):
    """
    Greedy construction: build rank-1 terms to satisfy tensor entries.
    
    Each rank-1 term (u, v, w) contributes u[i*m+j] * v[j*p+k] * w[k*n+i]
    to the (i,j,k) entry. We need the sum over all terms to equal 1.
    
    Strategy: 
    - Maintain residual R[i,j,k] = 1 - current_sum
    - At each step, find (u, v, w) from a candidate set that best reduces
      the residual (e.g., minimizes sum of squared residuals)
    - Use half-integer entries in {-1, -0.5, 0, 0.5, 1}
    """
    deadline = time.time() + time_budget
    
    # Initialize residual to 1 for all entries
    # R[i,j,k] = 1
    n_rows = n * m
    n_cols = m * p
    n_out = p * n
    
    # We'll track the running sum for each (i,j,k)
    # There are n*m*p = 96 entries
    residuals = np.ones((n, m, p), dtype=float)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Candidate vectors: sparse half-integer vectors
    # For U: length 12, for V: length 32, for W: length 24
    # Generate a set of candidate vectors
    
    def gen_candidates(length, max_nonzero=4, values=[-1, -0.5, 0.5, 1]):
        """Generate candidate vectors with at most max_nonzero non-zero entries."""
        candidates = []
        # Include zero vector
        candidates.append(np.zeros(length))
        # Include unit vectors
        for idx in range(length):
            for val in values:
                v = np.zeros(length)
                v[idx] = val
                candidates.append(v.copy())
        # Include pairs
        if length <= 32:
            for idx1 in range(length):
                for idx2 in range(idx1 + 1, length):
                    for val1 in values:
                        for val2 in values:
                            v = np.zeros(length)
                            v[idx1] = val1
                            v[idx2] = val2
                            candidates.append(v.copy())
        return candidates
    
    # Limit candidate set size for efficiency
    U_cands = gen_candidates(n_rows, max_nonzero=2)
    V_cands = gen_candidates(n_cols, max_nonzero=2)
    W_cands = gen_candidates(n_out, max_nonzero=2)
    
    # For efficiency, limit the number of candidates
    max_cands = 200
    if len(U_cands) > max_cands:
        import random
        random.shuffle(U_cands)
        U_cands = U_cands[:max_cands]
    if len(V_cands) > max_cands:
        import random
        random.shuffle(V_cands)
        V_cands = V_cands[:max_cands]
    if len(W_cands) > max_cands:
        import random
        random.shuffle(W_cands)
        W_cands = W_cands[:max_cands]
    
    U_cands = np.array(U_cands)  # (nU, n_rows)
    V_cands = np.array(V_cands)  # (nV, n_cols)
    W_cands = np.array(W_cands)  # (nW, n_out)
    
    rank = 0
    while rank < max_rank and time.time() < deadline:
        # Find best (u, v, w) to add
        best_score = -1e10
        best_u, best_v, best_w = None, None, None
        
        # Subsample for speed
        nU_sub = min(len(U_cands), 50)
        nV_sub = min(len(V_cands), 50)
        nW_sub = min(len(W_cands), 50)
        
        for ui in range(nU_sub):
            u = U_cands[ui]
            if np.all(u == 0):
                continue
            for vi in range(nV_sub):
                v = V_cands[vi]
                if np.all(v == 0):
                    continue
                # Compute contribution of u*v to each (i,j,k)
                # contrib[i,j,k] = u[i*m+j] * v[j*p+k]
                contrib = np.zeros((n, m, p), dtype=float)
                for i in range(n):
                    for j in range(m):
                        uval = u[i * m + j]
                        if uval == 0:
                            continue
                        for k in range(p):
                            contrib[i, j, k] = uval * v[j * p + k]
                
                for wi in range(nW_sub):
                    w = W_cands[wi]
                    if np.all(w == 0):
                        continue
                    # Full contribution: contrib * w[k*n+i]
                    full_contrib = np.zeros((n, m, p), dtype=float)
                    for i in range(n):
                        for k in range(p):
                            wval = w[k * n + i]
                            if wval == 0:
                                continue
                            full_contrib[i, :, k] = contrib[i, :, k] * wval
                    
                    # Score: how well does this reduce the residual?
                    # We want to minimize ||residual - full_contrib||^2
                    # = ||residual||^2 - 2*<residual, full_contrib> + ||full_contrib||^2
                    # Since ||residual||^2 is constant, maximize <residual, full_contrib> - 0.5*||full_contrib||^2
                    score = np.dot(residuals, full_contrib) - 0.5 * np.dot(full_contrib, full_contrib)
                    
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None or best_score <= 0:
            break
        
        # Add this term
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Update residual
        for i in range(n):
            for j in range(m):
                uval = best_u[i * m + j]
                if uval == 0:
                    continue
                for k in range(p):
                    vval = best_v[j * p + k]
                    wval = best_w[k * n + i]
                    residuals[i, j, k] -= uval * vval * wval
        
        rank += 1
        
        # Check if we've satisfied all entries
        if np.max(np.abs(residuals)) < 1e-9:
            break
    
    if rank == 0:
        return None
    
    # Build the matrices
    U_mat = np.column_stack(U_cols).tolist()
    V_mat = np.column_stack(V_cols).tolist()
    W_mat = np.column_stack(W_cols).tolist()
    
    # Verify
    if _verify(U_mat, V_mat, W_mat):
        return {"U": U_mat, "V": V_mat, "W": W_mat, "rank": rank}
    
    return None


def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns the lowest-rank valid decomposition found.
    """
    start_time = time.time()
    total_budget = 135  # Safety margin before 150s limit
    
    # Always have schoolbook as fallback
    best = _schoolbook()
    best_rank = 96
    
    # Try greedy construction
    greedy_result = _greedy_construct(time_budget=total_budget - 10)
    if greedy_result is not None:
        if greedy_result["rank"] < best_rank:
            best = greedy_result
            best_rank = greedy_result["rank"]
    
    # Return in the required format
    return {"U": best["U"], "V": best["V"], "W": best["W"]}

# EVOLVE-BLOCK-END
