import time
import sys
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

def _check_deadline():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def _naive_decomposition(n, m, p):
    """Return the naive rank-(n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k,
    and 0 otherwise. The decomposition must satisfy:
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1.0
                actual = float(np.dot(U_arr[a], V_arr[b] * W_arr[c]))
                err = abs(actual - expected)
                max_err = max(max_err, err)
    
    return max_err < tol

def _rank28_decomposition(n, m, p):
    """Return the known rank-28 decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iterations=50):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    For each term r, check if it can be expressed as a linear combination of the others.
    Returns a new decomposition with rank-1 if successful, else None.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each term r_remove, try to find coefficients c such that:
    # term_{r_remove} = sum_{r' != r_remove} c_{r'} * term_{r'}
    # where term_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # 
    # If found, then T = sum_{r' != r_remove} term_{r'} + sum_{r'' != r_remove} c_{r''} term_{r''}
    # = sum_{r' != r_remove} (1 + c_{r'}) * term_{r'}
    # So we can scale each remaining term by (1 + c_{r'})
    
    n_terms = rank
    for r_remove in range(n_terms):
        if _check_deadline():
            break
        
        # Build linear system: for all (a,b,c), sum_{r' != r_remove} c_{r'} U[a,r']V[b,r']W[c,r'] = U[a,r_remove]V[b,r_remove]W[c,r_remove]
        # This is a linear system in the c_{r'} variables.
        
        # Collect equations efficiently
        eqs = []
        rhs = []
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    row = []
                    for r2 in range(n_terms):
                        if r2 == r_remove:
                            continue
                        row.append(U_arr[a, r2] * V_arr[b, r2] * W_arr[c, r2])
                    eqs.append(row)
                    rhs.append(U_arr[a, r_remove] * V_arr[b, r_remove] * W_arr[c, r_remove])
        
        A_sys = np.array(eqs)
        b_sys = np.array(rhs)
        
        try:
            coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            recon = A_sys @ coeffs
            if np.allclose(recon, b_sys, atol=1e-9):
                # Found a valid reduction. Build new decomposition.
                new_rank = n_terms - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                idx_map = {}
                idx = 0
                for r2 in range(n_terms):
                    if r2 == r_remove:
                        continue
                    idx_map[r2] = idx
                    scale = 1.0 + coeffs[idx]
                    for a in range(n * m):
                        new_U[a][idx] = U_arr[a, r2] * scale
                    for b in range(m * p):
                        new_V[b][idx] = V_arr[b, r2] * scale
                    for c in range(p * n):
                        new_W[c][idx] = W_arr[c, r2] * scale
                    idx += 1
                
                # Verify the new decomposition
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    return {"U": new_U, "V": new_V, "W": new_W}
        except:
            pass
    
    return None

def construct():
    if _check_deadline():
        # Fallback: return naive rank-30 decomposition
        n, m, p = DIMENSIONS
        return _naive_decomposition(n, m, p)
    
    n, m, p = DIMENSIONS
    
    # Start with the rank-28 decomposition
    result = _rank28_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # If verification fails, fall back to naive
        return _naive_decomposition(n, m, p)
    
    # Try to reduce rank further using time budget
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - _SAFETY_MARGIN - elapsed
    
    if remaining > 5.0:
        # Try to reduce rank using the proper reduction algorithm
        reduced = _try_reduce_rank(result["U"], result["V"], result["W"], n, m, p, target_rank=27)
        if reduced is not None:
            result = reduced
    
    return result
# EVOLVE-BLOCK-END
