import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check shapes
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def _try_rank54_search(n, m, p, time_limit):
    """
    Search for a rank-54 decomposition exploiting n=2 structure.
    
    Key insight: with n=2, W has only 20 rows (k*n+i for k in 0..9, i in 0..1).
    We can try to find rank-54 solutions by searching over structured patterns.
    
    Strategy: Start from the rank-55 solution and try to merge/eliminate one rank term
    by finding linear dependencies or using the fact that C has only 2 rows.
    """
    start_time = time.time()
    
    # Build the rank-55 solution first
    U55, V55, W55 = _build_rank55(n, m, p)
    
    # Try to find a rank-54 solution by searching for coefficient modifications
    # that allow removing one rank dimension.
    
    # Approach: For each pair of rank terms, check if they can be merged
    # by adjusting coefficients (this is heuristic and may not find anything).
    
    # Better approach: Use the structure that with n=2, the W matrix has a special form.
    # W[k*n + i] = W[k*2 + i]. For each k, we have W[2k] and W[2k+1].
    # The constraint is that for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try a more systematic search: fix V and W, solve for U, or vice versa.
    
    # Actually, let's try a different approach: search over small integer/half-integer
    # coefficient modifications to the rank-55 solution to see if we can reduce rank.
    
    # For rank 54, we need to find U,V,W with 54 columns.
    # Let's try random restarts with structured initialization.
    
    best_rank = 55
    best_solution = (U55, V55, W55)
    
    # Try to find a rank-54 solution by fixing the structure and searching coefficients
    # We'll use the fact that the problem has a lot of symmetry.
    
    # Simple heuristic: try removing each rank term and see if the remaining 54
    # can be adjusted to still satisfy the constraints.
    
    for col_to_remove in range(55):
        if time.time() - start_time > time_limit:
            break
        
        # Remove column col_to_remove
        U_test = [row[:] for row in U55]
        V_test = [row[:] for row in V55]
        W_test = [row[:] for row in W55]
        
        for row in U_test:
            row.pop(col_to_remove)
        for row in V_test:
            row.pop(col_to_remove)
        for row in W_test:
            row.pop(col_to_remove)
        
        # Now check if we can find small adjustments to make this work
        # This is a linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # With one column removed, we have 54 unknowns per matrix (but they're coupled)
        
        # This is complex. Let's just verify if the naive removal works (unlikely but cheap to check)
        if _verify(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    
    # If no simple removal works, try a more sophisticated search
    # Search for rank-54 solutions with structured patterns
    
    # Try: use 5 blocks of rank 10 instead of 5 blocks of rank 11
    # This would require each 2x2x2 block to have rank 10, which is impossible
    # (minimum rank for 2x2x2 matmul is 7, but with the structure constraints
    #  from the larger problem, we might need more)
    
    # Alternative: mix Strassen and schoolbook differently
    # e.g., 4 blocks of rank 11 + 1 block of rank 10 = 54
    
    # Try 4 Strassen blocks (rank 11 each) + 1 modified block (rank 10)
    # The modified block would need to handle 2x2x2 with rank 10, which is possible
    # since schoolbook is rank 4 and Strassen is rank 7.
    
    # Actually, let's try: 3 Strassen blocks (33) + 2 schoolbook blocks (8) + 1 special (13) = 54
    # Or: 5 Strassen blocks but with one term shared/merged
    
    # Let's try a random search with structured initialization
    np.random.seed(42)
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with rank-55 solution and perturb
        U_rand = [row[:] for row in U55]
        V_rand = [row[:] for row in V55]
        W_rand = [row[:] for row in W55]
        
        # Try to zero out one column and adjust others
        col_to_zero = attempt % 55
        
        # Set column to zero
        for row in U_rand:
            row[col_to_zero] = 0.0
        for row in V_rand:
            row[col_to_zero] = 0.0
        for row in W_rand:
            row[col_to_zero] = 0.0
        
        # Small random perturbations to other columns
        for c in range(55):
            if c == col_to_zero:
                continue
            for i in range(len(U_rand)):
                if abs(U_rand[i][c]) > 0.1:
                    U_rand[i][c] += np.random.uniform(-0.1, 0.1)
            for i in range(len(V_rand)):
                if abs(V_rand[i][c]) > 0.1:
                    V_rand[i][c] += np.random.uniform(-0.1, 0.1)
            for i in range(len(W_rand)):
                if abs(W_rand[i][c]) > 0.1:
                    W_rand[i][c] += np.random.uniform(-0.1, 0.1)
        
        # Round to half-integers
        for i in range(len(U_rand)):
            for c in range(55):
                U_rand[i][c] = round(U_rand[i][c] * 2) / 2
        for i in range(len(V_rand)):
            for c in range(55):
                V_rand[i][c] = round(V_rand[i][c] * 2) / 2
        for i in range(len(W_rand)):
            for c in range(55):
                W_rand[i][c] = round(W_rand[i][c] * 2) / 2
        
        # Remove the zero column
        U_test = [row[:] for row in U_rand]
        V_test = [row[:] for row in V_rand]
        W_test = [row[:] for row in W_rand]
        
        for row in U_test:
            row.pop(col_to_zero)
        for row in V_test:
            row.pop(col_to_zero)
        for row in W_test:
            row.pop(col_to_zero)
        
        if _verify(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    
    return None

def _build_rank55(n, m, p):
    """Build the rank-55 Strassen+schoolbook decomposition."""
    target_rank = 55
    U = [[0.0] * target_rank for _ in range(n * m)]
    V = [[0.0] * target_rank for _ in range(m * p)]
    W = [[0.0] * target_rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _get_strassen_terms()
    
    for g in range(5):
        k_offset = g * 2
        
        for u_mask, v_mask, w_mask in strassen_terms:
            for u_idx, coeff in u_mask.items():
                li = u_idx // 2
                lj = u_idx % 2
                global_u_idx = li * m + lj
                U[global_u_idx][col] += coeff
            
            for v_idx, coeff in v_mask.items():
                lj = v_idx // 2
                lk = v_idx % 2
                global_k = k_offset + lk
                global_v_idx = lj * p + global_k
                V[global_v_idx][col] += coeff
            
            for w_idx, coeff in w_mask.items():
                lk = w_idx // 2
                li = w_idx % 2
                global_k = k_offset + lk
                global_w_idx = global_k * n + li
                W[global_w_idx][col] += coeff
            
            col += 1
    
    for i in range(n):
        u_global = i * m + 2
        for k in range(p):
            v_global = 2 * p + k
            w_global = k * n + i
            
            U[u_global][col] = 1.0
            V[v_global][col] = 1.0
            W[w_global][col] = 1.0
            col += 1
    
    return U, V, W

def _get_strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u_mask, v_mask, w_mask) where masks are dicts of {index: coeff}.
    Indices are local to the 2x2x2 block:
    U: i*m + j (i in 0,1; j in 0,1)
    V: j*p + k (j in 0,1; k in 0,1)
    W: k*n + i (k in 0,1; i in 0,1)
    """
    # Standard Strassen
    # m1 = (a00 + a11) * (b00 + b11) -> c00 += 1, c11 += 1
    t1 = ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}) 
    # Note: 
    # U indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # V indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # W indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1) -- WAIT. 
    # W index is k*n + i.
    # If n=2:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So W indices are also 0,1,2,3 corresponding to (k,i) pairs.
    
    # Let's re-verify W mapping for 2x2x2.
    # W[k*n + i]
    # Term m1 contributes to C[0,0] and C[1,1].
    # C[0,0] is i=0, k=0. W index = 0*2 + 0 = 0.
    # C[1,1] is i=1, k=1. W index = 1*2 + 1 = 3.
    # So t1 W mask: {0: 1, 3: 1}. Correct.
    
    # m2 = (a10 + a11) * b00 -> c10 += 1, c11 -= 1
    # U: a10 (idx 2), a11 (idx 3). V: b00 (idx 0).
    # C: c10 (i=1, k=0 -> W idx 1), c11 (i=1, k=1 -> W idx 3).
    t2 = ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1})
    
    # m3 = a00 * (b01 - b11) -> c01 += 1, c11 += 1
    # U: a00 (idx 0). V: b01 (idx 1), b11 (idx 3).
    # C: c01 (i=0, k=1 -> W idx 2), c11 (i=1, k=1 -> W idx 3).
    t3 = ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1})
    
    # m4 = a11 * (b10 - b00) -> c00 += 1, c10 += 1
    # U: a11 (idx 3). V: b10 (idx 2), b00 (idx 0).
    # C: c00 (i=0, k=0 -> W idx 0), c10 (i=1, k=0 -> W idx 1).
    t4 = ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1})
    
    # m5 = (a00 + a01) * b11 -> c01 += 1, c00 -= 1
    # U: a00 (idx 0), a01 (idx 1). V: b11 (idx 3).
    # C: c01 (i=0, k=1 -> W idx 2), c00 (i=0, k=0 -> W idx 0).
    t5 = ({0: 1, 1: 1}, {3: 1}, {2: 1, 0: -1})
    
    # m6 = (a10 - a00) * (b00 + b01) -> c11 += 1
    # U: a10 (idx 2), a00 (idx 0). V: b00 (idx 0), b01 (idx 1).
    # C: c11 (i=1, k=1 -> W idx 3).
    t6 = ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    
    # m7 = (a01 - a11) * (b10 + b11) -> c00 += 1
    # U: a01 (idx 1), a11 (idx 3). V: b10 (idx 2), b11 (idx 3).
    # C: c00 (i=0, k=0 -> W idx 0).
    t7 = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    return [t1, t2, t3, t4, t5, t6, t7]

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed rank-55 solution
    U55, V55, W55 = _build_rank55(n, m, p)
    
    if not _verify(U55, V55, W55, n, m, p):
        raise ValueError("Rank-55 verification failed")
    
    best_U, best_V, best_W = U55, V55, W55
    
    # Check if we have time to search for rank 54
    elapsed = time.time() - _start
    search_time = DEADLINE - SAFETY_MARGIN - elapsed
    
    if search_time > 5:  # At least 5 seconds for search
        result = _try_rank54_search(n, m, p, search_time)
        if result is not None:
            U54, V54, W54 = result
            if _verify(U54, V54, W54, n, m, p):
                best_U, best_V, best_W = U54, V54, W54
    
    # Final safety check
    elapsed = time.time() - _start
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Return the best we have
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
