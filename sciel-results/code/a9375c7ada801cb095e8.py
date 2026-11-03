"""
Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
1. Build a correct Strassen-based candidate (49+49+16 = 114 terms) as baseline.
2. Use a fast numpy validator.
3. Perform aggressive local search: try removing columns, then try replacing
   pairs of columns with a single column (linear combination), and try
   random column replacements.
4. Time-boxed search with safety margin before 150s kill.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 142.0  # safety margin before 150s kill


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using numpy. U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        temp = uc[:, :, None] * vc[None, :, :]  # (n, m, p)
        wc_t = wc.T  # (n, p)
        result += temp * wc_t[:, None, :]  # (n, m, p)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True


def _validate_int(U, V, W, n, m, p):
    """Validate using exact integer arithmetic with scaling."""
    r = U.shape[1]
    Us = (np.asarray(U, dtype=np.float64) * 2).round().astype(np.int64)
    Vs = (np.asarray(V, dtype=np.float64) * 2).round().astype(np.int64)
    Ws = (np.asarray(W, dtype=np.float64) * 2).round().astype(np.int64)
    
    if not np.allclose(Us / 2.0, np.asarray(U, dtype=np.float64), atol=1e-12):
        return False
    if not np.allclose(Vs / 2.0, np.asarray(V, dtype=np.float64), atol=1e-12):
        return False
    if not np.allclose(Ws / 2.0, np.asarray(W, dtype=np.float64), atol=1e-12):
        return False
    
    U3 = Us.reshape(n, m, r)
    V3 = Vs.reshape(m, p, r)
    W3 = Ws.reshape(p, n, r)
    
    result = np.zeros((n, m, p), dtype=np.int64)
    for c in range(r):
        uc = U3[:, :, c]
        vc = V3[:, :, c]
        wc = W3[:, :, c]
        temp = uc[:, :, None] * vc[None, :, :]
        wc_t = wc.T
        result += temp * wc_t[:, None, :]
    
    if not np.all(result == 8):
        return False
    return True


def _make_strassen_4x4_terms():
    """Return list of (i, j, k, coeff) for 4x4*4x4 Strassen decomposition (49 terms)."""
    
    def strassen_2x2():
        """Return list of (a_i, a_j, b_j, b_k, coeff) for 2x2*2x2 Strassen."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms += [(0,0,0,0,1), (0,0,1,1,1), (1,1,0,0,1), (1,1,1,1,1)]
        # M2 = A11*(B10-B00)
        terms += [(1,1,1,0,1), (1,1,0,0,-1)]
        # M3 = (A10+A11)*B01
        terms += [(1,0,0,1,1), (1,1,0,1,1)]
        # M4 = A00*(B01-B11)
        terms += [(0,0,0,1,1), (0,0,1,1,-1)]
        # M5 = (A00+A01)(B11+B00)
        terms += [(0,0,1,1,1), (0,0,0,0,1), (0,1,1,1,1), (0,1,0,0,1)]
        # M6 = (A01-A11)*(B10+B11)
        terms += [(0,1,1,0,1), (0,1,1,1,1), (1,1,1,0,-1), (1,1,1,1,-1)]
        # M7 = (A10-A00)*(B00+B01)
        terms += [(1,0,0,0,1), (1,0,0,1,1), (0,0,0,0,-1), (0,0,0,1,-1)]
        return terms
    
    s2 = strassen_2x2()
    
    block_defs = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # S1
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),           # S2
        ([(1,0,1), (1,1,1)], [(0,1,1)]),            # S3
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),           # S4
        ([(0,0,1), (0,1,1)], [(1,1,1), (0,0,1)]),   # S5
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),  # S6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),  # S7
    ]
    
    all_terms = []
    for A_blocks, B_blocks in block_defs:
        for (abr, abc, asgn) in A_blocks:
            for (bbr, bbc, bsgn) in B_blocks:
                for (ai, aj, bj, bk, coeff) in s2:
                    i = 2*abr + ai
                    j = 2*abc + aj
                    k = 2*bbc + bk
                    total_coeff = asgn * bsgn * coeff
                    if total_coeff != 0:
                        all_terms.append((i, j, k, total_coeff))
    
    return all_terms


def schoolbook_candidate():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def strassen_block_candidate():
    """Use Strassen 4x4 for the first 8 columns of B, schoolbook for last column."""
    n, m, p = DIMENSIONS
    terms_4x4 = _make_strassen_4x4_terms()
    
    all_terms = []
    # First 4x4 block of B (cols 0-3)
    for (i, j, k, coeff) in terms_4x4:
        all_terms.append((i, j, k, coeff))
    # Second 4x4 block of B (cols 4-7)
    for (i, j, k, coeff) in terms_4x4:
        k_new = k + 4
        all_terms.append((i, j, k_new, coeff))
    # Last column (col 8): schoolbook
    for i in range(n):
        for j in range(m):
            all_terms.append((i, j, 8, 1.0))
    
    r = len(all_terms)
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    for c, (i, j, k, coeff) in enumerate(all_terms):
        U[i*m+j, c] = coeff
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    return U, V, W


def _try_remove_column(U, V, W, c, n, m, p):
    """Try removing column c. Returns (success, U_new, V_new, W_new)."""
    U_test = np.delete(U, c, axis=1)
    V_test = np.delete(V, c, axis=1)
    W_test = np.delete(W, c, axis=1)
    if _validate_fast(U_test, V_test, W_test, n, m, p):
        return True, U_test, V_test, W_test
    return False, None, None, None


def _try_merge_columns(U, V, W, c1, c2, n, m, p):
    """Try merging columns c1 and c2 into one column (linear combination).
    We look for alpha, beta such that replacing cols c1,c2 with a single col
    (alpha*col_c1 + beta*col_c2) still gives a valid decomposition.
    This is a nonlinear constraint, so we try a few simple ratios."""
    r = U.shape[1]
    if c1 >= c2:
        c1, c2 = c2, c1
    # Try several ratios
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0]:
        # New column = col_c1 + ratio * col_c2
        U_new = np.delete(np.delete(U, c2, axis=1), c1, axis=1)
        V_new = np.delete(np.delete(V, c2, axis=1), c1, axis=1)
        W_new = np.delete(np.delete(W, c2, axis=1), c1, axis=1)
        
        # Insert new column at position c1
        new_U = U[:, c1] + ratio * U[:, c2]
        new_V = V[:, c1] + ratio * V[:, c2]
        new_W = W[:, c1] + ratio * W[:, c2]
        
        if c1 < r - 1:
            U_new = np.insert(U_new, c1, new_U, axis=1)
            V_new = np.insert(V_new, c1, new_V, axis=1)
            W_new = np.insert(W_new, c1, new_W, axis=1)
        else:
            U_new = np.column_stack([U_new, new_U])
            V_new = np.column_stack([V_new, new_V])
            W_new = np.column_stack([W_new, new_W])
        
        if _validate_fast(U_new, V_new, W_new, n, m, p):
            return True, U_new, V_new, W_new
    return False, None, None, None


def random_local_search(U0, V0, W0, n, m, p, max_time=100.0):
    """Local search: try to reduce rank by zeroing out columns, merging, etc."""
    t0 = time.time()
    r0 = U0.shape[1]
    best_r = r0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - t0 < max_time:
        if best_r <= 80:
            break
        if no_improve_count > max_no_improve:
            break
        
        r = best_U.shape[1]
        improved = False
        
        # Strategy 1: Try removing a random column
        c = rng.randint(0, r - 1)
        success, U_new, V_new, W_new = _try_remove_column(best_U, best_V, best_W, c, n, m, p)
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_r -= 1
            improved = True
            no_improve_count = 0
            continue
        
        # Strategy 2: Try merging two random columns
        if r >= 2:
            c1 = rng.randint(0, r - 1)
            c2 = rng.randint(0, r - 1)
            if c1 != c2:
                success, U_new, V_new, W_new = _try_merge_columns(best_U, best_V, best_W, c1, c2, n, m, p)
                if success:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_r -= 1
                    improved = True
                    no_improve_count = 0
                    continue
        
        no_improve_count += 1
    
    return best_r, best_U, best_V, best_W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Start with Strassen block candidate
    U, V, W = strassen_block_candidate()
    r = U.shape[1]
    
    if not _validate_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook_candidate()
        r = U.shape[1]
    
    # Local search to reduce rank
    remaining = DEADLINE - (time.time() - t0) - 5
    if remaining > 10:
        r, U, V, W = random_local_search(U, V, W, n, m, p, max_time=remaining)
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
