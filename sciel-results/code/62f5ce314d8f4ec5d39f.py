"""Different strategy: rank-reduction search from schoolbook via term merging."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as numpy arrays."""
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


def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_merge(U, V, W, c1, c2):
    """
    Try to merge columns c1 and c2 into a single column.
    Returns (new_U, new_V, new_W) with one fewer column, or None if infeasible.
    
    The idea: for the merged column c_new, we need:
      U[., c_new] * V[., c_new] * W[., c_new] = U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2]
    
    We try to find alpha, beta such that we can express the sum as a rank-1 term.
    Specifically, try: set new column = (U[:,c1]+U[:,c2], V[:,c1]+V[:,c2], W[:,c1]+W[:,c2])
    and check if the resulting tensor still decomposes correctly.
    
    More generally: we can try linear combinations. For simplicity, try:
      new_U = U[:,c1] + U[:,c2], new_V = V[:,c1] + V[:,c2], new_W = W[:,c1] + W[:,c2]
    This works when the two terms have the same support structure (e.g., same i,j,k pattern).
    
    A better approach: for each pair, check if the sum of the two rank-1 tensors
    can be expressed as a single rank-1 tensor. This is true iff the three vectors
    (U1+U2), (V1+V2), (W1+W2) form a valid rank-1 decomposition of T1+T2.
    T1+T2[i,j,k] = U1[i,j]*V1[j,k]*W1[k,i] + U2[i,j]*V2[j,k]*W2[k,i]
    We need: exists (u,v,w) such that u[i,j]*v[j,k]*w[k,i] = T1+T2 for all (i,j,k).
    
    Practical test: if U1==U2 and V1==V2 and W1==W2 (up to scaling), we can merge.
    Or more generally, try a few linear combinations.
    """
    r = U.shape[1]
    
    # Try simple sum: (U1+U2, V1+V2, W1+W2)
    # The new tensor contribution is:
    # (U1+U2)[i,j] * (V1+V2)[j,k] * (W1+W2)[k,i]
    # = U1*V1*W1 + U1*V1*W2 + U1*V2*W1 + U1*V2*W2 + U2*V1*W1 + U2*V1*W2 + U2*V2*W1 + U2*V2*W2
    # We need this to equal U1*V1*W1 + U2*V2*W2 (the original two terms).
    # This requires the cross terms to vanish, which is a strong condition.
    
    # Alternative: if the two columns are identical (or scalar multiples), merge trivially.
    u1, u2 = U[:, c1], U[:, c2]
    v1, v2 = V[:, c1], V[:, c2]
    w1, w2 = W[:, c1], W[:, c2]
    
    # Check if they are proportional
    # If u1 = a*u2, v1 = b*v2, w1 = c*w2 and a*b*c = 1, then T1 = T2 and we can just keep one.
    # Actually if u1=a*u2, v1=b*v2, w1=c*w2, then T1 = a*b*c * T2.
    # For them to be the same term, need a*b*c=1.
    
    # Try: can we find scalars such that the sum is rank-1?
    # Simplest: if u1==u2, v1==v2, w1==w2: merge by keeping one (or summing W).
    # If u1==u2, v1==v2: then T1+T2 = u1*v1*(w1+w2), which is rank-1.
    
    # Strategy: try several merge patterns
    # Pattern 1: same U and V, different W -> new W = W1+W2
    if np.allclose(u1, u2, atol=1e-12) and np.allclose(v1, v2, atol=1e-12):
        new_U = u1.copy()
        new_V = v1.copy()
        new_W = w1 + w2
        # Verify: new_U[i,j]*new_V[j,k]*new_W[k,i] should equal T1+T2
        # = u1[i,j]*v1[j,k]*(w1[k,i]+w2[k,i]) = u1*v1*w1 + u1*v1*w2 = T1 + T2 (since u1=u2, v1=v2)
        return new_U, new_V, new_W
    
    # Pattern 2: same U and W, different V
    if np.allclose(u1, u2, atol=1e-12) and np.allclose(w1, w2, atol=1e-12):
        return u1.copy(), v1 + v2, w1.copy()
    
    # Pattern 3: same V and W, different U
    if np.allclose(v1, v2, atol=1e-12) and np.allclose(w1, w2, atol=1e-12):
        return u1 + u2, v1.copy(), w1.copy()
    
    # Pattern 4: u1 = a*u2, v1 = b*v2, w1 = c*w2 with a*b*c = 1 (same tensor)
    # Check if proportional
    nz_u1 = np.nonzero(u1)[0]
    nz_u2 = np.nonzero(u2)[0]
    if len(nz_u1) > 0 and len(nz_u2) > 0 and set(nz_u1) == set(nz_u2):
        a = u1[nz_u1[0]] / u2[nz_u2[0]]
        if np.allclose(u1, a * u2, atol=1e-12):
            nz_v1 = np.nonzero(v1)[0]
            nz_v2 = np.nonzero(v2)[0]
            if len(nz_v1) > 0 and len(nz_v2) > 0 and set(nz_v1) == set(nz_v2):
                b = v1[nz_v1[0]] / v2[nz_v2[0]]
                if np.allclose(v1, b * v2, atol=1e-12):
                    nz_w1 = np.nonzero(w1)[0]
                    nz_w2 = np.nonzero(w2)[0]
                    if len(nz_w1) > 0 and len(nz_w2) > 0 and set(nz_w1) == set(nz_w2):
                        c = w1[nz_w1[0]] / w2[nz_w2[0]]
                        if np.allclose(w1, c * w2, atol=1e-12):
                            if abs(a * b * c - 1.0) < 1e-9:
                                # Same tensor, keep one
                                return u1.copy(), v1.copy(), w1.copy()
    
    # Pattern 5: try (u1+u2, v1+v2, w1+w2) and verify it works
    # This is a gamble but worth trying for some structures
    new_u = u1 + u2
    new_v = v1 + v2
    new_w = w1 + w2
    # Quick check: the new term's contribution should equal T1+T2
    # We'll verify after merging, so just return and let the caller verify
    return new_u, new_v, new_w


def _rank_reduction_search(n, m, p, max_time=100):
    """
    Start from schoolbook and try to reduce rank by merging terms.
    Returns best (U, V, W) found.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # In schoolbook, each term c corresponds to a unique (i,j,k) triple.
    # Two terms with the same i but different (j,k) have same U column (only one nonzero at [i*m+j]).
    # Actually in schoolbook, U[i*m+j, c] = 1 only for the specific (i,j) of term c.
    # So two different terms have different U columns (different nonzero positions).
    # The only way to merge is if they share structure.
    
    # In schoolbook, term (i,j,k) has:
    #   U nonzero at position i*m+j
    #   V nonzero at position j*p+k
    #   W nonzero at position k*n+i
    #
    # Two terms (i,j,k1) and (i,j,k2) share the same U column (nonzero at i*m+j).
    # Their V columns differ (j*p+k1 vs j*p+k2) and W columns differ (k1*n+i vs k2*n+i).
    # Pattern 1 (same U, same V) doesn't apply since V differs.
    # Pattern 5 (sum) won't work in general.
    
    # So simple schoolbook doesn't have mergeable pairs via our patterns.
    # We need a smarter initial decomposition or a different merging strategy.
    
    # Alternative: use a block-based initial decomposition that has more structure.
    # Let's try a 3x3 block decomposition (3 blocks of 2x2 in each dimension).
    # Each 2x2x2 block can use Strassen (rank 7) instead of schoolbook (rank 8).
    # Total: 9 blocks * 7 = 63? No, that's not right for 6x6x6.
    
    # Actually for 6x6x6 with 3x3 blocks of size 2:
    # We have 3x3 = 9 block positions for A and B, giving 9*9=81 block products.
    # Each block product is 2x2 * 2x2 -> 2x2, which has rank 8 (schoolbook) or 7 (Strassen).
    # Total rank: 81 * 8 = 648 (schoolbook) or 81 * 7 = 567 (Strassen per block).
    # That's way worse than 216. This approach doesn't help.
    
    # Better: use the 2x2 Strassen at the 6x6 level (split into 3x3 blocks).
    # That gives rank 189 as in the current program.
    # Let me just implement Strassen properly and also try to reduce further.
    
    # For now, let's implement Strassen 2x2 block (rank 189) as the starting point,
    # then try to reduce.
    
    U, V, W = _strassen_2x2(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try random pair merges
    r = U.shape[1]
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if U.shape[1] <= best_rank:
            break
        
        # Pick two random columns to try merging
        c1, c2 = random.sample(range(U.shape[1]), 2)
        
        result = _try_merge(U, V, W, c1, c2)
        if result is not None:
            new_U, new_V, new_W = result
            # Remove the two old columns and add the new one
            keep = [c for c in range(U.shape[1]) if c != c1 and c != c2]
            new_U_full = np.column_stack([new_U[keep], new_U[:, -1:]]) if False else None
            
            # Actually, let's rebuild properly
            # Remove c1 and c2, add new column at end
            keep_mask = np.ones(U.shape[1], dtype=bool)
            keep_mask[c1] = False
            keep_mask[c2] = False
            kept_cols = np.where(keep_mask)[0]
            
            cand_U = np.column_stack([U[:, kept_cols], new_U[:, 0:1]]) if new_U.ndim == 2 else None
            # new_U is a 1D array (single column)
            cand_U = np.hstack([U[:, kept_cols], new_U.reshape(-1, 1)])
            cand_V = np.hstack([V[:, kept_cols], new_V.reshape(-1, 1)])
            cand_W = np.hstack([W[:, kept_cols], new_W.reshape(-1, 1)])
            
            # Verify
            if _verify_np(cand_U, cand_V, cand_W, n, m, p):
                U, V, W = cand_U, cand_V, cand_W
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W


def _strassen_2x2(n, m, p):
    """
    Strassen 2x2 block decomposition for 6x6x6.
    Split into 2x2 blocks of size 3x3.
    Each 3x3*3x3 product uses schoolbook (rank 27).
    7 Strassen steps -> rank 7*27 = 189.
    """
    nh = mh = ph = 3
    block_rank = nh * mh * ph  # 27
    total_rank = 7 * block_rank  # 189
    
    # Strassen steps: (A_block_signs, B_block_signs)
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1: (A00+A11)*(B00+B11)
        ({2: 1, 3: 1}, {0: 1}),              # M2: (A10+A11)*B00
        ({0: 1}, {1: 1, 3: -1}),             # M3: A00*(B01-B11)
        ({3: 1}, {2: 1, 0: -1}),             # M4: A11*(B10-B00)
        ({0: 1, 1: 1}, {3: 1}),              # M5: (A00+A01)*B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}),       # M6: (A10-A00)*(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),       # M7: (A01-A11)*(B10+B11)
    ]
    
    # C block combinations
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},   # C00 = M1 - M4 + M6 - M7
        {2: 1, 4: 1},                  # C01 = M3 + M5
        {1: 1, 3: 1},                  # C10 = M2 + M4
        {0: 1, 1: -1, 2: 1, 6: 1},    # C11 = M1 - M2 + M3 + M7
    ]
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph
                
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k
                            
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * (mh * ph) + j * ph + k
                            
                            U[gi * m + gj, col] = a_sign
                            V[gj * p + gk, col] = b_sign
                            W[gk * n + gi, col] = c_sign
    
    return U, V, W


def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def _verify_decomp(decomp, n, m, p):
    """Verify decomposition from list format."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    return _verify_np(U, V, W, n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # First, get the Strassen decomposition (rank 189)
    U, V, W = _strassen_2x2(n, m, p)
    
    if not _verify_np(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try rank reduction via merging
    if time.time() < deadline - 10:
        U, V, W = _rank_reduction_search(n, m, p, max_time=min(90, deadline - time.time() - 5))
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Return best found
    return _to_lists(best_U, best_V, best_W)

# EVOLVE-BLOCK-END
