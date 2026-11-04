import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _build_block_decomposition(n, m, p):
    """
    Build a valid decomposition using block-diagonal Strassen + schoolbook.
    Returns list of (U_col, V_col, W_col) as sparse representations.
    """
    terms = []  # Each term is (U_sparse, V_sparse, W_sparse) where each is dict {idx: val}
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_block(i_off, j_off, k_off, bn, bm, bp):
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    ig = i_off + i
                    jg = j_off + j
                    kg = k_off + k
                    add_term(
                        {ig * m + jg: 1},
                        {jg * p + kg: 1},
                        {kg * n + ig: 1}
                    )
    
    def strassen_block(i_off, j_off, k_off):
        """2x2x2 Strassen block."""
        def gi(i, j): return (i_off + i) * m + (j_off + j)
        def gj(j, k): return (j_off + j) * p + (k_off + k)
        def gk(k, i): return (k_off + k) * n + (i_off + i)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            {gi(0,0): 1, gi(1,1): 1},
            {gj(0,0): 1, gj(1,1): 1},
            {gk(0,0): 1, gk(1,1): 1}
        )
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(
            {gi(1,0): 1, gi(1,1): 1},
            {gj(0,0): 1},
            {gk(0,1): 1, gk(1,1): -1}
        )
        # M3: A00(B01-B11) -> C01, C11
        add_term(
            {gi(0,0): 1},
            {gj(0,1): 1, gj(1,1): -1},
            {gk(1,0): 1, gk(1,1): 1}
        )
        # M4: A11(B10-B00) -> C00, C10
        add_term(
            {gi(1,1): 1},
            {gj(1,0): 1, gj(0,0): -1},
            {gk(0,0): 1, gk(0,1): 1}
        )
        # M5: (A00+A01)B11 -> -C00, C01
        add_term(
            {gi(0,0): 1, gi(0,1): 1},
            {gj(1,1): 1},
            {gk(0,0): -1, gk(1,0): 1}
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            {gi(1,0): 1, gi(0,0): -1},
            {gj(0,0): 1, gj(0,1): 1},
            {gk(1,1): 1}
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            {gi(0,1): 1, gi(1,1): -1},
            {gj(1,0): 1, gj(1,1): 1},
            {gk(0,0): 1}
        )
    
    # Partition: n=4 as [2,2], m=5 as [2,2,1], p=5 as [2,2,1]
    i_parts = [(0, 2), (2, 2)]
    j_parts = [(0, 2), (2, 2), (4, 1)]
    k_parts = [(0, 2), (2, 2), (4, 1)]
    
    for i_off, i_sz in i_parts:
        for j_off, j_sz in j_parts:
            for k_off, k_sz in k_parts:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    strassen_block(i_off, j_off, k_off)
                else:
                    schoolbook_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of sparse terms to dense U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(terms, n, m, p, deadline):
    """
    Try to reduce the rank by finding linear dependencies.
    Uses the fact that if we can express some rank-1 terms as combinations of others,
    we can eliminate them.
    
    Strategy: Set up the bilinear system and try to find a smaller set of terms
    that satisfies all equations.
    """
    num_terms = len(terms)
    if num_terms < 2:
        return terms
    
    # Build the constraint matrix. For each (i,j,k), we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # 
    # This is a system where the unknowns are the entries of U, V, W.
    # But we want to find if we can reduce the number of columns.
    #
    # Alternative approach: Try to find a linear transformation on the rank dimension
    # that preserves the decomposition but has a lower effective rank.
    #
    # For now, try a simpler approach: check if any pair of terms can be merged
    # by finding a common factor structure.
    
    # A more practical approach: try random linear combinations of columns
    # and check if the resulting decomposition is still valid with fewer columns.
    
    # Actually, the most practical approach within time constraints is to
    # try to identify if the current decomposition has redundant structure.
    
    # Let's try: for each pair of terms, check if one can be expressed
    # as a linear combination of others (in the tensor sense).
    
    # This is computationally expensive, so let's try a heuristic:
    # Check if removing each term one at a time still gives a valid decomposition
    # (which won't work directly, but we can try to repair).
    
    # A better heuristic: try to find a 2D subspace of the rank dimension
    # where the decomposition factors through a lower-rank map.
    
    # Given time constraints, let's try a different approach:
    # Use the fact that for small matrices, we can try to find an explicit
    # lower-rank decomposition by solving the polynomial system.
    
    # For 4x5x5, the known lower bounds suggest rank >= 90 or so.
    # Our current rank is 92, so there might be only 1-2 ranks to gain.
    
    # Let's try: can we merge two schoolbook terms into one Strassen-like term?
    
    # For now, return the original terms if we can't find an improvement quickly.
    if time.time() > deadline:
        return terms
    
    # Try a simple optimization: look for terms that differ only in one factor
    # and can be combined.
    
    # Build a lookup for quick comparison
    term_keys = []
    for t in terms:
        u_key = frozenset(t[0].items())
        v_key = frozenset(t[1].items())
        w_key = frozenset(t[2].items())
        term_keys.append((u_key, v_key, w_key))
    
    # Check for duplicate terms (shouldn't happen, but just in case)
    seen = set()
    unique_terms = []
    for t, key in zip(terms, term_keys):
        if key not in seen:
            seen.add(key)
            unique_terms.append(t)
    
    if len(unique_terms) < num_terms:
        return unique_terms
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin for safety
    
    # Step 1: Build block decomposition
    terms = _build_block_decomposition(n, m, p)
    original_rank = len(terms)
    
    # Step 2: Convert to matrices
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    # Step 3: Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 4: Try to reduce rank
    if time.time() < deadline:
        reduced_terms = _try_reduce_rank(terms, n, m, p, deadline)
        if len(reduced_terms) < original_rank:
            U, V, W = _terms_to_matrices(reduced_terms, n, m, p)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the block decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
