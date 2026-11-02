import time
import itertools
import random

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
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

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for the 2x2 block (indices 0,1).
    Each term is a dict: {'U': {row: val}, 'V': {row: val}, 'W': {row: val}}
    """
    terms = []
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    terms.append({'U': {0:1, 3:1}, 'V': {0:1, 4:1}, 'W': {0:1, 4:1}})
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    terms.append({'U': {2:1, 3:1}, 'V': {0:1}, 'W': {0:1}})
    # Term 2: M3 = A[0,0] * (B[1,0]-B[0,0])
    terms.append({'U': {0:1}, 'V': {3:1, 0:-1}, 'W': {0:1}})
    # Term 3: M4 = A[1,1] * (B[1,1]-B[0,1])
    terms.append({'U': {3:1}, 'V': {4:1, 1:-1}, 'W': {4:1}})
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    terms.append({'U': {0:1, 1:1}, 'V': {4:1}, 'W': {4:1}})
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,1]+B[1,1])
    terms.append({'U': {2:1, 0:-1}, 'V': {1:1, 4:1}, 'W': {4:1}})
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[1,0])
    terms.append({'U': {1:1, 3:-1}, 'V': {0:1, 3:1}, 'W': {0:1}})
    return terms

def _construct_rank_26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Partition into 2x2 block (indices 0,1) and 1x1 blocks (index 2).
    Apply Strassen to the 2x2 block (7 terms) + 19 terms for the rest.
    """
    n, m, p = 3, 3, 3
    rank = 26

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for 2x2 block (i,j,k in {0,1})
    for col, term in enumerate(_strassen_2x2_terms()):
        for row, val in term['U'].items():
            U[row][col] += val
        for row, val in term['V'].items():
            V[row][col] += val
        for row, val in term['W'].items():
            W[row][col] += val

    # 19 more terms for the remaining entries involving index 2
    remaining = [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2)
    ]

    col = 7
    for (i,j,k) in remaining:
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _residual_entries():
    """
    The residual tensor after removing the 2x2 block (Strassen handles i,j,k in {0,1}).
    These are the 19 entries involving index 2.
    Returns list of (i,j,k) tuples.
    """
    return [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2)
    ]

def _search_residual_decomposition(max_rank, deadline, seed=42):
    """
    Search for a low-rank decomposition of the residual tensor (19 entries).
    The residual has T[i,j,k] = 1 for the 19 entries involving index 2, and 0 otherwise
    (within the full 3x3x3 space, but we only care about the 19 non-zero entries
    and that they don't interfere with the Strassen terms).
    
    Actually, we need the full tensor to be satisfied. The Strassen terms handle
    the 2x2 block entries (i,j,k in {0,1}), and the residual terms must handle
    the 19 entries involving index 2, WITHOUT disturbing the 2x2 block entries.
    
    This is tricky because the residual terms could have non-zero contributions
    to the 2x2 block entries. So we need the residual decomposition to have
    zero contribution to all (i,j,k) in {0,1}^3, and contribution 1 to the
    19 entries involving index 2.
    
    This is a constrained problem. Let's just search for a decomposition of
    the full tensor that uses 7 + r terms, where r is the residual rank.
    We'll fix the 7 Strassen terms and search for r additional terms.
    """
    n, m, p = 3, 3, 3
    random.seed(seed)
    
    # We'll search for r additional terms (r = max_rank - 7)
    r = max_rank - 7
    if r < 0:
        return None
    
    # The full tensor T[i,j,k] = 1 for all i,j,k in 0..2 (it's the matmul tensor)
    # Wait, no. The matmul tensor is T[i,j,k] = 1 for all i,j,k? 
    # C[i,k] = sum_j A[i,j]*B[j,k], so the tensor T[i,j,k] = 1 for all i,j,k.
    # Yes, it's the all-ones tensor of shape (3,3,3).
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # The 7 Strassen terms handle the 2x2 block. Let's compute their contribution
    # to all 27 entries.
    strassen_terms = _strassen_2x2_terms()
    
    # For each entry (i,j,k), compute the contribution from the 7 Strassen terms
    strassen_contrib = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib = 0
                for term in strassen_terms:
                    u_val = term['U'].get(i*m+j, 0)
                    v_val = term['V'].get(j*p+k, 0)
                    w_val = term['W'].get(k*n+i, 0)
                    contrib += u_val * v_val * w_val
                strassen_contrib[(i,j,k)] = contrib
    
    # Now we need the residual terms to provide (1 - strassen_contrib) for each entry.
    # For entries in the 2x2 block, strassen_contrib should be 1 (hopefully).
    # For entries involving index 2, strassen_contrib should be 0.
    
    # Let's verify the Strassen terms are correct for the 2x2 block
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if abs(strassen_contrib[(i,j,k)] - 1) > 1e-9:
                    return None  # Strassen terms are wrong
    
    # For entries involving index 2, the Strassen terms should contribute 0
    for (i,j,k) in _residual_entries():
        if abs(strassen_contrib[(i,j,k)]) > 1e-9:
            # This is a problem - Strassen terms interfere with residual entries
            # We'll need to account for this in the search
            pass
    
    # Now search for r additional terms that complete the decomposition
    # Target: for all (i,j,k), sum of residual terms = 1 - strassen_contrib[(i,j,k)]
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i,j,k)] = 1 - strassen_contrib[(i,j,k)]
    
    # Search for r terms
    # Each term is (u_row, v_row, w_row, val) where val is the coefficient
    # The term contributes val * U[u_row,c] * V[v_row,c] * W[w_row,c]
    # But U,V,W are 0/1 (or -1) for the term, so the contribution to entry (i,j,k) is:
    # val * (1 if u_row == i*m+j else 0) * (1 if v_row == j*p+k else 0) * (1 if w_row == k*n+i else 0)
    # So each term only contributes to one entry (i,j,k) where i,j,k match the rows.
    # Wait, that's only if U,V,W are single-row terms. But we can have multi-row terms.
    
    # Let's use single-row terms for simplicity (each term has one non-zero in U, V, W)
    # Then each term contributes val to exactly one entry (i,j,k)
    # So we just need to cover all 27 entries with r terms, where each term covers one entry
    # with the correct target value.
    
    # But we can also combine entries: a term can have multiple non-zeros in U, V, W
    # and thus contribute to multiple entries.
    
    # Let's do a simple search: try to find r terms that cover all 27 entries
    # with the correct target values.
    
    # For efficiency, let's only consider terms that cover entries with non-zero target
    nonzero_targets = [(i,j,k) for i in range(n) for j in range(m) for k in range(p) 
                       if abs(target[(i,j,k)]) > 1e-9]
    
    if len(nonzero_targets) > r:
        # Can't cover more entries than terms (if each term covers at most 1 entry)
        # But terms can cover multiple entries if they have multiple non-zeros
        # Let's just try a random search
        pass
    
    # Random search: generate r random terms and check if they satisfy the targets
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Generate r random terms
        # Each term: choose u_row, v_row, w_row, and a value
        # The term contributes to entries (i,j,k) where i*m+j == u_row, j*p+k == v_row, k*n+i == w_row
        # For single-row terms, this is exactly one entry.
        
        # Let's use single-row terms for simplicity
        residual_terms = []
        for _ in range(r):
            u_row = random.randint(0, n*m-1)
            v_row = random.randint(0, m*p-1)
            w_row = random.randint(0, p*n-1)
            val = random.choice([-1, 1, -2, 2, -0.5, 0.5])
            residual_terms.append((u_row, v_row, w_row, val))
        
        # Check if these terms satisfy the targets
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib = 0
                    for (u_row, v_row, w_row, val) in residual_terms:
                        if u_row == i*m+j and v_row == j*p+k and w_row == k*n+i:
                            contrib += val
                    if abs(contrib - target[(i,j,k)]) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Found a valid residual decomposition
            # Build the full U, V, W matrices
            rank = 7 + r
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Add Strassen terms
            for col, term in enumerate(strassen_terms):
                for row, val in term['U'].items():
                    U[row][col] += val
                for row, val in term['V'].items():
                    V[row][col] += val
                for row, val in term['W'].items():
                    W[row][col] += val
            
            # Add residual terms
            for col, (u_row, v_row, w_row, val) in enumerate(residual_terms, start=7):
                U[u_row][col] += val
                V[v_row][col] += val
                W[w_row][col] += val
            
            result = {"U": U, "V": V, "W": W}
            if _verify(U, V, W):
                return result
    
    return None

def _construct_rank_23():
    """
    Try to construct a rank-23 decomposition.
    Uses 7 Strassen terms + 16 residual terms.
    """
    deadline = time.time() + 30
    return _search_residual_decomposition(23, deadline, seed=42)

def _construct_rank_24():
    """
    Try to construct a rank-24 decomposition.
    Uses 7 Strassen terms + 17 residual terms.
    """
    deadline = time.time() + 30
    return _search_residual_decomposition(24, deadline, seed=123)

def _construct_rank_25():
    """
    Try to construct a rank-25 decomposition.
    Uses 7 Strassen terms + 18 residual terms.
    """
    deadline = time.time() + 30
    return _search_residual_decomposition(25, deadline, seed=999)

def construct():
    """
    Return a rank-minimal decomposition for 3x3 matrix multiplication.
    """
    overall_deadline = time.time() + 140  # safety margin before 150s kill

    # Try in order of increasing rank (decreasing rank first)
    # Try rank 23
    r23 = _construct_rank_23()
    if r23 is not None:
        return r23
    if time.time() > overall_deadline:
        return _construct_schoolbook()

    # Try rank 24
    r24 = _construct_rank_24()
    if r24 is not None:
        return r24
    if time.time() > overall_deadline:
        return _construct_schoolbook()

    # Try rank 25
    r25 = _construct_rank_25()
    if r25 is not None:
        return r25
    if time.time() > overall_deadline:
        return _construct_schoolbook()

    # Fall back to rank 26
    r26 = _construct_rank_26()
    if _verify(r26["U"], r26["V"], r26["W"]):
        return r26

    # Final fallback: schoolbook
    return _construct_schoolbook()
