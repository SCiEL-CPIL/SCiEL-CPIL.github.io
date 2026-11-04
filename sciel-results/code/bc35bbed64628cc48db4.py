"""
Matrix multiplication 5x5x5 - rank-minimal decomposition
Uses 2+3 block decomposition with Strassen for 2x2x2 and recursive splitting for 3-blocks.
"""
import time

DIMENSIONS = [5, 5, 5]
TIME_LIMIT = 120.0
DEADLINE = time.time() + TIME_LIMIT - 5.0  # 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

def decompose(n, m, p):
    """
    Returns (U, V, W) lists for n x m x p matrix multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Uses recursive block decomposition with Strassen for 2x2x2.
    """
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 1 and m == 1:
        # 1x1xp: just p terms
        U = [[1]*p]
        V = [[1]*p]
        W = [[1]*p]
        return U, V, W
    
    if n == 1:
        # 1mxp: m*p terms
        rank = m * p
        U = [[1]*rank]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p)]
        col = 0
        for j in range(m):
            for k in range(p):
                V[j*p+k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W
    
    if m == 1:
        # nx1xp: n*p terms
        rank = n * p
        U = [[0]*rank for _ in range(n)]
        V = [[1]*rank]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                W[k*n+i][col] = 1
                col += 1
        return U, V, W
    
    if p == 1:
        # nxmx1: n*m terms
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m)]
        W = [[0]*rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W
    
    # 2x2x2: Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Check deadline
    if time.time() > DEADLINE:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    # Recursive decomposition
    # Choose split strategy based on dimensions
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try splitting each dimension
    # For 5: try 2+3, 1+4
    # For 3: try 1+2
    # For 2: try 1+1 (but use Strassen if all are 2)
    
    splits = []
    
    # Determine possible splits for each dimension
    def get_splits(dim):
        s = []
        for a in range(1, dim):
            b = dim - a
            if a < b:
                s.append((a, b))
            elif a == b:
                s.append((a, b))
        return s
    
    n_splits = get_splits(n)
    m_splits = get_splits(m)
    p_splits = get_splits(p)
    
    # Try combinations of splits
    # To limit search, try a few promising combinations
    # Priority: split the largest dimension first, use 2+3 for 5, 1+2 for 3
    
    # Generate candidate split triples
    candidates = []
    
    # Strategy: try to use Strassen when possible
    # For 5x5x5: split as (2,3) in some dimensions
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                # Compute expected rank for this split
                # Each sub-block contributes its own rank
                expected_rank = 0
                for (n1, n2) in [ns]:
                    for (m1, m2) in [ms]:
                        for (p1, p2) in [ps]:
                            # 8 sub-blocks
                            sub_ranks = [
                                _estimate_rank(n1, m1, p1),
                                _estimate_rank(n1, m1, p2),
                                _estimate_rank(n1, m2, p1),
                                _estimate_rank(n1, m2, p2),
                                _estimate_rank(n2, m1, p1),
                                _estimate_rank(n2, m1, p2),
                                _estimate_rank(n2, m2, p1),
                                _estimate_rank(n2, m2, p2),
                            ]
                            expected_rank += sum(sub_ranks)
                candidates.append((expected_rank, ns, ms, ps))
    
    # Sort by expected rank
    candidates.sort(key=lambda x: x[0])
    
    # Try top candidates
    for expected_rank, ns, ms, ps in candidates[:10]:
        if time.time() > DEADLINE:
            break
        
        result = _try_split(n, m, p, ns, ms, ps)
        if result is not None:
            U_r, V_r, W_r = result
            actual_rank = len(W_r[0]) if W_r and W_r[0] else 0
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _estimate_rank(n, m, p):
    """Estimate rank for n x m x p without computing."""
    if n == 1 or m == 1 or p == 1:
        return n * m * p
    if n == 2 and m == 2 and p == 2:
        return 7
    # Heuristic: use known bounds
    if n <= 3 and m <= 3 and p <= 3:
        # Conservative estimate
        return n * m * p
    return n * m * p

def _try_split(n, m, p, n_split, m_split, p_split):
    """
    Try to decompose n x m x p by splitting into blocks.
    n_split = (n1, n2), m_split = (m1, m2), p_split = (p1, p2)
    """
    n1, n2 = n_split
    m1, m2 = m_split
    p1, p2 = p_split
    
    # 8 sub-blocks
    sub_blocks = [
        (n1, m1, p1),  # top-left-front
        (n1, m1, p2),  # top-left-back
        (n1, m2, p1),  # top-right-front
        (n1, m2, p2),  # top-right-back
        (n2, m1, p1),  # bottom-left-front
        (n2, m1, p2),  # bottom-left-back
        (n2, m2, p1),  # bottom-right-front
        (n2, m2, p2),  # bottom-right-back
    ]
    
    # Recursively decompose each sub-block
    sub_factors = []
    total_rank = 0
    
    for (sn, sm, sp) in sub_blocks:
        U_s, V_s, W_s = decompose(sn, sm, sp)
        sub_factors.append((sn, sm, sp, U_s, V_s, W_s))
        total_rank += len(W_s[0])
    
    # Now assemble the global factors
    # U global row index: i * m + j
    # V global row index: j * p + k  
    # W global row index: k * n + i
    
    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for (sn, sm, sp, U_s, V_s, W_s) in sub_factors:
        # Determine offsets for this sub-block
        # This sub-block corresponds to specific i, j, k ranges
        # We need to figure out which sub-block this is
        
        # The sub-blocks are ordered as:
        # 0: n1,m1,p1  -> i in [0,n1), j in [0,m1), k in [0,p1)
        # 1: n1,m1,p2  -> i in [0,n1), j in [0,m1), k in [p1,p1+p2)
        # 2: n1,m2,p1  -> i in [0,n1), j in [m1,m1+m2), k in [0,p1)
        # 3: n1,m2,p2  -> i in [0,n1), j in [m1,m1+m2), k in [p1,p1+p2)
        # 4: n2,m1,p1  -> i in [n1,n1+n2), j in [0,m1), k in [0,p1)
        # 5: n2,m1,p2  -> i in [n1,n1+n2), j in [0,m1), k in [p1,p1+p2)
        # 6: n2,m2,p1  -> i in [n1,n1+n2), j in [m1,m1+m2), k in [0,p1)
        # 7: n2,m2,p2  -> i in [n1,n1+n2), j in [m1,m1+m2), k in [p1,p1+p2)
        
        idx = sub_factors.index((sn, sm, sp, U_s, V_s, W_s))
        
        if idx == 0:
            i_off, j_off, k_off = 0, 0, 0
        elif idx == 1:
            i_off, j_off, k_off = 0, 0, p1
        elif idx == 2:
            i_off, j_off, k_off = 0, m1, 0
        elif idx == 3:
            i_off, j_off, k_off = 0, m1, p1
        elif idx == 4:
            i_off, j_off, k_off = n1, 0, 0
        elif idx == 5:
            i_off, j_off, k_off = n1, 0, p1
        elif idx == 6:
            i_off, j_off, k_off = n1, m1, 0
        else:
            i_off, j_off, k_off = n1, m1, p1
        
        # Copy factors with appropriate row mappings
        for r in range(sn * sm):
            # r = i_local * sm + j_local
            i_local = r // sm
            j_local = r % sm
            i_global_idx = (i_off + i_local) * m + (j_off + j_local)
            for c in range(len(U_s[r])):
                U_global[i_global_idx][col_offset + c] = U_s[r][c]
        
        for r in range(sm * sp):
            # r = j_local * sp + k_local
            j_local = r // sp
            k_local = r % sp
            j_global_idx = (j_off + j_local) * p + (k_off + k_local)
            for c in range(len(V_s[r])):
                V_global[j_global_idx][col_offset + c] = V_s[r][c]
        
        for r in range(sp * sn):
            # r = k_local * sn + i_local
            k_local = r // sn
            i_local = r % sn
            k_global_idx = (k_off + k_local) * n + (i_off + i_local)
            for c in range(len(W_s[r])):
                W_global[k_global_idx][col_offset + c] = W_s[r][c]
        
        col_offset += len(U_s[0])
    
    return U_global, V_global, W_global

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    
    U rows: i*m+j = i*2+j for i,j in {0,1}
    V rows: j*p+k = j*2+k for j,k in {0,1}
    W rows: k*n+i = k*2+i for k,i in {0,1}
    """
    # Strassen matrices (7 terms)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U factors (A matrix): rows are (i,j)
    # i,j: 00, 01, 10, 11 -> rows 0,1,2,3
    U = [
        # col: M1  M2  M3  M4  M5  M6  M7
        [1,  0,  1,  0,  1, -1,  0],   # A00 (i=0,j=0)
        [0,  0,  0,  0,  1,  0,  1],   # A01 (i=0,j=1)
        [0,  1,  0,  0,  0,  1,  0],   # A10 (i=1,j=0)
        [1,  1,  0,  1,  0,  0, -1],   # A11 (i=1,j=1)
    ]
    
    # V factors (B matrix): rows are (j,k)
    # j,k: 00, 01, 10, 11 -> rows 0,1,2,3
    V = [
        # col: M1  M2  M3  M4  M5  M6  M7
        [1,  1,  0,  0,  0,  1,  0],   # B00 (j=0,k=0)
        [0,  0,  0, -1, 0,  0,  1],   # B01 (j=0,k=1)
        [1,  0,  1,  1,  0,  1,  0],   # B10 (j=1,k=0)
        [1,  0, -1, 0,  1,  0,  1],   # B11 (j=1,k=1)
    ]
    
    # W factors (C matrix): rows are (k,i)
    # k,i: 00, 01, 10, 11 -> rows 0,1,2,3
    W = [
        # col: M1  M2  M3  M4  M5  M6  M7
        [1,  0,  0,  0,  0,  1,  0],   # C00 (k=0,i=0)
        [0,  0,  0,  1,  0,  0,  1],   # C01 (k=0,i=1)
        [0,  1,  1,  0,  0,  0,  0],   # C10 (k=1,i=0)
        [1,  1,  0,  1,  1,  0,  0],   # C11 (k=1,i=1)
    ]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W
