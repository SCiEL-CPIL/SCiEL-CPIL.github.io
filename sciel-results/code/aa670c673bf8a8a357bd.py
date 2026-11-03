import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_terms():
    """Return the 7 Strassen rank-1 terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _schoolbook_terms(n, m, p):
    """Return schoolbook rank-1 terms for (n,m,p)."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
    return terms

def _get_compositions(d):
    """Generate all compositions of d into positive integers."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in range(1, remaining + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(d, [])
    return results

_rank_memo = {}

def _compute_rank(n, m, p, deadline=None):
    """Compute minimal rank for (n,m,p) matrix multiplication tensor."""
    if deadline is not None and time.time() > deadline:
        return n * m * p
    
    key = (n, m, p)
    if key in _rank_memo:
        return _rank_memo[key]
    
    if n == 1 and m == 1 and p == 1:
        _rank_memo[key] = 1
        return 1
    if n == 2 and m == 2 and p == 2:
        _rank_memo[key] = 7
        return 7
    if n == 1 or m == 1 or p == 1:
        rank = n * m * p
        _rank_memo[key] = rank
        return rank
    
    n_comps = _get_compositions(n)
    m_comps = _get_compositions(m)
    p_comps = _get_compositions(p)
    
    best_rank = n * m * p
    
    for n_comp in n_comps:
        if len(n_comp) < 2:
            continue
        for m_comp in m_comps:
            if len(m_comp) < 2:
                continue
            for p_comp in p_comps:
                if len(p_comp) < 2:
                    continue
                if len(n_comp) == 2 and len(m_comp) == 2 and len(p_comp) == 2:
                    n1, n2 = n_comp
                    m1, m2 = m_comp
                    p1, p2 = p_comp
                    rank = (_compute_rank(n1, m1, p1, deadline) +
                           _compute_rank(n1, m2, p1, deadline) +
                           _compute_rank(n1, m1, p2, deadline) +
                           _compute_rank(n1, m2, p2, deadline) +
                           _compute_rank(n2, m1, p1, deadline) +
                           _compute_rank(n2, m2, p1, deadline) +
                           _compute_rank(n2, m1, p2, deadline) +
                           _compute_rank(n2, m2, p2, deadline))
                    if rank < best_rank:
                        best_rank = rank
    
    for n_comp in n_comps:
        if len(n_comp) == 2:
            n1, n2 = n_comp
            rank = _compute_rank(n1, m, p, deadline) + _compute_rank(n2, m, p, deadline)
            if rank < best_rank:
                best_rank = rank
    
    for m_comp in m_comps:
        if len(m_comp) == 2:
            m1, m2 = m_comp
            rank = _compute_rank(n, m1, p, deadline) + _compute_rank(n, m2, p, deadline)
            if rank < best_rank:
                best_rank = rank
    
    for p_comp in p_comps:
        if len(p_comp) == 2:
            p1, p2 = p_comp
            rank = _compute_rank(n, m, p1, deadline) + _compute_rank(n, m, p2, deadline)
            if rank < best_rank:
                best_rank = rank
    
    _rank_memo[key] = best_rank
    return best_rank

def _build_decomposition(n, m, p, i_off, j_off, k_off, U, V, W, col, deadline):
    """Recursively build the decomposition, adding to U, V, W starting at column col."""
    if deadline is not None and time.time() > deadline:
        return col
    
    if n == 2 and m == 2 and p == 2:
        terms = _strassen_terms()
        for U_d, V_d, W_d in terms:
            for (li, lj), val in U_d.items():
                U[(i_off + li) * m + (j_off + lj)][col] = val
            for (lj, lk), val in V_d.items():
                V[(j_off + lj) * p + (k_off + lk)][col] = val
            for (lk, li), val in W_d.items():
                W[(k_off + lk) * n + (i_off + li)][col] = val
            col += 1
        return col
    
    if n == 1 or m == 1 or p == 1:
        for li in range(n):
            for lj in range(m):
                for lk in range(p):
                    U[(i_off + li) * m + (j_off + lj)][col] = 1
                    V[(j_off + lj) * p + (k_off + lk)][col] = 1
                    W[(k_off + lk) * n + (i_off + li)][col] = 1
                    col += 1
        return col
    
    best_rank = n * m * p
    best_split = None
    
    for split in range(1, n):
        n1, n2 = split, n - split
        rank = _compute_rank(n1, m, p, deadline) + _compute_rank(n2, m, p, deadline)
        if rank < best_rank:
            best_rank = rank
            best_split = ('n', split)
    
    for split in range(1, m):
        m1, m2 = split, m - split
        rank = _compute_rank(n, m1, p, deadline) + _compute_rank(n, m2, p, deadline)
        if rank < best_rank:
            best_rank = rank
            best_split = ('m', split)
    
    for split in range(1, p):
        p1, p2 = split, p - split
        rank = _compute_rank(n, m, p1, deadline) + _compute_rank(n, m, p2, deadline)
        if rank < best_rank:
            best_rank = rank
            best_split = ('p', split)
    
    for n_split in range(1, n):
        for m_split in range(1, m):
            for p_split in range(1, p):
                n1, n2 = n_split, n - n_split
                m1, m2 = m_split, m - m_split
                p1, p2 = p_split, p - p_split
                rank = (_compute_rank(n1, m1, p1, deadline) +
                       _compute_rank(n1, m2, p1, deadline) +
                       _compute_rank(n1, m1, p2, deadline) +
                       _compute_rank(n1, m2, p2, deadline) +
                       _compute_rank(n2, m1, p1, deadline) +
                       _compute_rank(n2, m2, p1, deadline) +
                       _compute_rank(n2, m1, p2, deadline) +
                       _compute_rank(n2, m2, p2, deadline))
                if rank < best_rank:
                    best_rank = rank
                    best_split = ('all', n_split, m_split, p_split)
    
    if best_split is None:
        for li in range(n):
            for lj in range(m):
                for lk in range(p):
                    U[(i_off + li) * m + (j_off + lj)][col] = 1
                    V[(j_off + lj) * p + (k_off + lk)][col] = 1
                    W[(k_off + lk) * n + (i_off + li)][col] = 1
                    col += 1
        return col
    
    if best_split[0] == 'n':
        n1 = best_split[1]
        n2 = n - n1
        col = _build_decomposition(n1, m, p, i_off, j_off, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n2, m, p, i_off + n1, j_off, k_off, U, V, W, col, deadline)
    elif best_split[0] == 'm':
        m1 = best_split[1]
        m2 = m - m1
        col = _build_decomposition(n, m1, p, i_off, j_off, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n, m2, p, i_off, j_off + m1, k_off, U, V, W, col, deadline)
    elif best_split[0] == 'p':
        p1 = best_split[1]
        p2 = p - p1
        col = _build_decomposition(n, m, p1, i_off, j_off, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n, m, p2, i_off, j_off, k_off + p1, U, V, W, col, deadline)
    elif best_split[0] == 'all':
        n1, m1, p1 = best_split[1], best_split[2], best_split[3]
        n2, m2, p2 = n - n1, m - m1, p - p1
        col = _build_decomposition(n1, m1, p1, i_off, j_off, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n1, m2, p1, i_off, j_off + m1, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n1, m1, p2, i_off, j_off, k_off + p1, U, V, W, col, deadline)
        col = _build_decomposition(n1, m2, p2, i_off, j_off + m1, k_off + p1, U, V, W, col, deadline)
        col = _build_decomposition(n2, m1, p1, i_off + n1, j_off, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n2, m2, p1, i_off + n1, j_off + m1, k_off, U, V, W, col, deadline)
        col = _build_decomposition(n2, m1, p2, i_off + n1, j_off, k_off + p1, U, V, W, col, deadline)
        col = _build_decomposition(n2, m2, p2, i_off + n1, j_off + m1, k_off + p1, U, V, W, col, deadline)
    
    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U_np.shape[1]):
                    s += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110
    
    _rank_memo.clear()
    
    rank = _compute_rank(n, m, p, deadline)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = _build_decomposition(n, m, p, 0, 0, 0, U, V, W, 0, deadline)
    
    if col != rank:
        if col < rank:
            pass
        else:
            U = [row[:rank] for row in U]
            V = [row[:rank] for row in V]
            W = [row[:rank] for row in W]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
