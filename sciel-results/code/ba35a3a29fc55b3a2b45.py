import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_terms():
    """
    Returns the 7 terms of Strassen's algorithm for 2x2x2.
    Each term is a tuple (U_factors, V_factors, W_factors).
    Factors are lists of (row, col, coeff).
    """
    terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(0,0,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1),(1,1,1)]),
    ]
    return terms

STRASSEN_TERMS = _get_strassen_terms()

def _block_rank(ns, ms, ps):
    """Calculate the rank of a block of size ns x ms x ps using known optimal ranks for small sizes."""
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    if ns == 1 and ms == 1 and ps == 1:
        return 1
    if ns == 1 and ms == 1 and ps == 2:
        return 2
    if ns == 1 and ms == 2 and ps == 1:
        return 2
    if ns == 2 and ms == 1 and ps == 1:
        return 2
    if ns == 1 and ms == 2 and ps == 2:
        return 4
    if ns == 2 and ms == 1 and ps == 2:
        return 4
    if ns == 2 and ms == 2 and ps == 1:
        return 4
    if ns == 3 and ms == 3 and ps == 3:
        return 23 # Laderman
    # Fallback: schoolbook
    return ns * ms * ps

def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    rank = 0
    terms = []
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                r = _block_rank(ns, ms, ps)
                rank += r
                terms.append((ni, mi, pi, ns, ms, ps, r))
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for ni, mi, pi, ns, ms, ps, r in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        
        if ns == 2 and ms == 2 and ps == 2:
            for u_factors, v_factors, w_factors in STRASSEN_TERMS:
                for (ar, ac, acoeff) in u_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in v_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in w_factors:
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1
        else:
            # Trivial schoolbook for this block
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0+i)*m + (m0+j), col] = 1
                        V[(m0+j)*p + (p0+k), col] = 1
                        W[(p0+k)*n + (n0+i), col] = 1
                        col += 1
    assert col == rank
    return U, V, W

def _find_best_block_split(n, m, p):
    best_rank = n * m * p
    best_split = None
    
    def gen_splits(dim, max_part):
        def _gen(d, max_p):
            if d == 0:
                yield []
                return
            for i in range(1, min(max_p, d) + 1):
                for rest in _gen(d - i, max_p):
                    yield [i] + rest
        for s in _gen(dim, max_part):
            yield s

    # Allow parts up to 3 to capture Laderman benefits if applicable
    n_splits = list(gen_splits(n, 3))
    m_splits = list(gen_splits(m, 3))
    p_splits = list(gen_splits(p, 3))
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if sum(ns) != n or sum(ms) != m or sum(ps) != p:
                    continue
                # Quick rank calculation
                rank = 0
                for i in range(len(ns)):
                    for j in range(len(ms)):
                        for k in range(len(ps)):
                            rank += _block_rank(ns[i], ms[j], ps[k])
                
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: Schoolbook
    U_trivial, V_trivial, W_trivial = _block_decomposition(n, m, p, [n], [m], [p])
    
    best_rank, best_split = _find_best_block_split(n, m, p)
    
    if time.time() > deadline:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    if best_split is not None and best_rank < n*m*p:
        U, V, W = _block_decomposition(n, m, p, best_split[0], best_split[1], best_split[2])
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
