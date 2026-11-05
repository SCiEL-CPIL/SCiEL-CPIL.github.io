import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _compute_block_rank(ns, ms, ps):
    """Compute rank for a single block using known algorithms."""
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    elif ns == 3 and ms == 3 and ps == 3:
        return 23
    else:
        return ns * ms * ps

def _get_all_splits(dim, max_part=3):
    """Generate all compositions of dim into parts of size 1, 2, and 3."""
    splits = []
    def _gen(remaining, current):
        if remaining == 0:
            splits.append(tuple(current))
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(dim, [])
    return splits

def _find_best_block_split(n, m, p):
    """Search over block splits to find the one with minimum rank."""
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    best_rank = n * m * p
    best_split = None
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = 0
                for ni, mi, pi in zip(ns, ms, ps):
                    rank += _compute_block_rank(ni, mi, pi)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2."""
    # Each term: (U entries, V entries, W entries)
    # U: (i,j), V: (j,k), W: (k,i) where i,j,k in {0,1}
    terms = [
        # Term 1: (A00+A11)(B00+B11) -> C00 += C11 += C10 += C01 -=
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # Term 2: (A10+A11)B00 -> C10 += C11 -=
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # Term 3: A00(B10-B00) -> C01 += C00 -=
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(0,0,-1)]),
        # Term 4: A11(B01-B10) -> C11 += C10 -=
        ([(1,1,1)], [(0,1,1),(1,0,-1)], [(1,1,1),(1,0,-1)]),
        # Term 5: (A00+A01)B11 -> C00 += C01 -=
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,-1)]),
        # Term 6: (A10-A00)(B00+B01) -> C10 += C11 +=
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]),
        # Term 7: (A01-A11)(B10+B11) -> C01 += C00 -=
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1),(0,0,-1)]),
    ]
    return terms

def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Generate a decomposition using block-wise Strassen and schoolbook."""
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
                block_rank = _compute_block_rank(ns, ms, ps)
                rank += block_rank
                if ns == 2 and ms == 2 and ps == 2:
                    terms.append((ni, mi, pi, 'strassen'))
                else:
                    terms.append((ni, mi, pi, 'schoolbook'))
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    for ni, mi, pi, method in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        ns = n_blocks[ni]
        ms = m_blocks[mi]
        ps = p_blocks[pi]
        
        if method == 'strassen':
            for a_factors, b_factors, c_factors in strassen_terms:
                for (ar, ac, acoeff) in a_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1
        else:
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0+i)*m + (m0+j), col] = 1
                        V[(m0+j)*p + (p0+k), col] = 1
                        W[(p0+k)*n + (n0+i), col] = 1
                        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Search for best block split
    best_rank, best_split = _find_best_block_split(n, m, p)
    
    if time.time() > deadline:
        U, V, W = _block_decomposition(n, m, p, [n], [m], [p])
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_split is not None and best_rank < n*m*p:
        U, V, W = _block_decomposition(n, m, p, list(best_split[0]), list(best_split[1]), list(best_split[2]))
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try some specific splits as fallback
    fallback_splits = [
        ([2,1], [2,2,1], [2,2,2,1]),
        ([2,1], [2,2,1], [2,2,1,2]),
        ([2,1], [2,2,1], [2,2,1,1,1]),
        ([2,1], [2,2,1], [3,2,2]),
        ([2,1], [3,2], [2,2,2,1]),
        ([2,1], [3,2], [3,2,2]),
        ([3], [5], [7]),
    ]
    
    for ns, ms, ps in fallback_splits:
        if time.time() > deadline:
            break
        try:
            U, V, W = _block_decomposition(n, m, p, ns, ms, ps)
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
    
    # Final fallback: schoolbook
    U, V, W = _block_decomposition(n, m, p, [n], [m], [p])
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
