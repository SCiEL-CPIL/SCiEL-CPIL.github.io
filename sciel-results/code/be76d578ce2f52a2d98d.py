import numpy as np
import time

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

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    # Build tensors
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)  
    # W: (p*n, rank) -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For each (i,j,k): dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
    # Use einsum: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _compute_rank(n, m, p, n_blocks, m_blocks, p_blocks, rank_333=23):
    """Compute the rank without building matrices."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                elif ns == 3 and ms == 3 and ps == 3:
                    rank += rank_333
                else:
                    rank += ns * ms * ps
    return rank

def _get_all_splits(dim, max_part=3):
    """Generate all compositions of dim into parts of size 1, 2, and 3."""
    splits = []
    def _gen(remaining, current):
        if remaining == 0:
            splits.append(current[:])
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(dim, [])
    return splits

def _find_best_block_split(n, m, p, rank_333=23):
    """Search over block splits to find the one with minimum rank."""
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    best_rank = n * m * p
    best_split = None
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = _compute_rank(n, m, p, ns, ms, ps, rank_333)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def _find_333_decomposition():
    """Find a rank-23 decomposition of the 3x3x3 matrix multiplication tensor."""
    rank = 23
    start_time = time.time()
    deadline = start_time + 120
    
    def residuals(p):
        res = np.zeros(27)
        for r in range(rank):
            u = p[r*27:r*27+9]
            v = p[r*27+9:r*27+18]
            w = p[r*27+18:r*27+27]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        res[i*9+j*3+k] += u[i*3+j] * v[j*3+k] * w[k*3+i]
        res -= 1.0
        return res
    
    # Try multiple initialization strategies
    best_result = None
    best_cost = np.inf
    
    # Strategy 1: Trivial 27 terms with 4 zeroed out
    params0_trivial = np.zeros(rank * 27)
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if idx < rank:
                    params0_trivial[idx*27 + i*3+j] = 1.0
                    params0_trivial[idx*27 + 9 + j*3+k] = 1.0
                    params0_trivial[idx*27 + 18 + k*3+i] = 1.0
                idx += 1
    
    # Strategy 2: Random initialization with small values
    np.random.seed(42)
    params0_random = np.random.randn(rank * 27) * 0.5
    
    # Strategy 3: Hybrid - start from trivial, add random perturbations to last 4
    params0_hybrid = params0_trivial.copy()
    for r in range(rank-4, rank):
        params0_hybrid[r*27:(r+1)*27] = np.random.randn(27) * 0.3
    
    initializations = [params0_trivial, params0_random, params0_hybrid]
    
    for init_idx, p0_base in enumerate(initializations):
        if time.time() > deadline:
            break
            
        for attempt in range(3):
            if time.time() > deadline:
                    break
                    
            if attempt == 0:
                p0 = p0_base.copy()
            else:
                p0 = p0_base + np.random.randn(rank * 27) * 0.2
            
            try:
                result = least_squares(residuals, p0, method='lm', max_nfev=15000)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
                    
                    # Check if we've found a good solution
                    if best_cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_cost < 1e-10:
            break
    
    if best_result is not None and best_cost < 1e-8:
        U = best_result.x[:rank*9].reshape(rank, 3, 3)
        V = best_result.x[rank*9:2*rank*9].reshape(rank, 3, 3)
        W = best_result.x[2*rank*9:3*rank*9].reshape(rank, 3, 3)
        return U, V, W
    
    return None

def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks, dec_333=None):
    """Generate a decomposition using block-wise Strassen and 3x3x3."""
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
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                    terms.append((ni, mi, pi, 'strassen'))
                elif ns == 3 and ms == 3 and ps == 3:
                    rank += 23
                    terms.append((ni, mi, pi, 'bini'))
                else:
                    rank += ns * ms * ps
                    terms.append((ni, mi, pi, 'trivial'))
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for ni, mi, pi, method in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        ns = n_blocks[ni]
        ms = m_blocks[mi]
        ps = p_blocks[pi]
        
        if method == 'strassen':
            strassen_data = [
                ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
                ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
                ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
                ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
            ]
            
            for a_factors, b_factors, c_factors in strassen_data:
                for (ar, ac, acoeff) in a_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1
        
        elif method == 'bini':
            if dec_333 is None:
                # Fallback to trivial
                for i in range(ns):
                    for j in range(ms):
                        for k in range(ps):
                            U[(n0+i)*m + (m0+j), col] = 1
                            V[(m0+j)*p + (p0+k), col] = 1
                            W[(p0+k)*n + (n0+i), col] = 1
                            col += 1
            else:
                U3, V3, W3 = dec_333
                for r in range(23):
                    for i in range(3):
                        for j in range(3):
                            U[(n0+i)*m + (m0+j), col] = U3[r, i, j]
                    for j in range(3):
                        for k in range(3):
                            V[(m0+j)*p + (p0+k), col] = V3[r, j, k]
                    for k in range(3):
                        for i in range(3):
                            W[(p0+k)*n + (n0+i), col] = W3[r, k, i]
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
    deadline = start_time + 115  # Leave buffer for final operations
    
    # Try to find 3x3x3 rank-23 decomposition
    dec_333 = None
    try:
        dec_333 = _find_333_decomposition()
    except Exception:
        dec_333 = None
    
    if time.time() > deadline:
        # Fallback to trivial
        U, V, W = _block_decomposition(n, m, p, [n], [m], [p])
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Search for best block split
    rank_333 = 23 if dec_333 is not None else 27
    best_rank, best_split = _find_best_block_split(n, m, p, rank_333)
    
    if time.time() > deadline:
        U, V, W = _block_decomposition(n, m, p, [n], [m], [p])
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_split is not None and best_rank < n*m*p:
        U, V, W = _block_decomposition(n, m, p, best_split[0], best_split[1], best_split[2], dec_333)
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try several fallback splits in order of expected rank
    fallback_splits = [
        [2,1], [2,2,1], [2,2,2,1],
        [2,1], [2,2,1], [2,2,2,1],
        [2,1], [2,2,1], [2,2,2,1],
        [n], [m], [p]
    ]
    
    for ns, ms, ps in fallback_splits:
        if time.time() > deadline:
            break
        try:
            U, V, W = _block_decomposition(n, m, p, ns, ms, ps, dec_333)
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
    
    # Final fallback
    U, V, W = _block_decomposition(n, m, p, [n], [m], [p])
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
