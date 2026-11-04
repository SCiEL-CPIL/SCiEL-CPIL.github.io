"""Matrix multiplication tensor decomposition for 3x5x5 using recursive Strassen-style splitting."""
import time
import numpy as np
from itertools import product

_DIM = (3, 5, 5)

def _strassen_2x2x2():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication.
    Each pattern is a dict with 'U', 'V', 'W' lists of (i, j, val) tuples."""
    return [
        {'U': [(0,0,1),(1,1,1)], 'V': [(0,0,1),(1,1,1)], 'W': [(0,0,1),(1,1,1)]},
        {'U': [(1,0,1),(1,1,1)], 'V': [(0,0,1)],           'W': [(0,1,1),(1,1,-1)]},
        {'U': [(0,0,1)],           'V': [(0,1,1),(1,1,-1)], 'W': [(1,0,1),(1,1,1)]},
        {'U': [(1,1,1)],           'V': [(1,0,1),(0,0,-1)], 'W': [(0,0,1),(0,1,1)]},
        {'U': [(0,0,1),(0,1,1)],   'V': [(1,1,1)],           'W': [(0,0,-1),(1,0,1)]},
        {'U': [(1,0,1),(0,0,-1)],  'V': [(0,0,1),(0,1,1)],   'W': [(1,1,1)]},
        {'U': [(0,1,1),(1,1,-1)],  'V': [(1,0,1),(1,1,1)],   'W': [(0,0,1)]},
    ]

def _strassen_3x3x3():
    """Return a decomposition for 3x3x3 using rank 21 (or similar).
    We'll use a known 3x3x3 decomposition if available, otherwise fall back to schoolbook.
    For now, use schoolbook (27) as a safe baseline, but try to find better."""
    # Known: 3x3x3 can be done in rank 21 (Bini et al.) but it's complex.
    # Let's just use schoolbook for 3x3x3 for now.
    return None  # Signal to use schoolbook

def _get_compositions(d, max_part):
    """Generate all compositions of d into parts of size <= max_part."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in range(1, max_part + 1):
            if part <= remaining:
                current.append(part)
                _gen(remaining - part, current)
                current.pop()
    _gen(d, [])
    return results

def _block_rank(nb, mb, pb):
    """Compute the rank for a single block of size nb x mb x pb.
    Uses Strassen for 2x2x2, schoolbook otherwise."""
    if nb == 2 and mb == 2 and pb == 2:
        return 7
    else:
        return nb * mb * pb

def _build_block(nb, mb, pb, i_off, j_off, k_off, n, m, p, U, V, W, col):
    """Build the decomposition for a single block and return the new col index."""
    if nb == 2 and mb == 2 and pb == 2:
        patterns = _strassen_2x2x2()
        for pattern in patterns:
            for li, lj, val in pattern['U']:
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][col] = val
            for lj, lk, val in pattern['V']:
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][col] = val
            for lk, li, val in pattern['W']:
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][col] = val
            col += 1
    else:
        for li in range(nb):
            for lj in range(mb):
                for lk in range(pb):
                    gi = i_off + li
                    gj = j_off + lj
                    gk = k_off + lk
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
    return col

def _verify(U, V, W, n, m, p):
    """Check that the decomposition is valid using vectorized NumPy."""
    Ua = np.array(U, dtype=np.float64).reshape(n, m, -1)
    Va = np.array(V, dtype=np.float64).reshape(m, p, -1)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, -1)
    # result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    result = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    n, m, p = _DIM
    t0 = time.time()
    deadline = t0 + 110  # Safety margin before 120s limit
    
    # Get all possible compositions for each dimension
    n_comps = _get_compositions(n, 3)
    m_comps = _get_compositions(m, 3)
    p_comps = _get_compositions(p, 3)
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_result = None
    
    # Try all combinations of splits
    for n_splits in n_comps:
        if time.time() > deadline:
            break
        for m_splits in m_comps:
            if time.time() > deadline:
                break
            for p_splits in p_comps:
                if time.time() > deadline:
                    break
                
                # Compute total rank
                rank = 0
                valid = True
                for i_b in range(len(n_splits)):
                    for j_b in range(len(m_splits)):
                        for k_b in range(len(p_splits)):
                            nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                            rank += _block_rank(nb, mb, pb)
                
                if rank < best_rank:
                    # Build the decomposition
                    n_off = [0]
                    for s in n_splits[:-1]:
                        n_off.append(n_off[-1] + s)
                    m_off = [0]
                    for s in m_splits[:-1]:
                        m_off.append(m_off[-1] + s)
                    p_off = [0]
                    for s in p_splits[:-1]:
                        p_off.append(p_off[-1] + s)
                    
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    
                    col = 0
                    for i_b in range(len(n_splits)):
                        for j_b in range(len(m_splits)):
                            for k_b in range(len(p_splits)):
                                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                                i_off = n_off[i_b]
                                j_off = m_off[j_b]
                                k_off = p_off[k_b]
                                col = _build_block(nb, mb, pb, i_off, j_off, k_off, n, m, p, U, V, W, col)
                    
                    # Verify
                    if _verify(U, V, W, n, m, p):
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    # If we found a better decomposition, return it
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
