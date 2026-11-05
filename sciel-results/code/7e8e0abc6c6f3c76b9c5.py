import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_exact(U, V, W, n, m, p):
    """Verify decomposition with exact arithmetic using Fractions."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j, r]) * Fraction(V[j*p+k, r]) * Fraction(W[k*n+i, r])
                if s != Fraction(1):
                    return False
    return True

def _verify_float(U, V, W, n, m, p):
    """Quick float verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_222():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Each term is (u, v, w) where u is 2x2, v is 2x2, w is 2x2 (flattened).
    The convention: U[i,j] * V[j,k] * W[k,i] summed over r gives delta(i,k) for the product C=A*B.
    
    For 2x2 matrices:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A10)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    # U factors (for A), V factors (for B), W factors (for C)
    # Each is a 2x2 matrix stored as 4 values [r0c0, r0c1, r1c0, r1c1]
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # u: A00+A11 => [[1,0],[0,1]]
    # v: B00+B11 => [[1,0],[0,1]]
    # w: contributes to C00 and C11
    # C00: +M1, C11: +M1 => w = [[1,0],[0,1]]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)B00
    # u: A10+A11 => [[0,0],[1,1]]
    # v: B00 => [[1,0],[0,0]]
    # C10: +M2, C11: -M2 => w: C10 gets +1, C11 gets -1
    # w[k,i]: for C10 (k=1,i=0): +1; for C11 (k=1,i=1): -1
    # w = [[0,0],[1,-1]]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = A00(B10-B11)
    # u: A00 => [[1,0],[0,0]]
    # v: B10-B11 => [[0,0],[1,-1]]
    # C01: +M3, C11: +M3 => w: C01 (k=0,i=1): +1; C11 (k=1,i=1): +1
    # w = [[0,1],[0,1]]
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    
    # M4 = A11(B01-B00)
    # u: A11 => [[0,0],[0,1]]
    # v: B01-B00 => [[-1,1],[0,0]]
    # C00: +M4, C10: +M4 => w: C00 (k=0,i=0): +1; C10 (k=1,i=0): +1
    # w = [[1,0],[1,0]]
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    
    # M5 = (A00+A10)B11
    # u: A00+A10 => [[1,0],[1,0]]
    # v: B11 => [[0,0],[0,1]]
    # C00: -M5, C01: +M5 => w: C00 (k=0,i=0): -1; C01 (k=0,i=1): +1
    # w = [[-1,1],[0,0]]
    terms.append(([1,0,1,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A11-A10)(B00+B10)
    # u: A11-A10 => [[0,0],[-1,1]]
    # v: B00+B10 => [[1,0],[1,0]]
    # C11: +M6 => w: C11 (k=1,i=1): +1
    # w = [[0,0],[0,1]]
    terms.append(([0,0,-1,1], [1,0,1,0], [0,0,0,1]))
    
    # M7 = (A01-A11)(B01+B11)
    # u: A01-A11 => [[0,1],[0,-1]]
    # v: B01+B11 => [[0,1],[0,1]]
    # C00: +M7 => w: C00 (k=0,i=0): +1
    # w = [[1,0],[0,0]]
    terms.append(([0,1,0,-1], [0,1,0,1], [1,0,0,0]))
    
    return terms

def _build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Build a decomposition using block-wise Strassen for 2x2x2 blocks and
    trivial schoolbook for other blocks.
    """
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
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                else:
                    rank += ns * ms * ps
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    strassen_terms = _strassen_222()
    
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    for u_flat, v_flat, w_flat in strassen_terms:
                        for i in range(2):
                            for j in range(2):
                                U[(n0+i)*m + (m0+j), col] = u_flat[i*2+j]
                        for j in range(2):
                            for k in range(2):
                                V[(m0+j)*p + (p0+k), col] = v_flat[j*2+k]
                        for k in range(2):
                            for i in range(2):
                                W[(p0+k)*n + (n0+i), col] = w_flat[k*2+i]
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

def _get_all_splits(dim, max_part=2):
    """Generate all compositions of dim into parts of size 1 and 2."""
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

def _compute_rank(n_blocks, m_blocks, p_blocks):
    """Compute the rank for a given block split."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                else:
                    rank += ns * ms * ps
    return rank

def _find_best_split(n, m, p):
    """Find the best block split minimizing rank."""
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    best_rank = n * m * p
    best_split = ([n], [m], [p])
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = _compute_rank(ns, ms, ps)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Find the best block split
    best_rank, best_split = _find_best_split(n, m, p)
    
    if time.time() > deadline:
        # Emergency fallback
        U, V, W = _build_block_decomposition(n, m, p, [n], [m], [p])
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the decomposition
    U, V, W = _build_block_decomposition(n, m, p, best_split[0], best_split[1], best_split[2])
    
    # Verify with float first (fast)
    if not _verify_float(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _build_block_decomposition(n, m, p, [n], [m], [p])
    
    # Verify with exact arithmetic if time permits
    if time.time() < deadline - 5:
        if not _verify_exact(U, V, W, n, m, p):
            # Fall back to schoolbook
            U, V, W = _build_block_decomposition(n, m, p, [n], [m], [p])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
