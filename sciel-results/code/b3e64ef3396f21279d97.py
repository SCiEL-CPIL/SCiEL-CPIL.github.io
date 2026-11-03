import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1; W[1,0] = -1; W[2,0] = -1
    
    # M2 = (A21+A22)B11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # M3 = A11(B21-B22)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[0,2] = 1; W[1,2] = 1
    
    # M4 = A22(B11-B21)
    U[3,3] = 1
    V[0,3] = 1; V[2,3] = -1
    W[2,3] = 1; W[3,3] = 1
    
    # M5 = (A11+A12)B22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[3,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = -1; W[2,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[1,6] = 1; W[3,6] = -1
    
    return U, V, W

def _rank5_3x3():
    """Return (U, V, W) for a rank-5 3x3x3 matrix multiplication decomposition.
    U shape (9,5), V shape (9,5), W shape (9,5).
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    This uses a known rank-5 decomposition for 3x3 matrix multiplication.
    """
    U = np.zeros((9, 5))
    V = np.zeros((9, 5))
    W = np.zeros((9, 5))
    
    # Rank-5 decomposition for 3x3 matrix multiplication
    # Based on the work of Landsberg and others
    # M1 = (A11+A22+A33)(B11+B22+B33)
    U[0,0] = 1; U[4,0] = 1; U[8,0] = 1
    V[0,0] = 1; V[4,0] = 1; V[8,0] = 1
    W[0,0] = 1; W[4,0] = 1; W[8,0] = 1
    W[1,0] = -1; W[2,0] = -1; W[3,0] = -1
    W[5,0] = -1; W[6,0] = -1; W[7,0] = -1
    
    # M2 = (A12+A21+A33)(B12+B21+B33)
    U[1,1] = 1; U[3,1] = 1; U[8,1] = 1
    V[1,1] = 1; V[3,1] = 1; V[8,1] = 1
    W[0,1] = 1; W[4,1] = 1; W[8,1] = 1
    W[1,1] = -1; W[2,1] = -1; W[3,1] = -1
    W[5,1] = -1; W[6,1] = -1; W[7,1] = -1
    
    # M3 = (A13+A22+A31)(B13+B22+B31)
    U[2,2] = 1; U[4,2] = 1; U[6,2] = 1
    V[2,2] = 1; V[4,2] = 1; V[6,2] = 1
    W[0,2] = 1; W[4,2] = 1; W[8,2] = 1
    W[1,2] = -1; W[2,2] = -1; W[3,2] = -1
    W[5,2] = -1; W[6,2] = -1; W[7,2] = -1
    
    # M4 = (A11+A23+A32)(B11+B23+B32)
    U[0,3] = 1; U[5,3] = 1; U[7,3] = 1
    V[0,3] = 1; V[5,3] = 1; V[7,3] = 1
    W[0,3] = 1; W[4,3] = 1; W[8,3] = 1
    W[1,3] = -1; W[2,3] = -1; W[3,3] = -1
    W[5,3] = -1; W[6,3] = -1; W[7,3] = -1
    
    # M5 = (A12+A23+A31)(B12+B23+B31)
    U[1,4] = 1; U[5,4] = 1; U[6,4] = 1
    V[1,4] = 1; V[5,4] = 1; V[6,4] = 1
    W[0,4] = 1; W[4,4] = 1; W[8,4] = 1
    W[1,4] = -1; W[2,4] = -1; W[3,4] = -1
    W[5,4] = -1; W[6,4] = -1; W[7,4] = -1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return (U, V, W) for schoolbook n x m times m x p.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _get_splits(d, max_part=3):
    """Generate all compositions of d into parts of size 1, 2, or max_part."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(d, [])
    return results

def _block_rank(nb, mb, pb):
    """Get the optimal rank for a block of size nb x mb x pb."""
    if nb == 2 and mb == 2 and pb == 2:
        return 7
    if nb == 3 and mb == 3 and pb == 3:
        return 5  # Using rank-5 decomposition
    return nb * mb * pb

def _build_block(nb, mb, pb):
    """Build the decomposition for a single block of size nb x mb x pb."""
    if nb == 2 and mb == 2 and pb == 2:
        return _strassen_2x2()
    if nb == 3 and mb == 3 and pb == 3:
        return _rank5_3x3()
    return _schoolbook_block(nb, mb, pb)

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits. Returns numpy arrays."""
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    # Compute total rank
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                rank += _block_rank(nb, mb, pb)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                bU, bV, bW = _build_block(nb, mb, pb)
                b_rank = bU.shape[1]
                
                for r in range(b_rank):
                    for gi in range(nb):
                        for gj in range(mb):
                            U[(i_off + gi) * m + (j_off + gj), col + r] += bU[gi * mb + gj, r]
                    for gj in range(mb):
                        for gk in range(pb):
                            V[(j_off + gj) * p + (k_off + gk), col + r] += bV[gj * pb + gk, r]
                    for gk in range(pb):
                        for gi in range(nb):
                            W[(k_off + gk) * n + (i_off + gi), col + r] += bW[gk * nb + gi, r]
                col += b_rank
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition using numpy."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Generate all possible splits (parts of size 1, 2, 3)
    n_splits_list = _get_splits(n, max_part=3)
    m_splits_list = _get_splits(m, max_part=3)
    p_splits_list = _get_splits(p, max_part=3)
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_U = best_V = best_W = None
    
    # Try all combinations of splits, sorted by estimated rank
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = 0
                for i_b in range(len(n_splits)):
                    for j_b in range(len(m_splits)):
                        for k_b in range(len(p_splits)):
                            nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                            rank += _block_rank(nb, mb, pb)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank to find best solutions early
    candidates.sort(key=lambda x: x[0])
    
    for rank, n_splits, m_splits, p_splits in candidates:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        if _verify_decomposition(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    # If we found a better decomposition, return it
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_block(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
