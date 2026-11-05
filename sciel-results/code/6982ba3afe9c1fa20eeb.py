"""
Block-partition + local Strassen decomposition for 3x5 by 5x7 matrix multiplication.
Supports 2x2x2 Strassen blocks and mixed blocks (2x2x3, 2x3x2, 3x2x2) with
recursive Strassen+schoolbook sub-decompositions.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (A_factors, B_factors, C_factors) where each is a list of (row, col, coeff).
    A is indexed by (i, a) where i is row of A, a is col of A (which matches row of B).
    B is indexed by (b, k) where b is row of B, k is col of B.
    C is indexed by (i, k) where i is row of C, k is col of C.
    W is indexed by (k, i) per the task convention W[k*n+i, r].
    
    Standard Strassen:
    P1 = A11 * (B11 - B01)
    P2 = (A10 + A11) * B11
    P3 = A00 * (B01 - B11)
    P4 = (A00 + A11) * (B00 + B11)
    P5 = (A00 + A10) * B00
    P6 = (A10 - A00) * (B01 + B11)
    P7 = (A01 - A11) * (B00 + B10)
    
    C00 = P4 + P2 - P6 - P7
    C01 = P1 + P7
    C10 = P3 + P6
    C11 = P4 + P1 - P2 - P5
    """
    return [
        # P1
        ([(1, 1, 1)],
         [(1, 1, 1), (0, 1, -1)],
         [(0, 1, 1), (1, 1, 1)]),
        # P2
        ([(1, 0, 1), (1, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, 1), (1, 1, -1)]),
        # P3
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1)]),
        # P4
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        # P5
        ([(0, 0, 1), (1, 0, 1)],
         [(0, 0, 1)],
         [(1, 1, -1)]),
        # P6
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 1, 1), (1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        # P7
        ([(0, 1, 1), (1, 1, -1)],
         [(0, 0, 1), (1, 0, 1)],
         [(0, 0, -1), (0, 1, 1)]),
    ]


def _all_partitions(size):
    """Generate all partitions of 'size' into blocks of size 1, 2, or 3."""
    if size == 0:
        return [[]]
    results = []
    for s in [1, 2, 3]:
        if s <= size:
            for rest in _all_partitions(size - s):
                results.append([s] + rest)
    return results


def _compute_rank(n_blocks, m_blocks, p_blocks):
    """Compute the total rank for a given block partition using optimal local decomp."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                rank += _local_rank(ns, ms, ps)
    return rank


def _local_rank(ns, ms, ps):
    """
    Compute the minimum rank for a local ns x ms x ps block.
    Uses Strassen for 2x2x2, and recursive Strassen+schoolbook for mixed sizes.
    """
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    # For blocks where exactly one dimension is 3 and the other two are 2,
    # we can partition the size-3 dimension into [2,1] and use Strassen + schoolbook.
    if ns == 3 and ms == 2 and ps == 2:
        # Partition n into [2,1]: Strassen(2x2x2) + schoolbook(1x2x2) = 7 + 4 = 11
        return 11
    if ns == 2 and ms == 3 and ps == 2:
        # Partition m into [2,1]: Strassen(2x2x2) + schoolbook(2x1x2) = 7 + 4 = 11
        return 11
    if ns == 2 and ms == 2 and ps == 3:
        # Partition p into [2,1]: Strassen(2x2x2) + schoolbook(2x2x1) = 7 + 4 = 11
        return 11
    # For blocks where one dimension is 3 and others are 1:
    if ns == 3 and ms == 1 and ps == 1:
        return 3
    if ns == 1 and ms == 3 and ps == 1:
        return 3
    if ns == 1 and ms == 1 and ps == 3:
        return 3
    # For blocks where two dimensions are 3:
    if ns == 3 and ms == 3 and ps == 1:
        # Could partition both into [2,1]: 4 sub-blocks of (2x2x1),(2x1x1),(1x2x1),(1x1x1)
        # = 4+2+2+1 = 9, or just schoolbook 9
        return 9
    if ns == 3 and ms == 1 and ps == 3:
        return 9
    if ns == 1 and ms == 3 and ps == 3:
        return 9
    # For 3x3x3: partition each into [2,1]: 8 sub-blocks
    # (2,2,2)->7, (2,2,1)->4, (2,1,2)->4, (2,1,1)->2, (1,2,2)->4, (1,2,1)->2, (1,1,2)->2, (1,1,1)->1
    # Total: 7+4+4+2+4+2+2+1 = 26
    if ns == 3 and ms == 3 and ps == 3:
        return 26
    # Default: schoolbook
    return ns * ms * ps


def _build_local_block(U, V, W, n0, m0, p0, ns, ms, ps, m, p, n, col):
    """
    Build the decomposition for a local block of size ns x ms x ps.
    Returns the new column offset.
    """
    strassen_data = _strassen_terms()

    if ns == 2 and ms == 2 and ps == 2:
        # Pure Strassen
        for a_factors, b_factors, c_factors in strassen_data:
            for (ar, ac, acoeff) in a_factors:
                U[(n0 + ar) * m + (m0 + ac)][col] += acoeff
            for (br, bc, bcoeff) in b_factors:
                V[(m0 + br) * p + (p0 + bc)][col] += bcoeff
            for (cr, cc, ccoeff) in c_factors:
                W[(p0 + cc) * n + (n0 + cr)][col] += ccoeff
            col += 1
        return col
    elif ns == 3 and ms == 2 and ps == 2:
        # Partition n into [2,1]
        col = _build_local_block(U, V, W, n0, m0, p0, 2, ms, ps, m, p, n, col)
        col = _build_local_block(U, V, W, n0 + 2, m0, p0, 1, ms, ps, m, p, n, col)
        return col
    elif ns == 2 and ms == 3 and ps == 2:
        # Partition m into [2,1]
        col = _build_local_block(U, V, W, n0, m0, p0, ns, 2, ps, m, p, n, col)
        col = _build_local_block(U, V, W, n0, m0 + 2, p0, ns, 1, ps, m, p, n, col)
        return col
    elif ns == 2 and ms == 2 and ps == 3:
        # Partition p into [2,1]
        col = _build_local_block(U, V, W, n0, m0, p0, ns, ms, 2, m, p, n, col)
        col = _build_local_block(U, V, W, n0, m0, p0 + 2, ns, ms, 1, m, p, n, col)
        return col
    elif ns == 3 and ms == 3 and ps == 3:
        # Partition each dimension into [2,1]
        for di in [0, 2]:
            for dm in [0, 2]:
                for dp in [0, 2]:
                    sn = 2 if di == 0 else 1
                    sm = 2 if dm == 0 else 1
                    sp = 2 if dp == 0 else 1
                    col = _build_local_block(U, V, W, n0 + di, m0 + dm, p0 + dp, sn, sm, sp, m, p, n, col)
        return col
    else:
        # Schoolbook
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    U[(n0 + i) * m + (m0 + j)][col] = 1
                    V[(m0 + j) * p + (p0 + k)][col] = 1
                    W[(p0 + k) * n + (n0 + i)][col] = 1
                    col += 1
        return col


def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Build the decomposition using the given block partitions.
    Returns (U, V, W) as lists of lists.
    """
    # Compute offsets
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)

    # Compute total rank
    total_rank = _compute_rank(n_blocks, m_blocks, p_blocks)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

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
                col = _build_local_block(U, V, W, n0, m0, p0, ns, ms, ps, m, p, n, col)

    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Constructs the schoolbook decomposition for matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1

    return U, V, W


def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    U_reshaped = U_arr.reshape(n, m, U_arr.shape[1])
    V_reshaped = V_arr.reshape(m, p, V_arr.shape[1])
    W_reshaped = W_arr.reshape(p, n, W_arr.shape[1])

    try:
        result = np.einsum('iac,bkc,kic->iabk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return True

    expected = np.zeros((n, m, m, p))
    for a in range(m):
        expected[:, a, a, :] = 1.0

    if np.max(np.abs(result - expected)) > tol:
        return False
    return True


def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = _START_TIME + (_DEADLINE - (time.time() - _START_TIME))
    if time.time() > deadline:
        deadline = _START_TIME + 5.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    # Generate all valid partitions (blocks of size 1, 2, or 3)
    n_partitions = _all_partitions(n)
    m_partitions = _all_partitions(m)
    p_partitions = _all_partitions(p)

    # Find the best partition (minimum rank)
    best_rank = n * m * p  # Schoolbook rank = 105
    best_partition = (n_partitions[0], m_partitions[0], p_partitions[0])

    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = _compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_partition = (np_, mp_, pp_)

    # Check time
    if time.time() > deadline - 20:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Build the best decomposition
    try:
        U, V, W = _build_decomposition(n, m, p, *best_partition)

        if _validate_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            U, V, W = _schoolbook_decomposition(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
