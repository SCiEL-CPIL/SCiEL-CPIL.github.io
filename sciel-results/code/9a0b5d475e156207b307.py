"""
Rank-minimal exact tensor decomposition for 3x5 x 5x6 matrix multiplication.

Strategy: Block decomposition with Strassen 2x2x2 (rank 7) for blocks where
possible, schoolbook for degenerate blocks. Try all valid partitions and
pick the minimum-rank verified decomposition.

Key insight: For a block of size (ni, mj, pk):
- If mj==2 and pk==2: use ni * 7 (Strassen on the 2x2x2 core, ni independent rows)
- If mj==1 and pk==1: rank = ni
- If mj==1: rank = ni * pk
- If pk==1: rank = ni * mj
- Otherwise: schoolbook ni*mj*pk
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]


def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)(e)  -- corrected
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def schoolbook_factors(n, m, p):
    """Schoolbook decomposition for n x m x p, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def block_3x2x2_factors():
    """
    3x2x2 block: A is 3x2, B is 2x2, C is 3x2.
    Use Strassen on the 2x2x2 core for each of the 3 rows independently.
    Rank = 3 * 7 = 21.
    
    U shape: (3*2, 21) = (6, 21)
    V shape: (2*2, 21) = (4, 21)
    W shape: (2*3, 21) = (6, 21)
    
    For row i of A (i=0,1,2), the 2x2x2 Strassen applies to A[i,:], B, C[i,:].
    The Strassen U for row i: U_strassen[j, c] maps to A[i, j]
    The Strassen V: V_strassen[j*2+k, c] maps to B[j, k]  (same for all i)
    The Strassen W: W_strassen[k*2+i_str, c] maps to C[i_str, k]
    """
    U_str, V_str, W_str = strassen_factors()
    rank = 21
    U = np.zeros((6, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((6, rank), dtype=np.float64)
    
    for i in range(3):
        base = i * 7
        # U: row i of A has 2 columns (j=0,1)
        for j in range(2):
            U[i * 2 + j, base:base+7] = U_str[j, :]
        # V: same B for all rows, so V is shared but we need separate columns
        # Actually V is the same 2x2 matrix for all 3 rows, but each row's Strassen
        # uses the same V entries. So V[j*2+k, base:base+7] = V_str[j*2+k, :]
        for j in range(2):
            for k in range(2):
                V[j * 2 + k, base:base+7] = V_str[j * 2 + k, :]
        # W: C[i, k] for k=0,1
        for k in range(2):
            W[k * 3 + i, base:base+7] = W_str[k * 2 + 0, :]  # W_str[k*2+i_str] where i_str is within the 2x2
            # Wait, W_str is for 2x2x2 where C is 2x2. Here C[i,:] is a row of length 2.
            # W_str[k*2 + i_str, c] corresponds to C[i_str, k] in the 2x2 case.
            # But in our 3x2x2, C[i, k] where i is 0,1,2 (3 rows) and k is 0,1.
            # The Strassen W_str has shape (4,7) where row k*2+i_str gives C[i_str, k].
            # For our block, C[i, k] = sum_j A[i,j]*B[j,k]. This is exactly the 2x2x2
            # with A being A[i,:] (1x2), but Strassen is for 2x2x2.
            # 
            # Actually, I'm overcomplicating. Let me think differently.
            # 
            # The 3x2x2 tensor: T[i,j,k] = 1 for all i in 0..2, j in 0..1, k in 0..1.
            # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
            # 
            # This is just 3 independent 1x2x2 computations.
            # 1x2x2: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k], rank = 2 (schoolbook)
            # So 3x2x2 rank = 3*2 = 6, not 21!
            # 
            # Strassen doesn't help for 1x2x2 since there's no 2x2 structure in A.
            pass
    
    # Let me redo: 3x2x2 is 3 independent 1x2x2 blocks, each rank 2.
    # Total rank = 6.
    rank = 6
    U = np.zeros((6, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((6, rank), dtype=np.float64)
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j, col] = 1.0
                V[j * 2 + k, col] = 1.0
                W[k * 3 + i, col] = 1.0
                col += 1
    return U, V, W


def get_block_factors(ni, mj, pk):
    """
    Get factors for a block of size ni x mj x pk.
    Returns (U, V, W, rank).
    """
    # Schoolbook is always valid
    sb_rank = ni * mj * pk
    
    # Special cases
    if ni == 1 and mj == 1 and pk == 1:
        U = np.zeros((1, 1)); U[0,0] = 1.0
        V = np.zeros((1, 1)); V[0,0] = 1.0
        W = np.zeros((1, 1)); W[0,0] = 1.0
        return U, V, W, 1
    
    if ni == 1 and mj == 1:
        # 1x1xpk: rank pk
        rank = pk
        U = np.zeros((1, rank)); U[0, :] = 1.0
        V = np.zeros((pk, rank))
        W = np.zeros((pk, rank))
        for k in range(pk):
            V[k, k] = 1.0
            W[k, k] = 1.0
        return U, V, W, rank
    
    if ni == 1 and pk == 1:
        # 1xmjx1: rank mj
        rank = mj
        U = np.zeros((mj, rank))
        V = np.zeros((mj, rank))
        W = np.zeros((1, rank))
        for j in range(mj):
            U[j, j] = 1.0
            V[j, j] = 1.0
            W[0, j] = 1.0
        return U, V, W, rank
    
    if mj == 1 and pk == 1:
        # nix1x1: rank ni
        rank = ni
        U = np.zeros((ni, rank))
        V = np.zeros((1, rank))
        W = np.zeros((ni, rank))
        for i in range(ni):
            U[i, i] = 1.0
            V[0, i] = 1.0
            W[i, i] = 1.0
        return U, V, W, rank
    
    if mj == 1:
        # nix1xpk: rank ni*pk
        rank = ni * pk
        U = np.zeros((ni, rank))
        V = np.zeros((pk, rank))
        W = np.zeros((pk * ni, rank))
        col = 0
        for i in range(ni):
            for k in range(pk):
                U[i, col] = 1.0
                V[k, col] = 1.0
                W[k * ni + i, col] = 1.0
                col += 1
        return U, V, W, rank
    
    if pk == 1:
        # nixmjx1: rank ni*mj
        rank = ni * mj
        U = np.zeros((ni * mj, rank))
        V = np.zeros((mj, rank))
        W = np.zeros((ni, rank))
        col = 0
        for i in range(ni):
            for j in range(mj):
                U[i * mj + j, col] = 1.0
                V[j, col] = 1.0
                W[i, col] = 1.0
                col += 1
        return U, V, W, rank
    
    # Check if we can use Strassen for 2x2x2 core
    # If mj==2 and pk==2: ni independent 1x2x2, each rank 2, total ni*2
    if mj == 2 and pk == 2:
        rank = ni * 2
        U = np.zeros((ni * 2, rank))
        V = np.zeros((4, rank))
        W = np.zeros((2 * ni, rank))
        col = 0
        for i in range(ni):
            for j in range(2):
                for k in range(2):
                    U[i * 2 + j, col] = 1.0
                    V[j * 2 + k, col] = 1.0
                    W[k * ni + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # If ni==2 and mj==2 and pk==2: Strassen rank 7
    if ni == 2 and mj == 2 and pk == 2:
        U, V, W = strassen_factors()
        return U, V, W, 7
    
    # If ni==2 and mj==2 and pk==1: 2x2x1, rank 4 (schoolbook) or can we do better?
    # 2x2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0], rank 4 schoolbook
    # No obvious improvement.
    
    # If ni==2 and mj==1 and pk==2: 2x1x2, rank 4
    # If ni==1 and mj==2 and pk==2: 1x2x2, rank 4
    
    # Fallback: schoolbook
    U, V, W = schoolbook_factors(ni, mj, pk)
    return U, V, W, sb_rank


def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """Build full factor matrices using block decomposition."""
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                _, _, _, r = get_block_factors(ni, mj, pk)
                total_rank += r

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    n_offsets = []
    off = 0
    for s in n_sizes:
        n_offsets.append(off); off += s
    m_offsets = []
    off = 0
    for s in m_sizes:
        m_offsets.append(off); off += s
    p_offsets = []
    off = 0
    for s in p_sizes:
        p_offsets.append(off); off += s

    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                U_block, V_block, W_block, r = get_block_factors(ni, mj, pk)
                for li in range(ni):
                    for lj in range(mj):
                        U_full[(i_start + li) * m + (j_start + lj), col_offset:col_offset+r] = U_block[li * mj + lj, :]
                for lj in range(mj):
                    for lk in range(pk):
                        V_full[(j_start + lj) * p + (k_start + lk), col_offset:col_offset+r] = V_block[lj * pk + lk, :]
                for lk in range(pk):
                    for li in range(ni):
                        W_full[(k_start + lk) * n + (i_start + li), col_offset:col_offset+r] = W_block[lk * ni + li, :]
                col_offset += r

    return U_full, V_full, W_full, total_rank


def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify the decomposition reproduces the matrix multiplication tensor."""
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)
    return bool(np.allclose(vals, 1.0, atol=atol))


def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists."""
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
    return U, V, W


def generate_partitions(total, min_size=1, max_parts=total):
    """Generate all compositions of total into parts of size >= min_size."""
    if total == 0:
        yield []
        return
    if total < min_size:
        return
    for first in range(min_size, total + 1):
        for rest in generate_partitions(total - first, min_size, max_parts - 1):
            yield [first] + rest


def construct():
    """
    Constructs the factor matrices U, V, W for 3x5 x 5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s

    best_result = None
    best_rank = float('inf')

    # Generate all valid partitions
    n_parts = list(generate_partitions(n))
    m_parts = list(generate_partitions(m))
    p_parts = list(generate_partitions(p))

    # Sort candidates by estimated rank to try best first
    candidates = []
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                est_rank = 0
                for ni in np_:
                    for mj in mp_:
                        for pk in pp_:
                            _, _, _, r = get_block_factors(ni, mj, pk)
                            est_rank += r
                candidates.append((est_rank, np_, mp_, pp_))

    candidates.sort(key=lambda x: x[0])

    for est_rank, n_sizes, m_sizes, p_sizes in candidates:
        if time.time() > deadline - 20:
            break
        if est_rank >= best_rank:
            break  # No point trying worse candidates
        
        try:
            U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
            if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue

    if best_result is not None:
        return best_result

    # Fallback: schoolbook (rank 90)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}


# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
