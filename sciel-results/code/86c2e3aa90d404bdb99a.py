"""Matrix multiplication tensor decomposition for 3x4 * 4x7."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2. Rank 7."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise."""
    total_rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    total_rank += 7
                else:
                    total_rank += sn * sm * sp
    
    if total_rank == 0:
        return None, None, None, 0
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = nb[ci]
                                j = mb[cj]
                                k = pb[ck]
                                for c in range(7):
                                    U[i * m + j, col + c] = U_s[ci * 2 + cj, c]
                                    V[j * p + k, col + c] = V_s[cj * 2 + ck, c]
                                    W[k * n + i, col + c] = W_s[ck * 2 + ci, c]
                    col += 7
                else:
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = nb[ci]
                                j = mb[cj]
                                k = pb[ck]
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
    
    return U, V, W, total_rank

def _try_all_block_partitionings(n, m, p, deadline):
    """Try all possible block partitionings to find the minimum rank."""
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook
    
    def get_partitions(size):
        """Generate all ways to partition [0, size) into contiguous blocks of size 1 or 2."""
        partitions = []
        def _gen(pos, current):
            if pos == size:
                partitions.append(tuple(current))
                return
            if pos + 2 <= size:
                current.append((pos, pos + 1))
                _gen(pos + 2, current)
                current.pop()
            current.append((pos,))
            _gen(pos + 1, current)
            current.pop()
        _gen(0, [])
        return partitions
    
    n_parts = get_partitions(n)
    m_parts = get_partitions(m)
    p_parts = get_partitions(p)
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                elapsed = time.time()
                if elapsed > deadline - 5:
                    break
                U, V, W, rank = _build_block_decomposition(n, m, p, np_, mp_, pp_)
                if rank < best_rank:
                    if _check_decomposition(U, V, W, n, m, p):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
    
    return best_U, best_V, best_W, best_rank

def _try_3x2x2_block(n, m, p, deadline):
    """
    Try a specific decomposition that uses a 3x2x2 block with rank 10 (instead of 12).
    This is a known improvement for 3x2x2: use Strassen on the top 2x2 and handle the
    remaining 1x2 with 2 terms, but with some overlap.
    
    Actually, for 3x2x2, the rank is 10 (not 12). Let's construct it.
    The 3x2x2 tensor has 12 entries. We can use:
    - Strassen on the top 2x2 (rows 0,1) with rank 7
    - 3 terms for the bottom row (row 2): 1x2x2 = 2 terms? No, 1x2x2 = 2 terms.
    Wait, 1x2x2 is just 2 scalar multiplications. So total 7+2=9? No, the Strassen 
    decomposition for 2x2x2 handles the top 2x2 part, but the bottom row interacts 
    with the same columns.
    
    Actually, the rank of 3x2x2 is 10. Let's just use the block decomposition with 
    partition (2,1) for n, (2) for m, (2) for p:
    - 2x2x2 block: rank 7
    - 1x2x2 block: rank 2
    Total: 9. But wait, 1x2x2 is 2 terms, so total 9.
    
    But the schoolbook for 3x2x2 is 12. So 9 is better.
    
    Let's just rely on the block partitioning which already handles this.
    """
    return None, None, None, n * m * p

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try all block partitionings
    U, V, W, rank = _try_all_block_partitionings(n, m, p, deadline)
    if U is not None and rank < best_rank:
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Verify the best decomposition
    if _check_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
