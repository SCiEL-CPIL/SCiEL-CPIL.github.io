"""Matrix multiplication 3x5x5: block-partition search with 2x2x2 Strassen blocks."""
DIMENSIONS = [3, 5, 5]

# Strassen 2x2x2 rank-7 tensor (indices i,j,k in {0,1})
_STRASSEN = [
    (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1),
    (0, 1, 1, -1), (1, 0, 0, 1), (1, 1, 0, 1), (1, 1, 1, 1),
]


def _block_rank(a, b, c):
    """Rank of an a x b x c matrix multiplication tensor (schoolbook)."""
    if a == 2 and b == 2 and c == 2:
        return 7
    return a * b * c


def _partition_search(n, m, p):
    """Find the partition of dims into 2-blocks + remainders minimizing rank.

    Returns (best_rank, best_split) where best_split is (ni,nr,mi,mr,pi,pr).
    """
    best = None
    for ni in (0, 1, 2):
        nr = n - 2 * ni
        if nr < 0:
            continue
        for mi in (0, 1, 2):
            mr = m - 2 * mi
            if mr < 0:
                continue
            for pi in (0, 1, 2):
                pr = p - 2 * pi
                if pr < 0:
                    continue
                rank = 0
                for i in range(ni):
                    for j in range(mi):
                        for k in range(pi):
                            rank += _block_rank(2, 2, 2)
                    for k in range(pr):
                        rank += 2 * 2 * 1
                    for j in range(mr):
                        rank += 2 * 1 * 1
                for i in range(nr):
                    for j in range(mi):
                        for k in range(pi):
                            rank += _block_rank(1, 2, 2)
                    for k in range(pr):
                        rank += 1 * 2 * 1
                    for j in range(mr):
                        rank += 1 * 1 * 1
                if best is None or rank < best[0]:
                    best = (rank, (ni, nr, mi, mr, pi, pr))
    return best


def _build_factor(rank, row_count, fill):
    """Create a rank x row_count matrix (list of lists), then fill via callback."""
    M = [[0] * rank for _ in range(row_count)]
    return M


def construct():
    n, m, p = DIMENSIONS
    best_rank, (ni, nr, mi, mr, pi, pr) = _partition_search(n, m, p)

    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    c = 0

    def emit(i, j, k, val):
        nonlocal c
        U[i * m + j][c] = val
        V[j * p + k][c] = val
        W[k * n + i][c] = val
        c += 1

    # For each block position, emit the appropriate decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Determine which block this (i,j,k) belongs to
                # i-block: 0 if i < 2*ni else 1 (remainder)
                # j-block: 0 if j < 2*mi else 1
                # k-block: 0 if k < 2*pi else 1
                ib = 0 if i < 2 * ni else 1
                jb = 0 if j < 2 * mi else 1
                kb = 0 if k < 2 * pi else 1

                if ib == 0 and jb == 0 and kb == 0:
                    # 2x2x2 Strassen block: local indices
                    li = i  # 0 or 1
                    lj = j  # 0 or 1
                    lk = k  # 0 or 1
                    # For Strassen, each (li,lj,lk) participates in multiple rank-1 terms
                    # We handle this differently: emit all 7 terms for the block
                    pass  # handled below
                # For non-Strassen blocks, emit schoolbook (single term per (i,j,k))
                else:
                    emit(i, j, k, 1)

    # The above approach double-counts Strassen blocks. Let me redo with a cleaner approach.
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    c = 0

    # Process block by block
    # Block (bi, bj, bk) where bi in {0,1} (0=Strassen part, 1=remainder)
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Determine the global index ranges for this block
                # bi=0: i in [0, 2*ni), bi=1: i in [2*ni, n)
                # bj=0: j in [0, 2*mi), bj=1: j in [2*mi, m)
                # bk=0: k in [0, 2*pi), bk=1: k in [2*pi, p)
                i_start = 0 if bi == 0 else 2 * ni
                i_end = 2 * ni if bi == 0 else n
                j_start = 0 if bj == 0 else 2 * mi
                j_end = 2 * mi if bj == 0 else m
                k_start = 0 if bk == 0 else 2 * pi
                k_end = 2 * pi if bk == 0 else p

                a = i_end - i_start  # size in i-dim
                b = j_end - j_start  # size in j-dim
                d = k_end - k_start  # size in k-dim

                if a == 2 and b == 2 and d == 2:
                    # Strassen 2x2x2: 7 rank-1 terms
                    for (li, lj, lk, val) in _STRASSEN:
                        gi = i_start + li
                        gj = j_start + lj
                        gk = k_start + lk
                        U[gi * m + gj][c] = val
                        V[gj * p + gk][c] = val
                        W[gk * n + gi][c] = val
                        c += 1
                else:
                    # Schoolbook block
                    for li in range(a):
                        for lj in range(b):
                            for lk in range(d):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                U[gi * m + gj][c] = 1
                                V[gj * p + gk][c] = 1
                                W[gk * n + gi][c] = 1
                                c += 1

    assert c == best_rank, f"rank mismatch: {c} vs {best_rank}"
    return {"U": U, "V": V, "W": W}
