import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n=2, m=5, p=5):
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


def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1):
    """Add Strassen 2x2*2x2 block (7 multiplications)."""
    a00 = 0 * 5 + k0
    a01 = 0 * 5 + k1
    a10 = 1 * 5 + k0
    a11 = 1 * 5 + k1

    b00 = k0 * 5 + j0
    b01 = k0 * 5 + j1
    b10 = k1 * 5 + j0
    b11 = k1 * 5 + j1

    c00 = j0 * 2 + 0
    c01 = j1 * 2 + 0
    c10 = j0 * 2 + 1
    c11 = j1 * 2 + 1

    U[a00][offset + 0] += 1
    U[a11][offset + 0] += 1
    V[b00][offset + 0] += 1
    V[b11][offset + 0] += 1
    W[c00][offset + 0] += 1
    W[c11][offset + 0] += 1

    U[a10][offset + 1] += 1
    U[a11][offset + 1] += 1
    V[b00][offset + 1] += 1
    W[c10][offset + 1] += 1
    W[c11][offset + 1] += -1

    U[a00][offset + 2] += 1
    V[b01][offset + 2] += 1
    V[b11][offset + 2] += -1
    W[c01][offset + 2] += 1
    W[c11][offset + 2] += 1

    U[a11][offset + 3] += 1
    V[b10][offset + 3] += 1
    V[b00][offset + 3] += -1
    W[c00][offset + 3] += 1
    W[c10][offset + 3] += 1

    U[a00][offset + 4] += 1
    U[a01][offset + 4] += 1
    V[b11][offset + 4] += 1
    W[c00][offset + 4] += -1
    W[c01][offset + 4] += 1

    U[a10][offset + 5] += 1
    U[a00][offset + 5] += -1
    V[b00][offset + 5] += 1
    V[b01][offset + 5] += 1
    W[c11][offset + 5] += 1

    U[a01][offset + 6] += 1
    U[a11][offset + 6] += -1
    V[b10][offset + 6] += 1
    V[b11][offset + 6] += 1
    W[c00][offset + 6] += 1

    return offset + 7


def _add_2x1(U, V, W, offset, k0, k1, j):
    a0k0 = 0 * 5 + k0
    a0k1 = 0 * 5 + k1
    a1k0 = 1 * 5 + k0
    a1k1 = 1 * 5 + k1
    bk0j = k0 * 5 + j
    bk1j = k1 * 5 + j
    c0j = j * 2 + 0
    c1j = j * 2 + 1

    U[a0k0][offset + 0] += 1
    V[bk0j][offset + 0] += 1
    W[c0j][offset + 0] += 1
    U[a0k1][offset + 1] += 1
    V[bk1j][offset + 1] += 1
    W[c0j][offset + 1] += 1
    U[a1k0][offset + 2] += 1
    V[bk0j][offset + 2] += 1
    W[c1j][offset + 2] += 1
    U[a1k1][offset + 3] += 1
    V[bk1j][offset + 3] += 1
    W[c1j][offset + 3] += 1

    return offset + 4


def _add_1x2(U, V, W, offset, k, j0, j1):
    a0k = 0 * 5 + k
    a1k = 1 * 5 + k
    bkj0 = k * 5 + j0
    bkj1 = k * 5 + j1
    c0j0 = j0 * 2 + 0
    c0j1 = j1 * 2 + 0
    c1j0 = j0 * 2 + 1
    c1j1 = j1 * 2 + 1

    U[a0k][offset + 0] += 1
    V[bkj0][offset + 0] += 1
    W[c0j0][offset + 0] += 1
    U[a0k][offset + 1] += 1
    V[bkj1][offset + 1] += 1
    W[c0j1][offset + 1] += 1
    U[a1k][offset + 2] += 1
    V[bkj0][offset + 2] += 1
    W[c1j0][offset + 2] += 1
    U[a1k][offset + 3] += 1
    V[bkj1][offset + 3] += 1
    W[c1j1][offset + 3] += 1

    return offset + 4


def _add_1x1(U, V, W, offset, k, j):
    a0k = 0 * 5 + k
    a1k = 1 * 5 + k
    bkj = k * 5 + j
    c0j = j * 2 + 0
    c1j = j * 2 + 1

    U[a0k][offset + 0] += 1
    V[bkj][offset + 0] += 1
    W[c0j][offset + 0] += 1
    U[a1k][offset + 1] += 1
    V[bkj][offset + 1] += 1
    W[c1j][offset + 1] += 1

    return offset + 2


def _add_schoolbook_block(U, V, W, offset, k_group, j_group):
    """Add schoolbook for a block of size nk x nj (n=2 fixed)."""
    nk = len(k_group)
    nj = len(j_group)
    for ii in range(2):
        for jj in range(nj):
            for kk in range(nk):
                a = ii * 5 + k_group[kk]
                b = k_group[kk] * 5 + j_group[jj]
                c = j_group[jj] * 2 + ii
                U[a][offset] += 1
                V[b][offset] += 1
                W[c][offset] += 1
                offset += 1
    return offset


def _block_rank(nk, nj):
    """Return rank for a block of nk k-values and nj j-values."""
    if nk == 2 and nj == 2:
        return 7
    elif nk == 2 and nj == 1:
        return 4
    elif nk == 1 and nj == 2:
        return 4
    elif nk == 1 and nj == 1:
        return 2
    else:
        return 2 * nk * nj  # schoolbook


def _build_partition(k_groups, j_groups):
    """Build decomposition for given partition."""
    n, m, p = 2, 5, 5
    total_rank = 0
    for kg in k_groups:
        for jg in j_groups:
            total_rank += _block_rank(len(kg), len(jg))

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    offset = 0
    for kg in k_groups:
        for jg in j_groups:
            nk, nj = len(kg), len(jg)
            if nk == 2 and nj == 2:
                offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1])
            elif nk == 2 and nj == 1:
                offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0])
            elif nk == 1 and nj == 2:
                offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1])
            elif nk == 1 and nj == 1:
                offset = _add_1x1(U, V, W, offset, kg[0], jg[0])
            else:
                offset = _add_schoolbook_block(U, V, W, offset, kg, jg)

    return U, V, W


def _all_partitions():
    """Generate all possible partitions of {0,1,2,3,4} into contiguous groups."""
    def gen_partitions(n):
        if n == 0:
            yield []
        else:
            for i in range(1, n + 1):
                for rest in gen_partitions(n - i):
                    yield list(range(n - i, n)) + rest

    return list(gen_partitions(5))


def construct():
    n, m, p = 2, 5, 5

    U_school, V_school, W_school = _schoolbook(n, m, p)

    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}

    best_result = None
    best_rank = n * m * p

    # Try the known good partition first
    k_parts = [[0, 1], [2, 3], [4]]
    j_parts = [[0, 1], [2, 3], [4]]
    U, V, W = _build_partition(k_parts, j_parts)
    if _verify_decomp(n, m, p, U, V, W):
        best_rank = len(U[0])
        best_result = {"U": U, "V": V, "W": W}

    if not _deadline_ok():
        return best_result

    # Try other partitions
    partitions = _all_partitions()
    for kp in partitions:
        if not _deadline_ok():
            break
        for jp in partitions:
            if not _deadline_ok():
                break
            # Skip if same as known good
            if kp == k_parts and jp == j_parts:
                continue
            total_r = sum(_block_rank(len(kg), len(jg)) for kg in kp for jg in jp)
            if total_r >= best_rank:
                continue
            U, V, W = _build_partition(kp, jp)
            if _verify_decomp(n, m, p, U, V, W):
                best_rank = total_r
                best_result = {"U": U, "V": V, "W": W}

    return best_result if best_result else {"U": U_school, "V": V_school, "W": W_school}
