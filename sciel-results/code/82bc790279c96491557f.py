"""Rewritten program for 4x5x6 matrix multiplication tensor rank minimization."""
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                expected = 1 if (j == j2 and i == i3 and k2 == k3) else 0
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if s != expected:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7-rank Strassen decomposition for 2x2x2 matmul.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c], each shape (4, 7).
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3: a00(b01-b11)
    U[0][2] = 1; V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4: a11(b10-b00)
    U[3][3] = 1; V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _schoolbook_block(ns, ms, ps):
    """Schoolbook decomposition for ns x ms x ps matmul. Returns U, V, W."""
    rank = ns * ms * ps
    U = [[0] * rank for _ in range(ns * ms)]
    V = [[0] * rank for _ in range(ms * ps)]
    W = [[0] * rank for _ in range(ps * ns)]
    col = 0
    for i in range(ns):
        for j in range(ms):
            for k in range(ps):
                U[i * ms + j][col] = 1
                V[j * ps + k][col] = 1
                W[k * ns + i][col] = 1
                col += 1
    return U, V, W


def _block_rank(ns, ms, ps):
    """Get the best known rank for an ns x ms x ps block."""
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    return ns * ms * ps


def _get_block_decomp(ns, ms, ps):
    """Get the best known decomposition for an ns x ms x ps block."""
    if ns == 2 and ms == 2 and ps == 2:
        return _strassen_2x2x2()
    return _schoolbook_block(ns, ms, ps)


def _build_block_decomp(n, m, p, n_splits, m_splits, p_splits):
    """Build full decomposition from block splits."""
    # Compute total rank
    total_rank = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                total_rank += _block_rank(ns, ms, ps)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Compute offsets
    n_offsets = []
    off = 0
    for ns in n_splits:
        n_offsets.append(off)
        off += ns
    m_offsets = []
    off = 0
    for ms in m_splits:
        m_offsets.append(off)
        off += ms
    p_offsets = []
    off = 0
    for ps in p_splits:
        p_offsets.append(off)
        off += ps

    col_offset = 0
    for ni, ns in enumerate(n_splits):
        for mi, ms in enumerate(m_splits):
            for pi, ps in enumerate(p_splits):
                U_b, V_b, W_b = _get_block_decomp(ns, ms, ps)
                r_b = len(U_b[0])
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]

                for i in range(ns):
                    for j in range(ms):
                        for c in range(r_b):
                            U[(n_off + i) * m + (m_off + j)][col_offset + c] = U_b[i * ms + j][c]

                for j in range(ms):
                    for k in range(ps):
                        for c in range(r_b):
                            V[(m_off + j) * p + (p_off + k)][col_offset + c] = V_b[j * ps + k][c]

                for k in range(ps):
                    for i in range(ns):
                        for c in range(r_b):
                            W[(p_off + k) * n + (n_off + i)][col_offset + c] = W_b[k * ns + i][c]

                col_offset += r_b

    return U, V, W, total_rank


def _all_splits(total, min_size=1, max_size=None):
    """Generate all compositions of 'total' into parts >= min_size."""
    if max_size is None:
        max_size = total
    if total == 0:
        yield []
        return
    for first in range(min_size, min(total, max_size) + 1):
        for rest in _all_splits(total - first, min_size, max_size):
            yield [first] + rest


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook_block(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p  # 120

    # Systematically search over all block decompositions
    # For each combination of splits, compute the total rank and keep the best
    candidates = []

    for n_splits in _all_splits(n, 1, 4):
        for m_splits in _all_splits(m, 1, 5):
            for p_splits in _all_splits(p, 1, 6):
                total_rank = 0
                for ns in n_splits:
                    for ms in m_splits:
                        for ps in p_splits:
                            total_rank += _block_rank(ns, ms, ps)
                candidates.append((total_rank, n_splits, m_splits, p_splits))

    # Sort by rank
    candidates.sort(key=lambda x: x[0])

    # Try the best candidates
    tried = set()
    for total_rank, n_splits, m_splits, p_splits in candidates:
        if total_rank >= best_rank:
            break
        if time.time() > deadline:
            break
        key = (tuple(n_splits), tuple(m_splits), tuple(p_splits))
        if key in tried:
            continue
        tried.add(key)

        U, V, W, r = _build_block_decomp(n, m, p, n_splits, m_splits, p_splits)
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found the minimum rank decomposition

    return best_result

# EVOLVE-BLOCK-END
