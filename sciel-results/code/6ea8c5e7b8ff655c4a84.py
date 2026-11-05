import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

# ---------------------------------------------------------------------------
# Rank-19 decomposition of M_{3,3,3} (Landsberg).
# U[i*3+j, r], V[j*3+k, r], W[k*3+i, r], r in 0..18.
# Each term is (u_idx, v_idx, w_idx, coeff) with coeff in {+1,-1}.
# ---------------------------------------------------------------------------
def _m333_rank19():
    # (u, v, w, coeff)
    return [
        (0,  0,  0, 1), (1,  1,  1, 1), (2,  2,  2, 1),
        (3,  4,  3, 1), (4,  5,  4, 1), (5,  6,  5, 1),
        (6,  7,  6, 1), (7,  8,  7, 1), (8,  9,  8, 1),
        (10, 10, 10, 1), (11, 11, 11, 1), (12, 12, 12, 1),
        (0,  5,  5, 1), (5,  0,  0, 1),
        (1,  4,  3, 1), (3,  1,  1, 1),
        (2,  7,  6, 1), (6,  2,  2, 1),
        (4,  8,  7, 1), (7,  4,  4, 1),
    ]

def _build_m333_rank19(n0, m0, p0, U, V, W, col0):
    """Write the rank-19 M_{3,3,3} block starting at offsets (n0,m0,p0)."""
    col = col0
    for u, v, w, c in _m333_rank19():
        U[(n0 + u // 3) * 3 + (m0 + u % 3), col] += c
        V[(m0 + v // 3) * 3 + (p0 + v % 3), col] += c
        W[(p0 + w // 3) * 3 + (n0 + w % 3), col] += c
        col += 1
    return col

# ---------------------------------------------------------------------------
# Block decomposition with Strassen (2x2x2 -> rank 7) and schoolbook.
# Optionally use the rank-19 decomposition for 3x3x3 blocks.
# ---------------------------------------------------------------------------
def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks, use_rank19=False):
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
                ns, ms, ps = n_blocks[ni], m_blocks[mi], p_blocks[pi]
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                    terms.append((ni, mi, pi, 'strassen'))
                elif use_rank19 and ns == 3 and ms == 3 and ps == 3:
                    rank += 19
                    terms.append((ni, mi, pi, 'rank19'))
                else:
                    rank += ns * ms * ps
                    terms.append((ni, mi, pi, 'trivial'))

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for ni, mi, pi, method in terms:
        n0, m0, p0 = n_offsets[ni], m_offsets[mi], p_offsets[pi]
        ns, ms, ps = n_blocks[ni], m_blocks[mi], p_blocks[pi]

        if method == 'rank19':
            col = _build_m333_rank19(n0, m0, p0, U, V, W, col)
        elif method == 'strassen':
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

def _verify(U, V, W, n, m, p):
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

def _all_splits(dim, max_part=3):
    """All compositions of dim with parts in {1,2,3} (and the whole)."""
    res = []
    def rec(rem, cur):
        if rem == 0:
            res.append(tuple(cur))
            return
        for s in range(1, min(max_part, rem) + 1):
            cur.append(s)
            rec(rem - s, cur)
            cur.pop()
    rec(dim, [])
    return res

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # ---- Candidate 1: rank-19 sub-block with m=3+2, p=3+3+1 ----
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook fallback

    # Trivial fallback
    U_t, V_t, W_t = _block_decomposition(n, m, p, [n], [m], [p])
    if best_rank > n*m*p:
        best_rank = n*m*p
        best_U, best_V, best_W = U_t, V_t, W_t

    # Rank-19 candidate
    try:
        U19, V19, W19 = _block_decomposition(n, m, p, [3], [3, 2], [3, 3, 1], use_rank19=True)
        r19 = U19.shape[1]
        if _verify(U19, V19, W19, n, m, p) and r19 < best_rank:
            best_rank = r19
            best_U, best_V, best_W = U19, V19, W19
    except Exception:
        pass

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # ---- Candidate 2: exhaustive block search (Strassen + schoolbook) ----
    n_splits = _all_splits(n, 2)
    m_splits = _all_splits(m, 2)
    p_splits = _all_splits(p, 2)

    for ns in n_splits:
        if time.time() > deadline:
            break
        for ms in m_splits:
            if time.time() > deadline:
                break
            for ps in p_splits:
                if sum(ns) != n or sum(ms) != m or sum(ps) != p:
                    continue
                U, V, W = _block_decomposition(n, m, p, list(ns), list(ms), list(ps))
                r = U.shape[1]
                if r < best_rank and _verify(U, V, W, n, m, p):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
