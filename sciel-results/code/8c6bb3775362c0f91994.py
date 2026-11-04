import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication."""
    return [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(d, [])
    return results

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits."""
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    rank += 7
                else:
                    rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits."""
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    rank = _compute_rank(n_splits, m_splits, p_splits)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    strassen_patterns = _strassen_patterns()
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off, j_off, k_off = n_off[i_b], m_off[j_b], p_off[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    for pattern in strassen_patterns:
                        for li, lj, val in pattern['U']:
                            U[(i_off + li) * m + (j_off + lj)][col] = val
                        for lj, lk, val in pattern['V']:
                            V[(j_off + lj) * p + (k_off + lk)][col] = val
                        for lk, li, val in pattern['W']:
                            W[(k_off + lk) * n + (i_off + li)][col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                U[(i_off + li) * m + (j_off + lj)][col] = 1
                                V[(j_off + lj) * p + (k_off + lk)][col] = 1
                                W[(k_off + lk) * n + (i_off + li)][col] = 1
                                col += 1
    return U, V, W

def _entries_valid(U, V, W):
    """Contract requires integer or half-integer entries only."""
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                xf = float(x)
                if xf != int(xf) and 2.0 * xf != int(2.0 * xf):
                    return False
    return True

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # safety margin before the 150s limit

    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)

    best_rank = n * m * p
    best_UVW = None

    # Precompute all split combinations and sort by predicted rank
    all_combos = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                r = _compute_rank(n_splits, m_splits, p_splits)
                all_combos.append((r, n_splits, m_splits, p_splits))
    all_combos.sort(key=lambda x: x[0])

    for r, n_splits, m_splits, p_splits in all_combos:
        if r >= best_rank:
            break
        if time.time() > deadline:
            break
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        if _entries_valid(U, V, W) and _verify_decomposition(U, V, W, n, m, p):
            best_rank = r
            best_UVW = (U, V, W)

    if best_UVW is not None:
        U, V, W = best_UVW
    else:
        U, V, W = _schoolbook_factors(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
