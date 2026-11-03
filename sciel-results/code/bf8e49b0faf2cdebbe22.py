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
            results.append(list(current)); return
        if remaining >= 1:
            current.append(1); _gen(remaining-1, current); current.pop()
        if remaining >= 2:
            current.append(2); _gen(remaining-2, current); current.pop()
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
                    rank += nb*mb*pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits."""
    n_off = [0]
    for s in n_splits[:-1]: n_off.append(n_off[-1]+s)
    m_off = [0]
    for s in m_splits[:-1]: m_off.append(m_off[-1]+s)
    p_off = [0]
    for s in p_splits[:-1]: p_off.append(p_off[-1]+s)
    rank = _compute_rank(n_splits, m_splits, p_splits)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
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
                            U[(i_off+li)*m + (j_off+lj)][col] = val
                        for lj, lk, val in pattern['V']:
                            V[(j_off+lj)*p + (k_off+lk)][col] = val
                        for lk, li, val in pattern['W']:
                            W[(k_off+lk)*n + (i_off+li)][col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                U[(i_off+li)*m + (j_off+lj)][col] = 1
                                V[(j_off+lj)*p + (k_off+lk)][col] = 1
                                W[(k_off+lk)*n + (i_off+li)][col] = 1
                                col += 1
    return U, V, W

def _entries_valid(U, V, W):
    """Contract requires integer or half-integer entries only."""
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                xf = float(x)
                if xf != int(xf) and 2.0*xf != int(2.0*xf):
                    return False
    return True

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U_np.shape[1]):
                    s += U_np[i*m+j, r]*V_np[j*p+k, r]*W_np[k*n+i, r]
                if not np.isclose(s, 1.0):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _dft_circulant_candidate(n, m, p):
    """Structural experiment (new modelling family): treat the n x m x p product
    as n independent 1 x m x p row-vector/matrix products, each attacked with a
    rank-m bilinear algorithm derived from the DFT over the cyclic group Z_m,
    exploiting circulant structure. Returns (U,V,W) or None if the construction
    cannot be expressed with integer/half-integer entries (the DFT roots of
    unity for m=5 are not integral, so this is expected to be rejected)."""
    omega = np.exp(2j*np.pi/m)
    rank = n*m
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for t in range(m):
            for j in range(m):
                U[i*m+j][col] = float(omega**(j*t).real)
            for j in range(m):
                for k in range(p):
                    V[j*p+k][col] = float(omega**((k-j)*t).real)
            for k in range(p):
                W[k*n+i][col] = float(omega**(k*t).real)
            col += 1
    if not _entries_valid(U, V, W):
        return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 110  # safety margin before the 120s/150s limit

    candidates = []  # list of (rank, (U, V, W))

    # 1) New structural direction: DFT/circulant rank-(n*m) candidate.
    dft = _dft_circulant_candidate(n, m, p)
    if dft is not None and _verify_decomposition(*dft, n, m, p):
        candidates.append((len(dft[0][0]), dft))

    # 2) Proven Strassen 2x2x2 block method (guaranteed rank 71 for 3x5x5).
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    best_rank = n*m*p
    best_UVW = None
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                if time.time() > deadline - 15:
                    break
                rank = _compute_rank(n_splits, m_splits, p_splits)
                if rank < best_rank:
                    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
                    if _entries_valid(U, V, W) and _verify_decomposition(U, V, W, n, m, p):
                        best_rank = rank
                        best_UVW = (U, V, W)
    if best_UVW is not None:
        candidates.append((best_rank, best_UVW))

    # 3) Fallback to schoolbook if nothing verified (should not happen).
    if not candidates:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

    candidates.sort(key=lambda x: x[0])
    _, (U, V, W) = candidates[0]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
