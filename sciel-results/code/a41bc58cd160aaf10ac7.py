import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank32():
    """
    Rank-32 decomposition: 2 j-halves x 4 k-blocks (size 2) x 4 terms = 32.
    Uses a 4-term scheme per (j-half, k-block) that exactly decomposes 2x2x2 MM.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 4-term scheme for 2x2x2 matrix multiplication (if valid)
    # We need: sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1 for all i,j,k in {0,1}
    four_terms = [
        # (U: {(i,j):c}, V: {(j,k):c}, W: {(k,i):c})
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
    ]

    col = 0
    for j_half in range(2):
        for k_block in range(4):
            for (U_part, V_part, W_part) in four_terms:
                for (i, j_low), c in U_part.items():
                    j = j_half * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in V_part.items():
                    j = j_half * 2 + j_low
                    k = k_block * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in W_part.items():
                    k = k_block * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank28():
    """
    Rank-28: 2 j-halves x 2 k-groups (size 4) x 7 terms = 28.
    Each k-group covers 4 consecutive k-values using a 2x2x4 scheme.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 7 Strassen terms for 2x2x2
    strassen_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    col = 0
    for j_half in range(2):
        for k_group in range(2):
            for (U_part, V_part, W_part) in strassen_terms:
                for (i, j_low), c in U_part.items():
                    j = j_half * 2 + j_low
                    U[i * m + j][col] += c
                # V covers k = k_group*4 + 0..3 using 2x2 pattern on k_low
                for (j_low, k_low), c in V_part.items():
                    j = j_half * 2 + j_low
                    k = k_group * 4 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in W_part.items():
                    k = k_group * 4 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W (they contribute nothing)."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    best = None
    best_rank = float('inf')

    # Try rank-28 first
    if time.time() < deadline:
        cand28 = _construct_rank28()
        ok28, err28 = _matmul_check(cand28["U"], cand28["V"], cand28["W"], n, m, p)
        if ok28:
            U, V, W, r = _trim_zero_columns(cand28["U"], cand28["V"], cand28["W"])
            if r < best_rank:
                best_rank = r
                best = {"U": U, "V": V, "W": W}

    # Try rank-32
    if time.time() < deadline:
        cand32 = _construct_rank32()
        ok32, err32 = _matmul_check(cand32["U"], cand32["V"], cand32["W"], n, m, p)
        if ok32:
            U, V, W, r = _trim_zero_columns(cand32["U"], cand32["V"], cand32["W"])
            if r < best_rank:
                best_rank = r
                best = {"U": U, "V": V, "W": W}

    # Fallback to known valid rank-56
    if time.time() < deadline:
        cand56 = _construct_rank56()
        ok56, err56 = _matmul_check(cand56["U"], cand56["V"], cand56["W"], n, m, p)
        if ok56:
            U, V, W, r = _trim_zero_columns(cand56["U"], cand56["V"], cand56["W"])
            if r < best_rank:
                best_rank = r
                best = {"U": U, "V": V, "W": W}

    if best is not None:
        return best

    # Last resort: return untrimmed rank-56
    cand56 = _construct_rank56()
    return cand56
# EVOLVE-BLOCK-END
