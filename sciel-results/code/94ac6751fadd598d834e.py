"""Tensor decomposition for 2x3x9 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 9]

# Strassen decomposition for 2x2x2: 7 terms
# U[i*2+j], V[j*2+k], W[k*2+i]
_STRASSEN_2x2x2 = [
    # (u, v, w) as (list of (idx, coeff), ...)
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)]),
    ([(2, 1)], [(1, 1), (3, 1)], [(1, 1), (3, 1)]),
    ([(0, 1), (2, 1)], [(0, 1)], [(0, 1), (2, 1)]),
    ([(1, 1), (3, 1)], [(2, 1)], [(0, 1), (2, -1)]),
    ([(0, 1)], [(0, 1), (2, 1)], [(0, 1), (2, -1)]),
    ([(1, 1), (3, 1)], [(2, 1), (3, 1)], [(2, 1), (3, 1)]),
    ([(0, 1), (3, 1)], [(0, 1), (1, -1)], [(0, 1), (3, 1)]),
]

def _build_2x3x3_decomp():
    """Build rank-13 decomposition for 2x3x3.
    
    U has 6 rows (i*3+j), V has 9 rows (j*3+k), W has 6 rows (k*3+i).
    Uses Strassen on (i in {0,1}, j in {0,1}, k in {0,1}) + schoolbook for j=2.
    """
    terms = []
    # Strassen terms for j in {0,1}, k in {0,1}
    for u_raw, v_raw, w_raw in _STRASSEN_2x2x2:
        u = [(idx, coeff) for idx, coeff in u_raw]
        # v_raw indices are j*2+k where j,k in {0,1}; map to j*3+k
        v = []
        for idx, coeff in v_raw:
            j = idx // 2
            k = idx % 2
            v.append((j * 3 + k, coeff))
        # w_raw indices are k*2+i where k,i in {0,1}; map to k*3+i
        w = []
        for idx, coeff in w_raw:
            k = idx // 2
            i = idx % 2
            w.append((k * 3 + i, coeff))
        terms.append((u, v, w))
    
    # Schoolbook for j=2, k in {0,1,2}
    for i in range(2):
        for k in range(3):
            u = [(i * 3 + 2, 1)]
            v = [(2 * 3 + k, 1)]
            w = [(k * 3 + i, 1)]
            terms.append((u, v, w))
    
    return terms  # 7 + 6 = 13 terms

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build 2x3x3 decomposition (rank 13)
    base_terms = _build_2x3x3_decomp()
    base_rank = len(base_terms)  # 13
    
    # For 2x3x9: split B's 9 columns into 3 blocks of 3
    # Each block b has columns k in {3b, 3b+1, 3b+2}
    # Total rank = 3 * 13 = 39
    total_rank = base_rank * 3
    
    U = [[0] * total_rank for _ in range(n * m)]   # 6 rows
    V = [[0] * total_rank for _ in range(m * p)]    # 27 rows
    W = [[0] * total_rank for _ in range(p * n)]    # 18 rows
    
    for block in range(3):
        col_offset = block * base_rank
        for term_idx, (u, v, w) in enumerate(base_terms):
            col = col_offset + term_idx
            for idx, coeff in u:
                U[idx][col] += coeff
            for idx, coeff in v:
                # v indices are j*3+k_local where k_local in {0,1,2}
                # Global k = k_local + 3*block
                j = idx // 3
                k_local = idx % 3
                k_global = k_local + 3 * block
                V[j * p + k_global][col] += coeff
            for idx, coeff in w:
                # w indices are k_local*3+i where k_local in {0,1,2}
                # Global k = k_local + 3*block
                k_local = idx // 3
                i = idx % 3
                k_global = k_local + 3 * block
                W[k_global * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
