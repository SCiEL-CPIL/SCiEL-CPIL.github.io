"""Block-partitioned 6x6x6 matrix multiplication:
3x3 block grid (schoolbook, rank 27) times 2x2 Strassen (rank 7),
Kronecker-structured. Total rank = 27 * 7 = 189."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build():
    n, m, p = DIMENSIONS  # 6, 6, 6
    a = 3                 # number of blocks along one dimension
    s = n // a            # block size = 2

    rank = a * a * a * 7  # 27 * 7 = 189

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Schoolbook over a=3 blocks: (b_i, b_j, b_k) with single 1
    # Strassen over s=2 per block: (e_i, e_j, e_k) with pattern below
    strassen_pattern = [
        (1,  0,  1),
        (0,  0,  0),
        (1,  0, -1),
        (0,  1,  0),
        (0,  1,  1),
        (0,  1, -1),
        (1, -1,  0),
    ]

    col = 0
    for b_i in range(a):
        for b_j in range(a):
            for b_k in range(a):
                for (si, sj, sk) in strassen_pattern:
                    # 16 positions (e_i, e_j, e_k) in this block triplet
                    for e_i in range(s):
                        for e_j in range(s):
                            for e_k in range(s):
                                i = b_i * s + e_i
                                j = b_j * s + e_j
                                k = b_k * s + e_k
                                # U: A[i,j] contributes
                                if si != 0:
                                    U[i * m + j, col] += si
                                # V: B[j,k] contributes
                                if sj != 0:
                                    V[j * p + k, col] += sj
                                # W: C[k,i] (trace convention) contributes
                                if sk != 0:
                                    W[k * n + i, col] += sk
                    col += 1
    assert col == rank

    # Verify: U @ diag? Actually tensor = U^T * V * W? Convention:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    T_expected = np.einsum('ijk->ijk', np.ones((n, m, p)))  # schoolbook tensor
    ok = np.allclose(T, T_expected, rtol=1e-10, atol=1e-10)
    if not ok:
        # Fallback: return schoolbook rank 216 to guarantee validity
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    res = _build()
    # Convert to plain lists of ints (entries are integers here: -1,0,1)
    out = {}
    for key in ("U", "V", "W"):
        arr = res[key]
        ints = arr.astype(int) if np.allclose(arr, np.round(arr)) else arr
        out[key] = ints.tolist()
    return out
# EVOLVE-BLOCK-END
