"""Matrix multiplication 4x5x5 with Strassen-hierarchy tensor network ansatz."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2(i1, i2, j1, j2, k1, k2, n, m, p):
    """Return 7 rank-1 terms for Strassen 2x2x2 block."""
    terms = []
    # m1 = (a_i1j1 + a_i2j2) * (b_j1k1 + b_j2k2)
    terms.append({
        'U': [(i1*m+j1, 1), (i2*m+j2, 1)],
        'V': [(j1*p+k1, 1), (j2*p+k2, 1)],
        'W': [(k1*n+i1, 1), (k2*n+i1, -1), (k1*n+i2, -1), (k2*n+i2, 1)]
    })
    # m2 = (a_i2j1 + a_i2j2) * b_j1k1
    terms.append({
        'U': [(i2*m+j1, 1), (i2*m+j2, 1)],
        'V': [(j1*p+k1, 1)],
        'W': [(k1*n+i2, 1), (k2*n+i2, -1)]
    })
    # m3 = a_i1j1 * (b_j2k1 - b_j2k2)
    terms.append({
        'U': [(i1*m+j1, 1)],
        'V': [(j2*p+k1, 1), (j2*p+k2, -1)],
        'W': [(k1*n+i1, 1), (k2*n+i1, -1)]
    })
    # m4 = a_i2j2 * (b_j1k1 - b_j1k2)
    terms.append({
        'U': [(i2*m+j2, 1)],
        'V': [(j1*p+k1, 1), (j1*p+k2, -1)],
        'W': [(k1*n+i1, -1), (k1*n+i2, 1), (k2*n+i2, -1)]
    })
    # m5 = (a_i1j1 + a_i1j2) * b_j2k2
    terms.append({
        'U': [(i1*m+j1, 1), (i1*m+j2, 1)],
        'V': [(j2*p+k2, 1)],
        'W': [(k2*n+i1, 1)]
    })
    # m6 = (a_i2j1 - a_i1j1) * (b_j1k1 + b_j1k2)
    terms.append({
        'U': [(i2*m+j1, 1), (i1*m+j1, -1)],
        'V': [(j1*p+k1, 1), (j1*p+k2, 1)],
        'W': [(k1*n+i2, -1), (k2*n+i2, 1)]
    })
    # m7 = (a_i1j2 - a_i2j2) * (b_j2k1 + b_j2k2)
    terms.append({
        'U': [(i1*m+j2, 1), (i2*m+j2, -1)],
        'V': [(j2*p+k1, 1), (j2*p+k2, 1)],
        'W': [(k1*n+i1, -1), (k2*n+i1, 1), (k2*n+i2, 1)]
    })
    return terms

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]

    terms = []

    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                len_i = len(i_blk)
                len_j = len(j_blk)
                len_k = len(k_blk)

                if len_i == 2 and len_j == 2 and len_k == 2:
                    i1, i2 = i_blk
                    j1, j2 = j_blk
                    k1, k2 = k_blk
                    terms.extend(_strassen_2x2x2(i1, i2, j1, j2, k1, k2, n, m, p))
                else:
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                terms.append({
                                    'U': [(i*m+j, 1)],
                                    'V': [(j*p+k, 1)],
                                    'W': [(k*n+i, 1)]
                                })

    rank = len(terms)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)

    for r, term in enumerate(terms):
        for idx, val in term['U']:
            U[idx, r] = val
        for idx, val in term['V']:
            V[idx, r] = val
        for idx, val in term['W']:
            W[idx, r] = val

    if time.time() > deadline:
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify correctness
    tensor_rec = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        tensor_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                max_err = max(max_err, abs(tensor_rec[i*m+j, j*p+k, k*n+i] - 1.0))

    if max_err > 1e-10:
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
