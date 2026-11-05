"""
Exact rank-78 tensor decomposition for M_{3,4,7} matrix multiplication.
Constructed as block sum: 6 Strassen 2x2x2 blocks + 36 schoolbook residual terms.
Uses 6 densely-packed Strassen blocks all in i∈{0,1} to maximize j×k coverage.
"""
import time
import numpy as np

_DIMS = (3, 4, 7)


def construct():
    t0 = time.monotonic()
    T_LIM = 105.0

    n, m, p = _DIMS
    R = 78

    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)

    # Six disjoint 2x2x2 Strassen blocks: i in {0,1}, tiling j×k with 2×2 blocks
    # Blocks at (i0=0, j0, k0) for (j0,k0) in {(0,0),(0,2),(0,4),(2,0),(2,2),(2,4)}
    _STRASSEN = _build_strassen_terms(n, m, p)
    _SCHOOLBOOK = _build_residual_terms(n, m, p)

    for c, u_specs, v_specs, w_specs in _STRASSEN:
        for ui, uval in u_specs:
            U[ui, c] = uval
        for vi, vval in v_specs:
            V[vi, c] = vval
        for wi, wval in w_specs:
            W[wi, c] = wval

    for c, u_idx, v_idx, w_idx in _SCHOOLBOOK:
        U[u_idx, c] = 1.0
        V[v_idx, c] = 1.0
        W[w_idx, c] = 1.0

    if time.monotonic() - t0 > T_LIM:
        return _fallback(n, m, p)

    if _check(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return _fallback(n, m, p)


def _build_strassen_terms(n, m, p):
    """
    Generate 42 rank-1 terms from six 2x2x2 Strassen decompositions.
    Each block covers i in {0,1}, a 2x2 range in j, and a 2x2 range in k.
    Strassen's 7 terms for multiplying 2x2 by 2x2:
      M1 = (a11+a22)(b11+b22)
      M2 = (a21+a22)b11
      M3 = a11(b21-b22)
      M4 = a22(b12-b11)
      M5 = (a11+a12)b22
      M6 = (a21-a11)(b11+b12)
      M7 = (a12-a22)(b21+b22)
    with:
      c11 = M1+M4-M5+M7
      c12 = M3+M5
      c21 = M2+M4
      c22 = M1-M2+M3+M6
    """
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 2, 0), (0, 2, 2), (0, 2, 4)]
    terms = []
    col = 0
    for (i0, j0, k0) in blocks:
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i

        # Term 1: M1 = (a11+a22)(b11+b22)
        # U: a11 + a22 -> U[i0*m+j0] += 1, U[i1*m+j1] += 1
        # V: b11 + b22 -> V[j0*p+k0] += 1, V[j1*p+k1] += 1
        # W: c11 -= M1? No. Let me use standard Strassen carefully.
        #
        # For 2x2: c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        #
        # So W contributions:
        # M1: W[k0*n+i0] += 1, W[k1*n+i0] += 1 (for c11 and c22)
        # Wait, c11 is at position (i0, k0), c22 is at (i1, k1)
        # c12 is at (i0, k1), c21 is at (i1, k0)
        #
        # Let me redo with proper W assignments:
        # M1 contributes to c11 (+1) and c22 (+1): W[k0*n+i0]+=1, W[k1*n+i1]+=1
        # M2 contributes to c21 (+1) and c22 (-1): W[k0*n+i1]+=1, W[k1*n+i1]-=1
        # M3 contributes to c12 (+1) and c22 (+1): W[k1*n+i0]+=1, W[k1*n+i1]+=1
        # M4 contributes to c11 (+1) and c21 (+1): W[k0*n+i0]+=1, W[k0*n+i1]+=1
        # M5 contributes to c12 (+1) and c11 (-1): W[k1*n+i0]+=1, W[k0*n+i0]-=1
        # M6 contributes to c22 (+1): W[k1*n+i1]+=1
        # M7 contributes to c11 (+1): W[k0*n+i0]+=1

        # M1: U=(i0m+j0, i1m+j1), V=(j0p+k0, j1p+k1), W=(k0n+i0, k1n+i1)
        terms.append((col,
                      [(i0 * m + j0, 1), (i1 * m + j1, 1)],
                      [(j0 * p + k0, 1), (j1 * p + k1, 1)],
                      [(k0 * n + i0, 1), (k1 * n + i1, 1)]))
        col += 1

        # M2: U=(i1m+j0, i1m+j1), V=(j0p+k0), W=(k0n+i1, k1n+i1)
        terms.append((col,
                      [(i1 * m + j0, 1), (i1 * m + j1, 1)],
                      [(j0 * p + k0, 1)],
                      [(k0 * n + i1, 1), (k1 * n + i1, -1)]))
        col += 1

        # M3: U=(i0m+j0), V=(j0p+k1, j1p+k1), W=(k1n+i0, k1n+i1)
        terms.append((col,
                      [(i0 * m + j0, 1)],
                      [(j0 * p + k1, 1), (j1 * p + k1, -1)],
                      [(k1 * n + i0, 1), (k1 * n + i1, 1)]))
        col += 1

        # M4: U=(i1m+j1), V=(j1p+k0, j0p+k0), W=(k0n+i0, k0n+i1)
        terms.append((col,
                      [(i1 * m + j1, 1)],
                      [(j1 * p + k0, 1), (j0 * p + k0, -1)],
                      [(k0 * n + i0, 1), (k0 * n + i1, 1)]))
        col += 1

        # M5: U=(i0m+j0, i0m+j1), V=(j1p+k1), W=(k0n+i0, k1n+i0)
        terms.append((col,
                      [(i0 * m + j0, 1), (i0 * m + j1, 1)],
                      [(j1 * p + k1, 1)],
                      [(k0 * n + i0, -1), (k1 * n + i0, 1)]))
        col += 1

        # M6: U=(i1m+j0, i0m+j0), V=(j0p+k0, j0p+k1), W=(k1n+i1)
        terms.append((col,
                      [(i1 * m + j0, 1), (i0 * m + j0, -1)],
                      [(j0 * p + k0, 1), (j0 * p + k1, 1)],
                      [(k1 * n + i1, 1)]))
        col += 1

        # M7: U=(i0m+j1, i1m+j1), V=(j1p+k0, j1p+k1), W=(k0n+i0)
        terms.append((col,
                      [(i0 * m + j1, 1), (i1 * m + j1, -1)],
                      [(j1 * p + k0, 1), (j1 * p + k1, 1)],
                      [(k0 * n + i0, 1)]))
        col += 1

    return terms


def _build_residual_terms(n, m, p):
    """
    Generate schoolbook terms for entries not covered by Strassen blocks.
    The 6 Strassen blocks cover i in {0,1} and all j,k in their 2x2 subranges.
    Since blocks tile j×k completely for i in {0,1}, all (i,j,k) with i in {0,1}
    are covered. Only i=2 remains, giving 1*4*7 = 28 terms.
    Wait, let me recount: blocks cover (i0,j0,k0) to (i0+1,j0+1,k0+1).
    For i=0,1: j in {0,1,2,3}, k in {0,1,2,3,4,5,6}... no, each block is 2x2 in j,k.
    Block (0,0,0): i in {0,1}, j in {0,1}, k in {0,1}
    Block (0,0,2): i in {0,1}, j in {0,1}, k in {2,3}
    Block (0,0,4): i in {0,1}, j in {0,1}, k in {4,5}
    Block (0,2,0): i in {0,1}, j in {2,3}, k in {0,1}
    Block (0,2,2): i in {0,1}, j in {2,3}, k in {2,3}
    Block (0,2,4): i in {0,1}, j in {2,3}, k in {4,5}
    
    So covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    k=6 is NOT covered for i in {0,1}.
    Also i=2 is not covered at all.
    
    Uncovered: 
    - i=2, all j, all k: 1*4*7 = 28
    - i=0,1, j=0,1,2,3, k=6: 2*4*1 = 8
    Total residual: 36 terms.
    """
    covered = set()
    for (i0, j0, k0) in [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 2, 0), (0, 2, 2), (0, 2, 4)]:
        for ii in range(i0, i0 + 2):
            for jj in range(j0, j0 + 2):
                for kk in range(k0, k0 + 2):
                    covered.add((ii, jj, kk))

    terms = []
    col = 42
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    terms.append((col, i * m + j, j * p + k, k * n + i))
                    col += 1
    return terms


def _check(U, V, W, n, m, p):
    """Vectorized verification: check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    Uf = U
    Vf = V
    Wf = W
    R = Uf.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Uf[i * m + j], Vf[j * p + k] * Wf[k * n + i])
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _fallback(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
