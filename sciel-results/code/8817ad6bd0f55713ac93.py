"""Matrix multiplication 2x4 x 4x7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_222():
    """Strassen decomposition of (2,2,2). Returns U,V,W each 4x7 with 0/±1 entries.
    U[i*2+j, c] = u_c[i,j], V[j*2+k, c] = v_c[j,k], W[k*2+i, c] = w_c[k,i]."""
    # Standard Strassen: C = A*B, A,B 2x2.
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A10)B11
    # m6 = (A01-A10)(B10+B11)
    # m7 = (A01-A11)(B00+B10)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    # Index: A[i,j] -> row i*2+j ; B[j,k] -> row j*2+k ; C[k,i] -> row k*2+i (W convention: W[k*n+i])
    # We'll build per-term lists of (i,j) for A, (j,k) for B, (k,i) for C with sign.
    terms = []
    # m1
    terms.append(([(0,0),(1,1)], [(0,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)],
                  [1,1], [1,1], [1,-1,1,1]))
    # m2
    terms.append(([(1,0),(1,1)], [(0,0)], [(1,0),(1,1)],
                  [1,1], [1], [0,0,1,1]))
    # m3
    terms.append(([(0,0)], [(0,1),(1,1)], [(0,1),(1,1)],
                  [1], [1,-1], [0,1,0,1]))
    # m4
    terms.append(([(1,1)], [(1,0),(0,0)], [(0,0),(1,0)],
                  [1], [1,-1], [1,0,1,0]))
    # m5
    terms.append(([(0,0),(0,1)], [(1,1)], [(0,1),(1,1)],
                  [1,1], [1], [0,0,-1,1]))
    # m6
    terms.append(([(0,1),(1,0)], [(1,0),(1,1)], [(1,1)],
                  [1,-1], [1,1], [0,0,0,1]))
    # m7
    terms.append(([(0,1),(1,1)], [(0,0),(1,0)], [(0,0),(1,0)],
                  [1,-1], [1,1], [1,0,0,-1]))
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for c, (a_idx, b_idx, ci_idx, a_s, b_s, ci_s) in enumerate(terms):
        for (i, j), s in zip(a_idx, a_s):
            U[i*2 + j][c] += s
        for (j, k), s in zip(b_idx, b_s):
            V[j*2 + k][c] += s
        for (k, i), s in zip(ci_idx, ci_s):
            W[k*2 + i][c] += s
    return U, V, W

def _mul_227():
    """Decomposition of (2,2,7): A(2x2)*B(2x7) -> C(2x7).
    Split B into 3 columns-blocks: B = [B_a(2x2) B_b(2x2) B_c(2x2) B_d(2x1)].
    C = A*B_a + A*B_b + A*B_c + A*B_d.
    Use Strassen (rank7) for each 2x2*2x2, schoolbook (rank2) for 2x2*2x1.
    Total rank = 3*7 + 2 = 23.
    U: (2*2=4) x 23 ; V: (2*7=14) x 23 ; W: (7*2=14) x 23.
    Convention: U[i*2+j], V[j*7+k], W[k*2+i].
    """
    U222, V222, W222 = _strassen_222()
    # Build the 2x2x7 factor.
    rank = 23
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(14)]
    W = [[0]*rank for _ in range(14)]
    col = 0
    # Three 2x2x2 blocks on B columns 0-1, 2-3, 4-5
    for block in range(3):
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2 + j][col + c] += U222[i*2 + j][c]
            for j in range(2):
                for k in range(2):
                    # B column offset = block*2 + k
                    V[j*7 + (block*2 + k)][col + c] += V222[j*2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[(block*2 + k)*2 + i][col + c] += W222[k*2 + i][c]
        col += 7
    # Last 2x2x1 block on B column 6 (schoolbook rank 2)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                U[i*2 + j][col + (k*2 + i)] = 1
                V[j*7 + 6][col + (k*2 + i)] = 1
                W[6*2 + i][col + (k*2 + i)] = 1
    return U, V, W

def _build_46():
    """T(2,4,7) <= 2 * T(2,2,7).
    Middle index j = 2*j2 + j1, j2,j1 in {0,1}.
    For each j2: A^{j2}[i, j1] = A[i, 2*j2 + j1] (2x2), B^{j2}[j1, k] = B[2*j2 + j1, k] (2x7).
    C[i,k] = sum_{j2} (A^{j2} B^{j2})[i,k].
    U: (2*4=8) x 46 ; V: (4*7=28) x 46 ; W: (7*2=14) x 46.
    Convention: U[i*4+j], V[j*7+k], W[k*2+i].
    """
    U227, V227, W227 = _mul_227()
    r227 = 23
    rank = 2 * r227
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(14)]
    for j2 in range(2):
        base = j2 * r227
        for c in range(r227):
            # A: i in 0..1, j1 in 0..1 -> A[i, 2*j2 + j1]
            for i in range(2):
                for j1 in range(2):
                    U[i*4 + (2*j2 + j1)][base + c] += U227[i*2 + j1][c]
            # B: j1 in 0..1, k in 0..6 -> B[2*j2 + j1, k]
            for j1 in range(2):
                for k in range(7):
                    V[(2*j2 + j1)*7 + k][base + c] += V227[j1*7 + k][c]
            # C: k in 0..6, i in 0..1 -> C[k, i]
            for k in range(7):
                for i in range(2):
                    W[k*2 + i][base + c] += W227[k*2 + i][c]
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=7):
    """Check sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == delta_{i?} for the mult tensor.
    The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    We need sum_c U[a,c] V[b,c] W[c2,c] = 1 when a=i*m+j, b=j*p+k, c2=k*n+i (same j), else 0.
    """
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c2 = k*n + i
                s = 0.0
                for c in range(r):
                    s += Ua[a, c] * Va[b, c] * Wa[c2, c]
                if abs(s - 1.0) > 1e-9:
                    ok = False
    # check some off-diagonal zeros
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    if i2 == i: continue
                    a = i*m + j
                    b = j*p + k
                    c2 = k*n + i2
                    s = 0.0
                    for c in range(r):
                        s += Ua[a, c] * Va[b, c] * Wa[c2, c]
                    if abs(s) > 1e-9:
                        ok = False
    return ok

def construct():
    t0 = time.time()
    deadline = 150 - 5  # safety margin
    U, V, W = _build_46()
    # verify quickly
    if time.time() < deadline:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    # fallback: schoolbook (always valid)
    n, m, p = DIMENSIONS
    rank = n * m * p
    Us = [[0] * rank for _ in range(n * m)]
    Vs = [[0] * rank for _ in range(m * p)]
    Ws = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i * m + j][column] = 1
                Vs[j * p + k][column] = 1
                Ws[k * n + i][column] = 1
                column += 1
    return {"U": Us, "V": Vs, "W": Ws}
# EVOLVE-BLOCK-END
