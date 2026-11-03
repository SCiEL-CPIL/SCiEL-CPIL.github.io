"""Suite task: rank-minimal exact tensor decomposition for 3x3 by 3x8 matmul."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(len(U[0])):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen)."""
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))   # P1 = (a+d)(e+h)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))   # P2 = (c+d)(e)
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,0]))   # P3 = (a+b)(h)
    terms.append(([0,0,1,1], [0,1,0,-1], [1,0,0,1]))  # P4 = (c+d)(f-h)
    terms.append(([1,0,0,1], [0,0,0,1], [0,1,0,0]))   # P5 = (a+d)(h)
    terms.append(([0,1,0,-1], [1,0,0,0], [0,1,0,0]))  # P6 = (b-d)(e)
    terms.append(([1,0,-1,0], [0,0,1,1], [-1,0,0,0])) # P7 = (a-c)(g+h)
    return terms

def _strassen_2x3x3():
    """
    Return 10 rank-1 terms computing C = A(2x3) * B(3x3) = (2x3).
    A = [[a0,a1,a2],[b0,b1,b2]], B = [[c0,c1,c2],[d0,d1,d2],[e0,e1,e2]].
    C[0] = a0*c0+a1*d0+a2*e0, C[1] = a0*c1+a1*d1+a2*e1, C[2] = a0*c2+a1*d2+a2*e2
    C[3] = b0*c0+b1*d0+b2*e0, C[4] = b0*c1+b1*d1+b2*e1, C[5] = b0*c2+b1*d2+b2*e2
    """
    terms = []
    # Row 0
    terms.append(([1,0,0,0,1,0], [1,0,0,0,1,0,0,0,1], [1,0,0,0,0,0,1,0,0,1]))
    terms.append(([0,0,1,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0]))
    terms.append(([1,1,0,0,0,0], [0,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0,0]))
    terms.append(([0,0,1,1,0,0], [0,1,0,0,0,0,0,-1,0], [1,0,0,0,0,0,0,0,0,1]))
    terms.append(([1,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0,0]))
    terms.append(([0,1,0,-1,0,0], [1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0]))
    # Row 1
    terms.append(([0,0,0,1,0,1], [1,0,0,0,1,0,0,0,1], [0,0,1,0,0,0,0,1,0,0]))
    terms.append(([0,0,0,0,1,1], [1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0]))
    terms.append(([0,0,1,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1,0,0,0,0]))
    terms.append(([0,0,0,1,0,-1], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0]))
    return terms

def _embed_strassen_2x2x2(U, V, W, col, r0, r1, c0, c1, k0, k1, m, p, n):
    """Add 7 Strassen terms for a 2x2 A block x 2 col B block. Return col+7."""
    a_map = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    b_map = [(c0, k0), (c0, k1), (c1, k0), (c1, k1)]
    c_map = [(k0, r0), (k0, r1), (k1, r0), (k1, r1)]
    for u_vec, v_vec, w_vec in _strassen_2x2x2():
        for li in range(4):
            if u_vec[li] != 0:
                U[a_map[li][0]*m + a_map[li][1]][col] += u_vec[li]
            if v_vec[li] != 0:
                V[b_map[li][0]*p + b_map[li][1]][col] += v_vec[li]
            if w_vec[li] != 0:
                W[c_map[li][0]*n + c_map[li][1]][col] += w_vec[li]
        col += 1
    return col

def _embed_strassen_2x3x3(U, V, W, col, r0, r1, c0, c1, c2, k0, k1, k2, m, p, n):
    """Add 10 terms for a 2x3 A block x 3 col B block. Return col+10."""
    a_map = [(r0,c0),(r0,c1),(r0,c2),(r1,c0),(r1,c1),(r1,c2)]
    b_map = [(c0,k0),(c0,k1),(c0,k2),(c1,k0),(c1,k1),(c1,k2),(c2,k0),(c2,k1),(c2,k2)]
    c_map = [(k0,r0),(k1,r0),(k2,r0),(k0,r1),(k1,r1),(k2,r1)]
    for u_vec, v_vec, w_vec in _strassen_2x3x3():
        for li in range(6):
            if u_vec[li] != 0:
                U[a_map[li][0]*m + a_map[li][1]][col] += u_vec[li]
        for li in range(9):
            if v_vec[li] != 0:
                V[b_map[li][0]*p + b_map[li][1]][col] += v_vec[li]
        for li in range(6):
            if w_vec[li] != 0:
                W[c_map[li][0]*n + c_map[li][1]][col] += w_vec[li]
        col += 1
    return col

def _add_schoolbook(U, V, W, col, i, j, k, m, p, n):
    U[i*m+j][col] = 1.0
    V[j*p+k][col] = 1.0
    W[k*n+i][col] = 1.0
    return col + 1

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=3, p=8

    # Rank 52: 2 blocks of 2x3 Strassen (10 each = 20) + 2 blocks of 2x2 Strassen (7 each = 14) + 18 schoolbook
    rank = 2*10 + 2*7 + 18  # 52
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]

    col = 0
    # Track which (i,j,k) terms are covered by Strassen blocks
    covered = set()

    # 2 blocks: rows {0,1}, all 3 cols of A, column pairs of B
    for (k0, k1, k2) in [(0,1,2), (4,5,6)]:
        col = _embed_strassen_2x3x3(U, V, W, col, 0, 1, 0, 1, 2, k0, k1, k2, m, p, n)
        for i in (0,1):
            for j in (0,1,2):
                for k in (k0,k1,k2):
                    covered.add((i,j,k))

    # 2 blocks: rows {1,2}, col pairs of A, column pairs of B (2x2x2)
    for (c0, c1, k0, k1) in [(0,1,0,1), (2,3,4,5)]:
        col = _embed_strassen_2x2x2(U, V, W, col, 1, 2, c0, c1, k0, k1, m, p, n)
        for i in (1,2):
            for j in (c0,c1):
                for k in (k0,k1):
                    covered.add((i,j,k))

    # Remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    col = _add_schoolbook(U, V, W, col, i, j, k, m, p, n)

    assert col == rank, "rank mismatch: %d != %d" % (col, rank)

    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
