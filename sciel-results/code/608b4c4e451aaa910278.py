"""Matrix multiplication 2x4 x 4x4 rank-minimal decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 4]
START = time.time()
DEADLINE = START + 140.0  # safety margin before 150s kill

def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def _check(U, V, W):
    """Verify U*V*W == schoolbook tensor for 2x4 x 4x4. Returns True if exact."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    C = [[0] * p for _ in range(n)]
    for c in range(rank):
        uc = [U[i][c] for i in range(nm)]
        vc = [V[j][c] for j in range(mp)]
        wc = [W[k][c] for k in range(pn)]
        for i in range(nm):
            a = uc[i]
            if a == 0:
                continue
            for j in range(mp):
                b = vc[j]
                if b == 0:
                    continue
                t = a * b
                for k in range(pn):
                    w = wc[k]
                    if w == 0:
                        continue
                    # W[k*n+i, r] -> k = k*n+i
                    kk, ii = divmod(k, n)
                    jj = j // p
                    kk2 = k  # W index
                    # A row = i*m+jj -> i, jj
                    C[ii][kk] += t * w
    # schoolbook
    for i in range(n):
        for j in range(m):
            for jj in range(m):
                a = (i * m + jj)
            for jj2 in range(m):
                pass
    # explicit schoolbook
    C2 = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0
            for jj in range(m):
                s += (1 if True else 0) * 0
    # simpler: C[i][j] = sum_l A[i][l]*B[l][j]
    for i in range(n):
        for j in range(p):
            s = 0
            for l in range(m):
                s += 0  # A,B are symbolic 1; product = 1 if index valid
    # Actually the tensor T[i*m+l, l*p+j, j*n+i] = 1. Sum over r U[i*m+l,r]V[l*p+j,r]W[j*n+i,r]
    # Let's just build T and compare.
    T = {}
    for i in range(n):
        for l in range(m):
            for j in range(p):
                T[(i * m + l, l * p + j, j * n + i)] = 1
    for (a, b, c), val in T.items():
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if s != val:
            return False
    return True

def _rank16():
    # Block decomposition: C = A1*B1 + A2*B2, with F(2,2,2) per block.
    # A: 2x4 rows: A1 = A[0:2,0:2] (rows i=0,1; cols 0,1), A2 = A[0:2,2:4]
    # B: 4x4 -> B1 = B[0:2,:] (rows 0,1), B2 = B[2:4,:]
    # B1 2x4 -> [B1a(2x2) B1b(2x2)], B2 -> [B2a B2b]
    # F(2,2,2): x = A1+I, y=B1a-B1b, z=B1a+B1b, w=A1-I
    #   M1=A1*B1a, M2=x*y, M3=x*z, M4=w*z
    #   P = M1+M4, Q=M2+M3 -> P+Q = A1*B1a + A1*B1b
    # Same for block 2.
    n, m, p = 2, 4, 4
    U = [[0] * 16 for _ in range(n * m)]   # 8 x 16
    V = [[0] * 16 for _ in range(m * p)]   # 16 x 16
    W = [[0] * 16 for _ in range(p * n)]   # 8 x 16
    # A indices: row index = i*m + j, i in [0,2), j in [0,4)
    # A1: i in {0,1}, j in {0,1}
    # A2: i in {0,1}, j in {2,3}
    # B indices: index = l*p + k, l in [0,4), k in [0,4)
    #   B1 rows l in {0,1}; B2 rows l in {2,3}
    #   B1a cols k in {0,1}; B1b cols k in {2,3}; same for B2
    # C indices: index = k*n + i, k in [0,4), i in [0,2)
    def a_idx(i, j): return i * m + j
    def b_idx(l, k): return l * p + k
    def c_idx(k, i): return k * n + i

    # Block 1 (A1 x B1)
    # M1 = A1 * B1a
    for i in range(2):
        for j in range(2):
            U[a_idx(i, j)][0] += 1
    for l in range(2):
        for k in range(2):
            V[b_idx(l, k)][0] += 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][0] += 1
    # M2 = x * y, x=A1+I, y=B1a-B1b
    for i in range(2):
        for j in range(2):
            U[a_idx(i, j)][1] += 1  # A1
        U[a_idx(i, i)][1] += 1      # I
    for l in range(2):
        for k in range(2):
            V[b_idx(l, k)][1] += 1      # B1a
            V[b_idx(l, k + 2)][1] -= 1  # -B1b
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][1] += 1
    # M3 = x * z, z=B1a+B1b
    for i in range(2):
        for j in range(2):
            U[a_idx(i, j)][2] += 1
        U[a_idx(i, i)][2] += 1
    for l in range(2):
        for k in range(2):
            V[b_idx(l, k)][2] += 1
            V[b_idx(l, k + 2)][2] += 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][2] += 1
    # M4 = w * z, w=A1-I
    for i in range(2):
        for j in range(2):
            U[a_idx(i, j)][3] += 1
        U[a_idx(i, i)][3] -= 1
    for l in range(2):
        for k in range(2):
            V[b_idx(l, k)][3] += 1
            V[b_idx(l, k + 2)][3] += 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][3] += 1
    # Block 2 (A2 x B2)
    for i in range(2):
        for j in range(2, 4):
            U[a_idx(i, j)][4] += 1
    for l in range(2, 4):
        for k in range(2):
            V[b_idx(l, k)][4] += 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][4] += 1
    for i in range(2):
        for j in range(2, 4):
            U[a_idx(i, j)][5] += 1
        U[a_idx(i, i)][5] += 1
    for l in range(2, 4):
        for k in range(2):
            V[b_idx(l, k)][5] += 1
            V[b_idx(l, k + 2)][5] -= 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][5] += 1
    for i in range(2):
        for j in range(2, 4):
            U[a_idx(i, j)][6] += 1
        U[a_idx(i, i)][6] += 1
    for l in range(2, 4):
        for k in range(2):
            V[b_idx(l, k)][6] += 1
            V[b_idx(l, k + 2)][6] += 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][6] += 1
    for i in range(2):
        for j in range(2, 4):
            U[a_idx(i, j)][7] += 1
        U[a_idx(i, i)][7] -= 1
    for l in range(2, 4):
        for k in range(2):
            V[b_idx(l, k)][7] += 1
            V[b_idx(l, k + 2)][7] += 1
    for k in range(4):
        for i in range(2):
            W[c_idx(k, i)][7] += 1
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    global DEADLINE
    start = time.time()
    # Try rank-16 first; fall back to schoolbook if invalid or time runs low.
    best = None
    try:
        cand = _rank16()
        if _check(cand["U"], cand["V"], cand["W"]):
            best = cand
    except Exception:
        best = None
    if best is None:
        best = _schoolbook()
    if time.time() > DEADLINE - 5:
        return best
    return best
# EVOLVE-BLOCK-END
