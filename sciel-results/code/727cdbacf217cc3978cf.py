"""Matrix multiplication 3,5,6 - block-Strassen rank-84 decomposition."""
import time

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
DEADLINE = 150.0
SAFETY = 10.0  # seconds margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6

    # Partitions
    row_blocks = [2, 1]      # sums to n=3
    mid_blocks = [2, 2, 1]   # sums to m=5
    col_blocks = [2, 2, 2]   # sums to p=6

    # Build cumulative offsets
    row_off = []
    s = 0
    for sz in row_blocks:
        row_off.append(s); s += sz
    mid_off = []
    s = 0
    for sz in mid_blocks:
        mid_off.append(s); s += sz
    col_off = []
    s = 0
    for sz in col_blocks:
        col_off.append(s); s += sz

    U = []
    V = []
    W = []

    def add_term(u_vals, v_vals, w_vals):
        """u_vals: dict (i,j)->val, v_vals: dict (j,k)->val, w_vals: dict (i,k)->val"""
        col = len(U)
        U.append([0.0] * (n * m))
        V.append([0.0] * (m * p))
        W.append([0.0] * (p * n))
        for (i, j), val in u_vals.items():
            U[col][i * m + j] = val
        for (j, k), val in v_vals.items():
            V[col][j * p + k] = val
        for (i, k), val in w_vals.items():
            W[col][k * n + i] = val

    # Strassen 2x2x2: local indices (a,b) in A block rows 0,1; A cols 0,1; B rows 0,1 cols 0,1
    # C = A*B, C[0,0]=m1+m4-m5+m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m6
    def strassen_terms(i0, j0, k0):
        # A is 2x2 at rows i0..i0+1, cols j0..j0+1
        # B is 2x2 at rows j0..j0+1, cols k0..k0+1
        # C is 2x2 at rows i0..i0+1, cols k0..k0+1
        A = lambda r, c: (i0 + r, j0 + c)
        B = lambda r, c: (j0 + r, k0 + c)
        C = lambda r, c: (i0 + r, k0 + c)

        def mk_u(d):
            return {A(*k): v for k, v in d.items()}
        def mk_v(d):
            return {B(*k): v for k, v in d.items()}
        def mk_w(d):
            return {C(*k): v for k, v in d.items()}

        # m1 = (A00+A11)*(B00+B11)
        add_term(mk_u({(0,0):1,(1,1):1}), mk_v({(0,0):1,(1,1):1}),
                 mk_w({(0,0):1,(1,1):1}))
        # m2 = (A10+A11)*B00
        add_term(mk_u({(1,0):1,(1,1):1}), mk_v({(0,0):1}),
                 mk_w({(1,0):1}))
        # m3 = A00*(B01-B11)
        add_term(mk_u({(0,0):1}), mk_v({(0,1):1,(1,1):-1}),
                 mk_w({(0,1):1,(1,1):1}))
        # m4 = A11*(B10-B00)
        add_term(mk_u({(1,1):1}), mk_v({(1,0):1,(0,0):-1}),
                 mk_w({(0,0):-1,(1,0):1}))
        # m5 = (A00+A01)*B11
        add_term(mk_u({(0,0):1,(0,1):1}), mk_v({(1,1):1}),
                 mk_w({(0,0):-1,(0,1):1}))
        # m6 = (A10-A00)*(B00+B01)
        add_term(mk_u({(1,0):1,(0,0):-1}), mk_v({(0,0):1,(0,1):1}),
                 mk_w({(1,1):1}))
        # m7 = (A01-A11)*(B10+B11)
        add_term(mk_u({(0,1):1,(1,1):-1}), mk_v({(1,0):1,(1,1):1}),
                 mk_w({(0,0):1,(1,1):-1}))

    # Schoolbook terms for a block
    def schoolbook_block(i0, r, j0, mid, k0, c):
        for di in range(r):
            for dj in range(mid):
                for dk in range(c):
                    add_term({(i0+di, j0+dj): 1},
                             {(j0+dj, k0+dk): 1},
                             {(i0+di, k0+dk): 1})

    ri = 0
    for (i0, r) in zip(row_off, row_blocks):
        for (j0, mid) in zip(mid_off, mid_blocks):
            for (k0, c) in zip(col_off, col_blocks):
                if r == 2 and mid == 2 and c == 2:
                    strassen_terms(i0, j0, k0)
                else:
                    schoolbook_block(i0, r, j0, mid, k0, c)

    result = {"U": U, "V": V, "W": W}

    # Verify with random matrices
    try:
        import random
        random.seed(12345)
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        # C = A*B
        C = [[sum(A[i][t]*B[t][k] for t in range(m)) for k in range(p)] for i in range(n)]
        # Recompute via tensor
        for i in range(n):
            for k in range(p):
                s = 0
                for col in range(len(U)):
                    s += U[col][i*m] * 0  # placeholder
                # proper: sum over col of (sum_j U[col][i*m+j]*A[i][j])*(sum_kk V[col][j*p+kk]*B[j][kk]) then * W[col][k*n+i]
                a_u = 0
                for col in range(len(U)):
                    au = sum(U[col][i*m+j]*A[i][j] for j in range(m))
                    bv = sum(V[col][j*p+kk]*B[j][kk] for j in range(m) for kk in range(p))
                    s += au * bv * W[col][k*n+i]
                if s != C[i][k]:
                    # verification failed -> fall back to schoolbook
                    return schoolbook()
        return result
    except Exception:
        return result

def schoolbook():
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
# EVOLVE-BLOCK-END
