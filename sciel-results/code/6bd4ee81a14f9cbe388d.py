"""Deterministic rank-97 decomposition for M_{4,4,7} (Strassen 4x4x4 + schoolbook 4x4x3)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def elapsed_ok():
        return (time.time() - t0) < DEADLINE

    n, m, p = DIMENSIONS  # n=4, m=4, p=7

    # ---- Build the 49-term Strassen decomposition for 4x4 x 4x4 ----
    # Indices: A row-major a[i*4+j], B b[i*4+j], C c[i*4+j]
    # Each term is (u_list, v_list, w_list) of (index, coef)
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    # Strassen 2x2: A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # p1=(a+d)(e+h) -> c00 +=, c11 +=
    # p2=(c+d)e     -> c10 +=, c11 +=
    # p3=a(f-h)     -> c01 +=
    # p4=d(g-e)     -> c00 +=, c10 +=
    # p5=(a+b)h     -> c00 -=, c01 +=
    # p6=(c-a)(e+f) -> c11 +=
    # p7=(b-d)(g+h) -> c00 +=
    # Relative indices within a 2x2: a=0,b=1,c=2,d=3 ; e=0,f=1,g=2,h=3
    def s2(u0, v0, w0, w1=None):
        # helper for a single product pattern
        pass

    # Define the 7 Strassen 2x2 patterns as (u_offsets, v_offsets, w_offsets)
    # offsets are relative to the 2x2 block origin (row*2+col)
    P = [
        # p1
        ([(0,1),(2,1)], [(0,1),(2,1)], [(0,1),(2,1)]),
        # p2
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)]),
        # p3
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        # p4
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # p5
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # p6
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # p7
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

    def idx2d(r, c, size):
        return r * size + c

    # Recursive Strassen for 4x4: split into 2x2 blocks.
    # Block origins (row,col) in {0,1} x {0,1}, each 2x2.
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B blocks: B00,B01,B10,B11 similarly
    # C blocks: C00,C01,C10,C11
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    def expand(u_offs, v_offs, w_offs, u_origin, v_origin, w_origin):
        u = [(u_origin[0]*2 + u_origin[1]*2 + (r*2+c), coef) for (r,c),coef in u_offs]
        v = [(v_origin[0]*2 + v_origin[1]*2 + (r*2+c), coef) for (r,c),coef in v_offs]
        w = [(w_origin[0]*2 + w_origin[1]*2 + (r*2+c), coef) for (r,c),coef in w_offs]
        return u, v, w

    # block origins
    A00, A01, A10, A11 = (0,0),(0,1),(1,0),(1,1)
    B00, B01, B10, B11 = (0,0),(0,1),(1,0),(1,1)
    C00, C01, C10, C11 = (0,0),(0,1),(1,0),(1,1)

    def add_2x2(u_o, v_o, w_o):
        for (uo, vo, wo) in P:
            u, v, w = expand(uo, vo, wo, u_o, v_o, w_o)
            add_term(u, v, w)

    # C00
    add_2x2(A00, B00, C00)
    add_2x2(A01, B10, C00)
    # C01
    add_2x2(A00, B01, C01)
    add_2x2(A01, B11, C01)
    # C10
    add_2x2(A10, B00, C10)
    add_2x2(A11, B10, C10)
    # C11
    add_2x2(A10, B01, C11)
    add_2x2(A11, B11, C11)

    rank_s = len(terms)
    assert rank_s == 49, rank_s

    # ---- Build the 48-term schoolbook for 4x4 x 4x3 (B columns 0,1,2) ----
    # C[i][k] = sum_l A[i][l] B[l][k], for k in {0,1,2}
    # W index = k*n + i
    school = []
    for i in range(n):
        for l in range(m):
            for k in range(3):
                school.append(([(i*m + l, 1)], [(l*3 + k, 1)], [((k*n + i), 1)]))
    rank_sc = len(school)
    assert rank_sc == 48, rank_sc

    total_rank = rank_s + rank_sc
    assert total_rank == 97, total_rank

    # Assemble U, V, W as numpy then to lists
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * 7, total_rank), dtype=float)
    W = np.zeros((7 * n, total_rank), dtype=float)

    col = 0
    for (u, v, w) in terms:
        for (idx, coef) in u:
            U[idx, col] += coef
        for (idx, coef) in v:
            V[idx, col] += coef
        for (idx, coef) in w:
            W[idx, col] += coef
        col += 1

    for (u, v, w) in school:
        for (idx, coef) in u:
            U[idx, col] += coef
        for (idx, coef) in v:
            V[idx, col] += coef
        for (idx, coef) in w:
            W[idx, col] += coef
        col += 1

    # ---- Verification: check tensor identity on random inputs ----
    def verify():
        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, 7))
            Cref = A @ B
            # U: (n*m, R) -> u = U @ A_flat
            Aflat = A.reshape(-1)
            Bflat = B.reshape(-1)
            u = U @ Aflat
            v = V @ Bflat
            C = W.T @ (u * v)  # (p*n, R) @ (R,) -> (p*n,)
            C = C.reshape((7, n)).T  # W[k*n+i] -> C[i][k]
            if not np.allclose(C, Cref, atol=1e-9, rtol=1e-9):
                return False
        return True

    if not verify():
        # Fallback: schoolbook rank 112
        rank = n * m * p
        Uf = np.zeros((n * m, rank), dtype=float)
        Vf = np.zeros((m * p, rank), dtype=float)
        Wf = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, col] = 1
                    Vf[j * p + k, col] = 1
                    Wf[k * n + i, col] = 1
                    col += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
