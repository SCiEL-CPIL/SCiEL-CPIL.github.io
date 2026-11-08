"""Rank-98 decomposition of 4x4 x 4x8 via two independent Strassen 4x4x4 blocks."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U,V,W) for 2x2 x 2x2 -> 2x2 at rank 7.
    U indexed [i*2+j], V [j*2+k], W [k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # A[i,j] index: i*2+j ; B[j,k] index: j*2+k ; M[i,k] index: k*2+i
    prods = [
        ([(0,1),(3,1)], [(0,1),(3,1)]),   # P1
        ([(2,1),(3,1)], [(0,1)]),          # P2
        ([(0,1)], [(2,1),(3,-1)]),         # P3
        ([(3,1)], [(1,1),(2,-1)]),         # P4
        ([(0,1),(1,1)], [(3,1)]),          # P5
        ([(2,1),(0,-1)], [(0,1),(2,1)]),   # P6
        ([(1,1),(3,-1)], [(1,1),(3,1)]),   # P7
    ]
    for c,(ac,bc) in enumerate(prods):
        for (idx,s) in ac:
            U[idx][c] += s
        for (idx,s) in bc:
            V[idx][c] += s
    combine = {
        0: [(0,1),(3,1),(4,-1),(6,1)],  # M00
        2: [(2,1),(4,1)],               # M01
        1: [(1,1),(3,1)],               # M10
        3: [(0,1),(1,-1),(2,1),(5,1)],  # M11
    }
    for widx, terms in combine.items():
        for (c,s) in terms:
            W[widx][c] += s
    return U, V, W


def _embed_2x2_into_4x4_4x4():
    """Build rank-49 U,V,W for 4x4 x 4x4 -> 4x4 using block Strassen."""
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    u_pat, v_pat, w_pat = _strassen_2x2()
    Ab = {'A00':(0,0),'A01':(0,1),'A10':(1,0),'A11':(1,1)}
    Bb = {'B00':(0,0),'B01':(0,1),'B10':(1,0),'B11':(1,1)}
    prod_terms = {
        0: [('A00','B00',1),('A00','B11',1),('A11','B00',1),('A11','B11',1)],
        1: [('A10','B00',1),('A11','B00',1)],
        2: [('A00','B10',1),('A00','B11',-1)],
        3: [('A11','B01',1),('A11','B10',-1)],
        4: [('A00','B11',1),('A01','B11',1)],
        5: [('A10','B00',1),('A10','B10',1),('A00','B00',-1),('A00','B10',-1)],
        6: [('A01','B01',1),('A01','B11',1),('A11','B01',-1),('A11','B11',-1)],
    }
    q_to_mblocks = {
        0: [(0,0,1),(1,1,1)],
        1: [(1,0,1),(1,1,-1)],
        2: [(0,1,1),(1,1,1)],
        3: [(0,0,1),(1,0,1)],
        4: [(0,0,-1),(0,1,1)],
        5: [(1,1,1)],
        6: [(0,0,1)],
    }
    col = 0
    for q in range(7):
        for (an, bn, s) in prod_terms[q]:
            bi, bj = Ab[an]
            bj2, bk2 = Bb[bn]
            for c in range(7):
                for ii in range(2):
                    for jj in range(2):
                        val = u_pat[ii*2+jj][c]
                        if val != 0:
                            U[(2*bi+ii)*4 + (2*bj+jj)][col] += s * val
                for jj in range(2):
                    for kk in range(2):
                        val = v_pat[jj*2+kk][c]
                        if val != 0:
                            V[(2*bj2+jj)*4 + (2*bk2+kk)][col] += val
                for (mi, mk, coeff) in q_to_mblocks[q]:
                    for ii in range(2):
                        for kk in range(2):
                            val = w_pat[kk*2+ii][c]
                            if val != 0:
                                W[(2*mk+kk)*4 + (2*mi+ii)][col] += s * coeff * val
            col += 1
    assert col == 49, f"expected 49 cols, got {col}"
    return U, V, W


def _build_4x4_4x8():
    """Build rank-98 U,V,W for 4x4 x 4x8 -> 4x8."""
    U16_49, V16_49, W16_49 = _embed_2x2_into_4x4_4x4()
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    for r in range(16):
        for c in range(49):
            U[r][c] = U16_49[r][c]
            U[r][c+49] = U16_49[r][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*8+k][c] = V16_49[j*4+k][c]
                V[j*8+(k+4)][c+49] = V16_49[j*4+k][c]
    for i in range(4):
        for k in range(4):
            for c in range(49):
                W[k*4+i][c] = W16_49[k*4+i][c]
                W[(k+4)*4+i][c+49] = W16_49[k*4+i][c]
    return U, V, W


def _verify(U, V, W):
    n, m, p = 4, 4, 8
    rank = len(U[0])
    import random
    random.seed(0)
    for _ in range(2000):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        s = 0
        for r in range(rank):
            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        if s != 1:
            return False, (i,j,k,s)
    return True, None


def construct():
    start = time.time()
    deadline = start + 100
    U, V, W = _build_4x4_4x8()
    ok, info = _verify(U, V, W)
    if not ok:
        n, m, p = 4, 4, 8
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
