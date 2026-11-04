"""Matrix multiplication 4x5x5 rank-minimal decomposition.
Uses Strassen's 7-term algorithm on 2x2x2 sub-blocks to reduce rank
from the schoolbook 100 to 92 (measured improvement)."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """(U,V,W) for 2x2x2 matmul, rank 7, trace convention W[k*n+i,r]."""
    n = m = p = 2
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    def A(i, j): return i*m + j
    def B(j, k): return j*p + k
    def C(i, k): return k*n + i
    # M1 = (A00+A11)(B00+B11)
    U[A(0,0)][0] += 1; U[A(1,1)][0] += 1
    V[B(0,0)][0] += 1; V[B(1,1)][0] += 1
    W[C(0,0)][0] += 1; W[C(1,1)][0] += 1
    # M2 = (A10+A11) B00
    U[A(1,0)][1] += 1; U[A(1,1)][1] += 1
    V[B(0,0)][1] += 1
    W[C(1,0)][1] += 1; W[C(1,1)][1] -= 1
    # M3 = A00 (B10-B11)
    U[A(0,0)][2] += 1
    V[B(1,0)][2] += 1; V[B(1,1)][2] -= 1
    W[C(0,1)][2] += 1; W[C(1,1)][2] += 1
    # M4 = A11 (B01-B11)
    U[A(1,1)][3] += 1
    V[B(0,1)][3] += 1; V[B(1,1)][3] -= 1
    W[C(0,0)][3] += 1; W[C(1,0)][3] += 1
    # M5 = (A00+A01) B11
    U[A(0,0)][4] += 1; U[A(0,1)][4] += 1
    V[B(1,1)][4] += 1
    W[C(0,0)][4] -= 1; W[C(0,1)][4] += 1
    # M6 = (A10-A00)(B00+B10)
    U[A(1,0)][5] += 1; U[A(0,0)][5] -= 1
    V[B(0,0)][5] += 1; V[B(1,0)][5] += 1
    W[C(1,1)][5] += 1
    # M7 = (A01-A11)(B01+B11)
    U[A(0,1)][6] += 1; U[A(1,1)][6] -= 1
    V[B(0,1)][6] += 1; V[B(1,1)][6] += 1
    W[C(0,0)][6] += 1
    return U, V, W

def _schoolbook_block(n, m, p):
    U = [[0]*(n*m*p) for _ in range(n*m)]
    V = [[0]*(n*m*p) for _ in range(m*p)]
    W = [[0]*(n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    n_parts = [2, 2]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    assert sum(n_parts) == n and sum(m_parts) == m and sum(p_parts) == p

    strassen = _strassen_2x2()
    terms = []
    for ib in range(len(n_parts)):
        for jb in range(len(m_parts)):
            for kb in range(len(p_parts)):
                ni = n_parts[ib]
                mj = m_parts[jb]
                pk = p_parts[kb]
                if ni == 2 and mj == 2 and pk == 2:
                    U_b, V_b, W_b = strassen
                    rank_b = 7
                else:
                    U_b, V_b, W_b = _schoolbook_block(ni, mj, pk)
                    rank_b = ni*mj*pk
                terms.append((U_b, V_b, W_b, ni, mj, pk, rank_b, ib, jb, kb))

    total_rank = sum(t[6] for t in terms)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    n_off = [0]
    for s in n_parts:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_parts:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_parts:
        p_off.append(p_off[-1] + s)

    col = 0
    for (U_b, V_b, W_b, ni, mj, pk, rank_b, ib, jb, kb) in terms:
        i0 = n_off[ib]
        j0 = m_off[jb]
        k0 = p_off[kb]
        for r in range(rank_b):
            for i in range(ni):
                for j in range(mj):
                    uval = U_b[i*mj + j][r]
                    if uval:
                        U[(i0+i)*m + (j0+j)][col] += uval
            for j in range(mj):
                for k in range(pk):
                    vval = V_b[j*pk + k][r]
                    if vval:
                        V[(j0+j)*p + (k0+k)][col] += vval
            for k in range(pk):
                for i in range(ni):
                    wval = W_b[k*ni + i][r]
                    if wval:
                        W[(k0+k)*n + (i0+i)][col] += wval
        col += rank_b

    assert col == total_rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
