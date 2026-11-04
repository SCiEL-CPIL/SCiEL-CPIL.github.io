import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    start_time = time.time()
    deadline = 140  # safety margin before the 150 s kill

    # ---- Target tensor T[a,b,c] with a=i*m+j, b=j*p+k, c=k*n+i ----
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def to_lists(mat):
        return mat.tolist()

    def verify_float(U, V, W, tol=1e-6):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T_target, atol=tol)

    # =================================================================
    # Rank-17 construction:
    #   - 7 Strassen terms for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    #   - 10 schoolbook terms for the remaining (i,j,k) triples
    #     (i in {0,1}, j in {0,1}, k=2) -> 2 terms
    #     (i in {0,1}, j=2, k in {0,1,2}) -> 6 terms
    #     (i in {0,1}, j=2, k=2) already counted -> total 2+6 = 8? 
    #     Wait: remaining triples are all (i,j,k) NOT in the 2x2x2 block.
    #     Total triples = 2*3*3 = 18. Block has 2*2*2 = 8. Remaining = 10.
    #     So 7 + 10 = 17.
    # =================================================================

    U = np.zeros((nm, 17))
    V = np.zeros((mp, 17))
    W = np.zeros((pn, 17))

    col = 0

    # ---- Strassen 2x2x2 on A[0:2, 0:2] * B[0:2, 0:2] ----
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global: a = i*3+j, b = j*3+k, c = k*2+i (since n=2, so c = k*n+i = k*2+i)
    #
    # Strassen 7 products for 2x2 matrices:
    # Let A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # C = [[m1-m4+m6-m7, m3+m5+m7],
    #      [m2+m4-m6,     m1-m2+m5]]
    #
    # In our tensor: C[i,k] gets contribution from A[i,j]*B[j,k] summed over j.
    # For the 2x2 block, C[i,k] for i in {0,1}, k in {0,1} is exactly the 2x2 product.
    # Global c index = k*2+i.
    #
    # For each Strassen term m_t, we need U[a, col], V[b, col], W[c, col] such that
    # the term contributes to the correct (a,b,c) positions.
    #
    # A[i,j] global index: a = i*3 + j
    # B[j,k] global index: b = j*3 + k
    # C[i,k] global index: c = k*2 + i

    # m1 = (a00+a11)(b00+b11)
    # U: a00=0*3+0=0, a11=1*3+1=4 -> U[0,col]=1, U[4,col]=1
    # V: b00=0*3+0=0, b11=1*3+1=4 -> V[0,col]=1, V[4,col]=1
    # W: C[0,0]=0*2+0=0, C[1,1]=1*2+1=3 -> W[0,col]=1, W[3,col]=1
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1

    # m2 = (a10+a11)b00
    # U: a10=1*3+0=3, a11=1*3+1=4 -> U[3,col]=1, U[4,col]=1
    # V: b00=0*3+0=0 -> V[0,col]=1
    # W: C[1,0]=0*2+1=1, C[1,1]=1*2+1=3 -> W[1,col]=1, W[3,col]=1
    # But C[1,1] gets m2 with -1 sign, C[1,0] gets m2 with +1 sign.
    # So W[1,col]=1, W[3,col]=-1
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[3, col] = -1
    col += 1

    # m3 = a00(b01-b11)
    # U: a00=0 -> U[0,col]=1
    # V: b01=0*3+1=1, b11=1*3+1=4 -> V[1,col]=1, V[4,col]=-1
    # W: C[0,1]=1*2+0=2 -> W[2,col]=1
    U[0, col] = 1
    V[1, col] = 1; V[4, col] = -1
    W[2, col] = 1
    col += 1

    # m4 = a11(b10-b00)
    # U: a11=1*3+1=4 -> U[4,col]=1
    # V: b10=1*3+0=3, b00=0 -> V[3,col]=1, V[0,col]=-1
    # W: C[0,0]=0 gets -m4, C[1,0]=1 gets +m4 -> W[0,col]=-1, W[1,col]=1
    U[4, col] = 1
    V[3, col] = 1; V[0, col] = -1
    W[0, col] = -1; W[1, col] = 1
    col += 1

    # m5 = (a00+a01)b11
    # U: a00=0, a01=0*3+1=1 -> U[0,col]=1, U[1,col]=1
    # V: b11=1*3+1=4 -> V[4,col]=1
    # W: C[0,1]=2 gets +m5, C[1,1]=3 gets +m5 -> W[2,col]=1, W[3,col]=1
    # But C[0,0]=0 gets -m5? No: C[0,0] = m1-m4+m6-m7, no m5.
    # C[0,1] = m3+m5+m7, C[1,1] = m1-m2+m5
    # So W[2,col]=1, W[3,col]=1
    U[0, col] = 1; U[1, col] = 1
    V[4, col] = 1
    W[2, col] = 1; W[3, col] = 1
    col += 1

    # m6 = (a10-a00)(b00+b01)
    # U: a10=3, a00=0 -> U[3,col]=1, U[0,col]=-1
    # V: b00=0, b01=1 -> V[0,col]=1, V[1,col]=1
    # W: C[0,0]=0 gets +m6, C[1,0]=1 gets -m6 -> W[0,col]=1, W[1,col]=-1
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[0, col] = 1; W[1, col] = -1
    col += 1

    # m7 = (a01-a11)(b10+b11)
    # U: a01=1, a11=4 -> U[1,col]=1, U[4,col]=-1
    # V: b10=3, b11=4 -> V[3,col]=1, V[4,col]=1
    # W: C[0,0]=0 gets -m7, C[0,1]=2 gets +m7 -> W[0,col]=-1, W[2,col]=1
    U[1, col] = 1; U[4, col] = -1
    V[3, col] = 1; V[4, col] = 1
    W[0, col] = -1; W[2, col] = 1
    col += 1

    # ---- Schoolbook for remaining 10 (i,j,k) triples ----
    block_ijk = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                block_ijk.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in block_ijk:
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    # col should be 17
    assert col == 17, f"Expected 17 terms, got {col}"

    # ---- Verify ----
    if not verify_float(U, V, W):
        # Fallback to schoolbook rank 18
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {
            "U": to_lists(U_sb),
            "V": to_lists(V_sb),
            "W": to_lists(W_sb)
        }

    return {
        "U": to_lists(U),
        "V": to_lists(V),
        "W": to_lists(W)
    }
