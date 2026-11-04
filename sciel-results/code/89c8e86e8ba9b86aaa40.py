"""Program for 4x4x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    # 7 rank-1 terms for 2x2x2 matrix multiplication (1x1 blocks).
    # Each term: (A_dict, B_dict, C_dict) over indices in {0,1}^2.
    # A[i,j], B[j,k], C[k,i].
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00 +=, C11 +=
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
    # M2 = (A10+A11)B00 -> C10 +=
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}))
    # M3 = A00(B01-B11) -> C01 +=
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}))
    # M4 = A11(B10-B00) -> C10 +=
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,0):1}))
    # M5 = (A00+A01)B11 -> C01 +=
    terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1}))
    # M6 = (A10-A00)(B00+B01) -> C11 +=
    terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)(B10+B11) -> C00 +=
    terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
    return terms

def _build_4x4x4_strassen():
    # Compose outer 2x2x2 (block level) with inner 2x2x2 (sub-block level).
    outer = _strassen_2x2_terms()
    inner = _strassen_2x2_terms()
    terms = []
    for oA, oB, oC in outer:
        for iA, iB, iC in inner:
            A = {}
            B = {}
            C = {}
            for (bI, bJ), oa in oA.items():
                for (li, lj), ia in iA.items():
                    gi, gj = bI*2+li, bJ*2+lj
                    A[(gi,gj)] = A.get((gi,gj),0) + oa*ia
            for (bJ, bK), ob in oB.items():
                for (lj, lk), ib in iB.items():
                    gj, gk = bJ*2+lj, bK*2+lk
                    B[(gj,gk)] = B.get((gj,gk),0) + ob*ib
            for (bK, bI), oc in oC.items():
                for (lk, li), ic in iC.items():
                    gk, gi = bK*2+lk, bI*2+li
                    C[(gk,gi)] = C.get((gk,gi),0) + oc*ic
            terms.append((A,B,C))
    return terms

def _decompose_4x4x4(terms):
    # Build U (16xR), V (20xR), W (20xR) for 4x4x4 part (k=0..3).
    n=m=p=4
    R = len(terms)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    for c,(A,B,C) in enumerate(terms):
        for (i,j),val in A.items():
            U[i*m+j][c] += val
        for (j,k),val in B.items():
            V[j*p+k][c] += val
        for (k,i),val in C.items():
            W[k*n+i][c] += val
    return U,V,W

def _schoolbook(n,m,p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1
                V[j*p+k][col]=1
                W[k*n+i][col]=1
                col+=1
    return U,V,W

def construct():
    n,m,p = DIMENSIONS
    # Try Strassen 4x4x4 (rank 49) + schoolbook 5th column (rank 16) = 65.
    try:
        terms = _build_4x4x4_strassen()
        U4,V4,W4 = _decompose_4x4x4(terms)
        R1 = len(terms)
        # 16 terms for k=4
        R2 = 16
        rank = R1 + R2
        U = [row[:] for row in U4]
        V = [row[:] for row in V4]
        W = [row[:] for row in W4]
        for r in range(rank - R1):
            for row in U: row.append(0)
            for row in V: row.append(0)
            for row in W: row.append(0)
        # add 16 terms
        t=0
        for i in range(n):
            for j in range(m):
                U[i*m+j][R1+t]=1
                V[j*p+4][R1+t]=1
                W[4*n+i][R1+t]=1
                t+=1
        # Verify
        import numpy as np
        A = np.random.rand(n,m)
        B = np.random.rand(m,p)
        C = A @ B
        # compute decomposition
        Ua=np.array(U); Va=np.array(V); Wa=np.array(W)
        # C[i,k] = sum_j A[i,j] B[j,k] ; our W index is k*n+i
        Cdec = np.zeros((n,p))
        for i in range(n):
            for k in range(p):
                s=0.0
                for j in range(m):
                    s += sum(Ua[i*m+j,c]*Va[j*p+k,c]*Wa[k*n+i,c] for c in range(rank))
                Cdec[i,k]=s
        if not np.allclose(Cdec, C, atol=1e-9):
            raise ValueError("Strassen verify failed")
        return {"U":U,"V":V,"W":W}
    except Exception:
        U,V,W = _schoolbook(n,m,p)
        return {"U":U,"V":V,"W":W}
# EVOLVE-BLOCK-END
