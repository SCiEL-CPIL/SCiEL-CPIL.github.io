import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _strassen_2x2():
    U=[[0]*7 for _ in range(4)]; V=[[0]*7 for _ in range(4)]; W=[[0]*7 for _ in range(4)]
    defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for t,(ue,ve,we) in enumerate(defs):
        for (i,j,c) in ue: U[i*2+j][t]+=c
        for (j,k,c) in ve: V[j*2+k][t]+=c
        for (k,i,c) in we: W[k*2+i][t]+=c
    return U,V,W

def _build_4x4x4():
    U=[[0]*49 for _ in range(16)]; V=[[0]*49 for _ in range(16)]; W=[[0]*49 for _ in range(16)]
    U2,V2,W2 = _strassen_2x2()
    outer = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for s,(ue,ve,we) in enumerate(outer):
        for t in range(7):
            col = s*7+t
            for (bi,bj,c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi*2+i; gj = bj*2+j
                        a_loc = i*2+j
                        cc = U2[a_loc][t]
                        if cc: U[gi*4+gj][col] += cc*c
            for (bj,bk,c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj*2+j; gk = bk*2+k
                        b_loc = j*2+k
                        cc = V2[b_loc][t]
                        if cc: V[gj*4+gk][col] += cc*c
            for (bk,bi,c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk*2+k; gi = bi*2+i
                        c_loc = k*2+i
                        cc = W2[c_loc][t]
                        if cc: W[gk*4+gi][col] += cc*c
    return U,V,W

def _build_4x4x2():
    U=[[0]*28 for _ in range(16)]; V=[[0]*28 for _ in range(8)]; W=[[0]*28 for _ in range(8)]
    U2,V2,W2 = _strassen_2x2()
    blocks = [(0,0,0,0,0),(0,2,2,0,0),(2,0,0,2,0),(2,2,2,2,0)]
    for bi,(ar,ac,br,cr,ck) in enumerate(blocks):
        base = bi*7
        for t in range(7):
            col = base+t
            for i in range(2):
                for j in range(2):
                    a_loc=i*2+j; cc=U2[a_loc][t]
                    if cc: U[(ar+i)*4+(ac+j)][col]+=cc
            for j in range(2):
                for k in range(2):
                    b_loc=j*2+k; cc=V2[b_loc][t]
                    if cc: V[(br+j)*2+(ck+k)][col]+=cc
            for k in range(2):
                for i in range(2):
                    c_loc=k*2+i; cc=W2[c_loc][t]
                    if cc: W[(ck+k)*4+(cr+i)][col]+=cc
    return U,V,W

def _build_4x2x4():
    # (4x2)*(2x4) rank 28
    U=[[0]*28 for _ in range(8)]; V=[[0]*28 for _ in range(8)]; W=[[0]*28 for _ in range(16)]
    U2,V2,W2 = _strassen_2x2()
    blocks = [(0,0,0,0,0),(0,0,2,0,0),(2,0,0,2,0),(2,0,2,2,0)]
    for bi,(ar,ac,br,bk,cr,ck) in enumerate(blocks):
        base = bi*7
        for t in range(7):
            col = base+t
            for i in range(2):
                for j in range(2):
                    a_loc=i*2+j; cc=U2[a_loc][t]
                    if cc: U[(ar+i)*2+(ac+j)][col]+=cc
            for j in range(2):
                for k in range(2):
                    b_loc=j*2+k; cc=V2[b_loc][t]
                    if cc: V[(br+j)*2+(bk+k)][col]+=cc
            for k in range(2):
                for i in range(2):
                    c_loc=k*2+i; cc=W2[c_loc][t]
                    if cc: W[(cr+k)*4+(ci+i)][col]+=cc
    return U,V,W

def _build_4x2x2():
    # (4x2)*(2x2) rank 14
    U=[[0]*14 for _ in range(8)]; V=[[0]*14 for _ in range(4)]; W=[[0]*14 for _ in range(8)]
    U2,V2,W2 = _strassen_2x2()
    blocks = [(0,0,0,0,0),(0,0,0,2,0),(2,0,0,0,0),(2,0,0,2,0)]
    for bi,(ar,ac,br,bk,cr,ck) in enumerate(blocks):
        base = bi*7
        for t in range(7):
            col = base+t
            for i in range(2):
                for j in range(2):
                    a_loc=i*2+j; cc=U2[a_loc][t]
                    if cc: U[(ar+i)*2+(ac+j)][col]+=cc
            for j in range(2):
                for k in range(2):
                    b_loc=j*2+k; cc=V2[b_loc][t]
                    if cc: V[(br+j)*2+(bk+k)][col]+=cc
            for k in range(2):
                for i in range(2):
                    c_loc=k*2+i; cc=W2[c_loc][t]
                    if cc: W[(cr+k)*4+(ci+i)][col]+=cc
    return U,V,W

def _build_2x3x3():
    # (2x3)*(3x3) rank 14 via 2x2 Strassen + 2x1x1 + 1x2x1 + 1x1x1
    # Actually, let's use a simpler approach: 2x3 * 3x3
    # Split into 2x2*2x2 (rank 7), 2x1*1x2 (rank 4), 1x2*2x1 (rank 4), 1x1*1x1 (rank 1)
    # Total: 7 + 4 + 4 + 1 = 16
    # But we can do better. Let's just use schoolbook for this small case: 2*3*3 = 18
    # Or use 2x2 Strassen for the 2x2 part and schoolbook for the rest.
    # For now, return schoolbook.
    rank = 2*3*3
    U=[[0]*rank for _ in range(6)]; V=[[0]*rank for _ in range(9)]; W=[[0]*rank for _ in range(6)]
    c=0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][c]=1; V[j*3+k][c]=1; W[k*2+i][c]=1; c+=1
    return U,V,W

def _combine(parts, n, m, p):
    total = sum(len(d[0][0]) for d in parts)
    U=[[0]*total for _ in range(n*m)]; V=[[0]*total for _ in range(m*p)]; W=[[0]*total for _ in range(p*n)]
    off=0
    for (Us,Vs,Ws, (i0,j0,ii,jj,k0,kk)) in parts:
        nA=ii; mA=jj; nB=kk;
        for r in range(nA*mA):
            i=r//mA; j=r%mA
            for c in range(len(Us[0])):
                U[(i0+i)*m+(j0+j)][off+c]=Us[r][c]
        for r in range(mA*nB):
            j=r//nB; k=r%nB
            for c in range(len(Vs[0])):
                V[(j0+j)*p+(k0+k)][off+c]=Vs[r][c]
        for r in range(nB*nA):
            k=r//nA; i=r%nA
            for c in range(len(Ws[0])):
                W[(k0+k)*n+(i0+i)][off+c]=Ws[r][c]
        off+=len(Us[0])
    return U,V,W

def _verify(U,V,W,n,m,p,trials=5,seed=12345):
    rng=random.Random(seed)
    rank=len(U[0])
    for _ in range(trials):
        A=[[rng.randint(-5,5) for _ in range(m)] for _ in range(n)]
        B=[[rng.randint(-5,5) for _ in range(p)] for _ in range(m)]
        C=[[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val=0
                for t in range(rank):
                    au=sum(U[i*m+j][t]*A[i][j] for j in range(m))
                    bu=sum(V[j*p+k][t]*B[j][k] for j in range(m))
                    val+=W[k*n+i][t]*au*bu
                if val!=C[i][k]:
                    return False
    return True

def _greedy_peel(n, m, p, time_limit=100):
    """
    Greedy peeling: at each step, find a rank-1 term that best approximates the residual.
    The residual is a 4-index tensor T[i,j,l,k] = delta_{j,l} initially.
    A rank-1 term is (U,V,W) where U[i,j], V[l,k], W[k,i].
    We want sum_t W[k,i,t] U[i,j,t] V[l,k,t] = delta_{j,l}.
    
    For a given (U,V,W), the contribution to T[i,j,l,k] is W[k,i] U[i,j] V[l,k].
    We want to minimize the Frobenius norm of the residual.
    
    This is a hard optimization problem. Let's use a simpler heuristic:
    At each step, pick (i0,j0,k0,l0) that maximizes |T[i0,j0,l0,k0]|, and set
    U[i0,j0]=1, V[l0,k0]=1, W[k0,i0]=1, and subtract this rank-1 term.
    
    But this is just schoolbook. To do better, we need to allow fractional values
    and optimize over all (i,j,l,k) simultaneously.
    
    Given the time constraint, let's try a random restart approach:
    1. Initialize U,V,W randomly with small values.
    2. For each (i,j,l,k), compute the residual R[i,j,l,k] = delta_{j,l} - sum_t W[k,i,t] U[i,j,t] V[l,k,t].
    3. Pick the (i,j,l,k) with largest |R|, and adjust U,V,W to reduce it.
    
    This is still very slow. Let's just return schoolbook for now.
    """
    return _build_schoolbook(n, m, p)

def _build_kronecker(n, m, p):
    """
    Kronecker product construction.
    For 4x5 * 5x6, we can view this as (2x2) x (2) for A, and (2) x (2x2) for B?
    Actually, 4 = 2*2, 5 = 5, 6 = 2*3.
    
    The multiplication tensor for (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2) can be
    decomposed using the Kronecker product of the multiplication tensors for
    (n1 x m1) * (m1 x p1) and (n2 x m2) * (m2 x p2).
    
    But here, the middle dimension is 5, which doesn't factor nicely.
    
    Alternative: use the fact that 4x5 * 5x6 can be embedded in 4x6 * 6x6,
    and use known decompositions for 4x6 * 6x6.
    
    For now, let's try a different approach: use the existing Strassen tiling
    and see if we can improve it.
    """
    # Try the tiling from inspiration 1:
    # A: 4x5, B: 5x6
    # Split A into [A11 A12] where A11 is 4x4, A12 is 4x1
    # Split B into [B11 B12; B21 B22] where B11 is 4x4, B12 is 4x2, B21 is 1x4, B22 is 1x2
    # C = [C11 C12; C21 C22] where C11 is 4x4, C12 is 4x2, C21 is 4x4, C22 is 4x2
    # Wait, C is 4x6, so C11 is 4x4, C12 is 4x2.
    
    parts = []
    
    # C11 = A11*B11: 4x4 * 4x4, rank 49
    U1,V1,W1 = _build_4x4x4()
    parts.append((U1,V1,W1,(0,0,4,4,0,4)))
    
    # C12 = A11*B12 + A12*B22: (4x4)*(4x2) + (4x1)*(1x2)
    # A11*B12: 4x4 * 4x2, rank 28
    U2,V2,W2 = _build_4x4x2()
    parts.append((U2,V2,W2,(0,0,4,4,4,2)))
    
    # A12*B22: (4x1)*(1x2)
    U3 = [[0]*8 for _ in range(4)]
    V3 = [[0]*8 for _ in range(2)]
    W3 = [[0]*8 for _ in range(8)]
    c=0
    for i in range(4):
        for k in range(2):
            U3[i][c]=1; V3[0][c]=1; W3[k*4+i][c]=1; c+=1
    parts.append((U3,V3,W3,(0,4,4,1,0,2)))
    
    U,V,W = _combine(parts, n, m, p)
    return U,V,W

def construct():
    start=time.time()
    n,m,p=DIMENSIONS
    deadline=start+120
    
    try:
        # Try Kronecker tiling
        U,V,W = _build_kronecker(n,m,p)
        rank=len(U[0])
        
        if time.time()>deadline-20:
            U,V,W = _build_schoolbook(n,m,p)
            return {"U":U,"V":V,"W":W}
        
        if _verify(U,V,W,n,m,p,trials=5):
            if rank < n*m*p:
                return {"U":U,"V":V,"W":W}
        
        # Fallback to schoolbook
        U,V,W = _build_schoolbook(n,m,p)
        return {"U":U,"V":V,"W":W}
    
    except Exception:
        U,V,W = _build_schoolbook(n,m,p)
        return {"U":U,"V":V,"W":W}

# EVOLVE-BLOCK-END
