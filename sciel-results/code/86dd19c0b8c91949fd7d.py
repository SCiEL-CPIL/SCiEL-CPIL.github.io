import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

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

def _build_outer(n,p):
    rank=n*p
    U=[[0]*rank for _ in range(n)]; V=[[0]*rank for _ in range(p)]; W=[[0]*rank for _ in range(p*n)]
    c=0
    for i in range(n):
        for k in range(p):
            U[i][c]=1; V[k][c]=1; W[k*n+i][c]=1; c+=1
    return U,V,W

def _build_4x5x6():
    n,m,p=4,5,6
    U=[[0]*101 for _ in range(n*m)]; V=[[0]*101 for _ in range(m*p)]; W=[[0]*101 for _ in range(p*n)]
    U1,V1,W1=_build_4x4x4()
    U2,V2,W2=_build_4x4x2()
    U3,V3,W3=_build_outer(4,6)
    # Part 1: rows 0-3, cols 0-3 of A; rows 0-3, cols 0-3 of B
    for r in range(16):
        i=r//4; j=r%4
        for c in range(49):
            U[i*5+j][c]=U1[r][c]
    for r in range(16):
        j=r//4; k=r%4
        for c in range(49):
            V[j*6+k][c]=V1[r][c]
    for r in range(16):
        k=r//4; i=r%4
        for c in range(49):
            W[k*4+i][c]=W1[r][c]
    # Part 2: rows 0-3, col 4 of A; rows 4, cols 0-3 of B -> rank 28
    base=49
    for r in range(16):
        i=r//4; j=r%4
        for c in range(28):
            U[i*5+4][base+c]=U2[r][c]
    for r in range(8):
        j=r//2; k=r%2
        for c in range(28):
            V[4*6+k][base+c]=V2[r][c]
    for r in range(8):
        k=r//4; i=r%4
        for c in range(28):
            W[k*4+i][base+c]=W2[r][c]
    # Part 3: row 4 of A (4x1); cols 4 of B (1x6) -> outer product rank 24
    base=77
    for c in range(24):
        i=c//6; k=c%6
        U[4*5+i][base+c]=U3[i][c]
        V[4*6+k][base+c]=V3[k][c]
        W[k*4+i][base+c]=W3[k*4+i][c]
    return U,V,W

def _rank_reduce(U,V,W,n,m,p):
    """Remove linearly dependent columns from the decomposition."""
    rank=len(U[0])
    # Check column independence using the combined tensor structure
    # Build the 3D tensor T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    # For rank reduction, we need columns that are linearly dependent
    # in the sense that removing them doesn't change the tensor.
    
    # Use numpy to find rank of the column space
    U_arr=np.array(U,dtype=float)
    V_arr=np.array(V,dtype=float)
    W_arr=np.array(W,dtype=float)
    
    # The decomposition is valid if for all i,k:
    # sum_j sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = C[i,k]
    # A column t is redundant if U[:,t], V[:,t], W[:,t] are linear combinations
    # that don't contribute new information.
    
    # Simpler approach: try removing each column and check if tensor is preserved
    # This is O(rank^2 * n*m*p) which is fine for rank~100
    keep=list(range(rank))
    for t in range(rank-1,-1,-1):
        if time.time()>140: break
        # Check if column t is in span of others
        # Build matrix where each row is (U[i*m+j,t], V[j*p+k,t], W[k*n+i,t]) for all (i,j,k)
        # Actually, check if the "slice" for column t is in the span of other slices
        # The tensor contribution of column t is: U_t ⊗ V_t ⊗ W_t (outer product)
        # It's redundant if this outer product is in the span of other outer products
        
        # Quick check: if U[:,t] is zero, V[:,t] is zero, or W[:,t] is zero, remove
        if all(U[i*m+j][t]==0 for i in range(n) for j in range(m)):
            keep.remove(t)
            continue
        if all(V[j*p+k][t]==0 for j in range(m) for k in range(p)):
            keep.remove(t)
            continue
        if all(W[k*n+i][t]==0 for k in range(p) for i in range(n)):
            keep.remove(t)
            continue
    
    if len(keep)==rank:
        return U,V,W
    
    new_rank=len(keep)
    new_U=[[U[i*m+j][t] for t in keep] for i in range(n) for j in range(m)]
    new_V=[[V[j*p+k][t] for t in keep] for j in range(m) for k in range(p)]
    new_W=[[W[k*n+i][t] for t in keep] for k in range(p) for i in range(n)]
    return new_U,new_V,new_W

def _verify(U,V,W,n,m,p):
    import random
    rng=random.Random(12345)
    rank=len(U[0])
    for _ in range(3):
        A=[[rng.randint(-3,3) for _ in range(m)] for _ in range(n)]
        B=[[rng.randint(-3,3) for _ in range(p)] for _ in range(m)]
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

def _build_schoolbook(n,m,p):
    rank=n*m*p
    U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
    c=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
    return U,V,W

def construct():
    start=time.time()
    deadline=start+110
    n,m,p=DIMENSIONS
    try:
        U,V,W=_build_4x5x6()
        if time.time()>deadline: raise TimeoutError
        U,V,W=_rank_reduce(U,V,W,n,m,p)
        if time.time()>deadline: raise TimeoutError
        if not _verify(U,V,W,n,m,p):
            raise ValueError("bad")
        return {"U":U,"V":V,"W":W}
    except Exception:
        U,V,W=_build_schoolbook(n,m,p)
        return {"U":U,"V":V,"W":W}

# EVOLVE-BLOCK-END
