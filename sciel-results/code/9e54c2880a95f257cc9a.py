"""Rewritten program: tensor-product of Strassen for 4x4x8 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 8]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# Strassen 2x2x2 decomposition (rank 7)
# Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
# For 2x2: U has 4 rows, V has 4 rows, W has 4 rows, 7 columns
_STRASSEN_U = [
    [1, 0, 0, 0, 0, 0, 0],  # a11
    [0, 0, 1, 1, 0, 0, 0],  # a12
    [0, 1, 0, 0, 1, 1, 0],  # a21
    [0, 0, 0, 0, 0, 0, 1],  # a22
]
_STRASSEN_V = [
    [0, 0, 0, 0, 0, 0, 0],  # b11 (row j=0,k=0)
    [0, 0, 0, 0, 0, 0, 0],  # b12 (row j=0,k=1)
    [1, 0, 0, 0, 0, 0, 0],  # b21 (row j=1,k=0)
    [0, 1, 0, 0, 0, 0, 0],  # b22 (row j=1,k=1)
]
_STRASSEN_W = [
    [1, 0, 0, 0, 0, 1, 0],  # c11 (row k=0,i=0)
    [0, 0, 1, 0, 0, 0, 0],  # c12 (row k=0,i=1)
    [0, 0, 0, 1, 0, 0, 0],  # c21 (row k=1,i=0)
    [0, 1, 0, 0, 1, 0, 1],  # c22 (row k=1,i=1)
]

# Let me re-derive Strassen carefully.
# Standard Strassen for C = A*B where A,B are 2x2:
# A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
#
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-d)(f+h)
# M7 = (b-d)(e+g)
#
# c11 = M1 - M4 + M5 + M6
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M3 + M2 - M7
#
# U[i*2+j, c] corresponds to A element (i,j):
# U[0,0]=a=1, U[0,1]=b=0, U[0,2]=c=0, U[0,3]=d=0, U[0,4]=0, U[0,5]=0, U[0,6]=0
# Wait, let me index properly.
# U row index = i*m + j where m=2 for the 2x2 case.
# U[0] = a (i=0,j=0), U[1] = b (i=0,j=1), U[2] = c (i=1,j=0), U[3] = d (i=1,j=1)
#
# M1 = (a+d)(e+h): U row: a->1, d->1; V: e->1, h->1; W: c11->1
# M2 = (c+d)e: U: c->1, d->1; V: e->1; W: c21->1
# M3 = a(f-h): U: a->1; V: f->1, h->-1; W: c12->1, c22->-1
# M4 = d(g-e): U: d->1; V: g->1, e->-1; W: c11->-1, c21->1
# M5 = (a+b)h: U: a->1, b->1; V: h->1; W: c11->1, c12->1
# M6 = (c-d)(f+h): U: c->1, d->-1; V: f->1, h->1; W: c11->1, c22->-1
# M7 = (b-d)(e+g): U: b->1, d->-1; V: e->1, g->1; W: c22->-1
#
# V row index = j*p + k where p=2 for 2x2.
# V[0]=e (j=0,k=0), V[1]=f (j=0,k=1), V[2]=g (j=1,k=0), V[3]=h (j=1,k=1)
#
# W row index = k*n + i where n=2 for 2x2.
# W[0]=c11 (k=0,i=0), W[1]=c12 (k=0,i=1), W[2]=c21 (k=1,i=0), W[3]=c22 (k=1,i=1)

_STRASSEN_U = [
    # a (row 0)
    [1, 0, 1, 0, 1, 0, 0],
    # b (row 1)
    [0, 0, 0, 0, 1, 0, 1],
    # c (row 2)
    [0, 1, 0, 0, 0, 1, 0],
    # d (row 3)
    [1, 1, 0, 1, 0, -1, -1],
]
_STRASSEN_V = [
    # e (row 0)
    [1, 1, 0, -1, 0, 0, 1],
    # f (row 1)
    [0, 0, 1, 0, 0, 1, 0],
    # g (row 2)
    [0, 0, 0, 1, 0, 0, 1],
    # h (row 3)
    [1, 0, -1, 0, 1, 1, 0],
]
_STRASSEN_W = [
    # c11 (row 0)
    [1, 0, 0, -1, 1, 1, 0],
    # c12 (row 1)
    [0, 0, 1, 0, 1, 0, 0],
    # c21 (row 2)
    [0, 1, 0, 1, 0, 0, 0],
    # c22 (row 3)
    [0, -1, -1, 0, 0, -1, -1],
]


def _tensor_product_matrices(U1, V1, W1, U2, V2, W2):
    """Compute tensor product of two bilinear decompositions.
    U1,V1,W1: decompositions for size (n1,m1,p1)
    U2,V2,W2: decompositions for size (n2,m2,p2)
    Result: decomposition for (n1*n2, m1*m2, p1*p2)
    
    U row index: (i1*m1+j1)*m2+j2 = i1*m1*m2 + j1*m2 + j2
    V row index: (j1*p1+k1)*p2+k2 = j1*p1*p2 + k1*p2 + k2  
    W row index: (k1*n1+i1)*n2+i2 = k1*n1*n2 + i1*n2 + i2
    """
    n1, m1, p1 = len(W1) // 2, 2, 2  # for 2x2 case
    # Actually let's infer from dimensions
    # U1 has n1*m1 rows, V1 has m1*p1 rows, W1 has p1*n1 rows
    # U2 has n2*m2 rows, V2 has m2*p2 rows, W2 has p2*n2 rows
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    
    n1m1 = len(U1)
    m1p1 = len(V1)
    p1n1 = len(W1)
    n2m2 = len(U2)
    m2p2 = len(V2)
    p2n2 = len(W2)
    
    # Infer n1, m1, p1 from the 2x2 case: n1=m1=p1=2
    # In general for tensor product we need to know the individual dimensions
    # For our case both are 2x2x2 so n1=m1=p1=n2=m2=p2=2
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 2
    
    U = [[0] * rank for _ in range(n1 * m1 * n2 * m2)]
    V = [[0] * rank for _ in range(m1 * p1 * m2 * p2)]
    W = [[0] * rank for _ in range(p1 * n1 * p2 * n2)]
    
    for a in range(r1):
        for b in range(r2):
            col = a * r2 + b
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val = U1[i1 * m1 + j1][a] * U2[i2 * m2 + j2][b]
                            if val != 0:
                                U[(i1 * m1 + j1) * m2 * n2 + i2 * m2 + j2][col] = val
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val = V1[j1 * p1 + k1][a] * V2[j2 * p2 + k2][b]
                            if val != 0:
                                V[(j1 * p1 + k1) * p2 * m2 + j2 * p2 + k2][col] = val
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val = W1[k1 * n1 + i1][a] * W2[k2 * n2 + i2][b]
                            if val != 0:
                                W[(k1 * n1 + i1) * n2 * p2 + k2 * n2 + i2][col] = val
    
    return U, V, W


def construct():
    if time.time() > _DEADLINE:
        # Fallback: schoolbook
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
    
    # Step 1: Tensor product of two Strassen 2x2x2 to get 4x4x4 rank 49
    U44, V44, W44 = _tensor_product_matrices(
        _STRASSEN_U, _STRASSEN_V, _STRASSEN_W,
        _STRASSEN_U, _STRASSEN_V, _STRASSEN_W
    )
    
    # Step 2: For 4x4x8, replicate the 4x4x4 decomposition for two 4-column blocks
    # The 8 output columns are split into two groups of 4.
    # For output column k (0..7), k = k4 + 4*k8 where k4 in 0..3, k8 in 0..1
    # Actually for 4x4x8: A is 4x4, B is 4x8, C is 4x8
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can think of B as two 4x4 blocks: B[:,0:4] and B[:,4:8]
    # Each block uses the same 4x4x4 decomposition
    # So total rank = 49 + 49 = 98
    
    rank44 = len(U44[0])  # should be 49
    total_rank = rank44 * 2  # 98
    
    n, m, p = 4, 4, 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(2):
        for r in range(rank44):
            col = block * rank44 + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U44[i * m + j][r]
            for j in range(m):
                for k4 in range(4):
                    k = k4 + block * 4
                    V[j * p + k][col] = V44[j * 4 + k4][r]
            for k4 in range(4):
                for i in range(n):
                    k = k4 + block * 4
                    W[k * n + i][col] = W44[k4 * n + i][r]
    
    return {"U": U, "V": V, "W": W}
