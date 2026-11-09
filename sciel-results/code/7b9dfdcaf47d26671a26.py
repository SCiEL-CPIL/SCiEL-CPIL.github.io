"""Matrix multiplication 4x4x9 tensor decomposition using Kronecker product structure."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
TIME_LIMIT = 120.0


def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _construct_strassen_2x2x2():
    """Strassen decomposition for 2x2x2, rank 7."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: (2*2, 7) = (4, 7) - coefficients for A elements
    # V: (2*2, 7) = (4, 7) - coefficients for B elements  
    # W: (2*2, 7) = (4, 7) - coefficients for C elements
    # Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    # A elements: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B elements: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C elements: C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l
    
    U = [[0.0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0.0]*7 for _ in range(4)]  # 2*2 = 4 rows
    W = [[0.0]*7 for _ in range(4)]  # 2*2 = 4 rows
    
    # M1 = (a+d)(e+h): A: a=1,d=1; B: e=1,h=1; C: i+=1, j+=0, k+=0, l+=1
    # W indexing: W[k*2+i, t]
    # C11=i: W[0*2+0]=W[0] += 1
    # C12=j: W[1*2+0]=W[2] += 0
    # C21=k: W[0*2+1]=W[1] += 0
    # C22=l: W[1*2+1]=W[3] += 1
    
    # t=0: M1
    U[0][0] = 1.0  # a
    U[3][0] = 1.0  # d
    V[0][0] = 1.0  # e
    V[3][0] = 1.0  # h
    W[0][0] = 1.0  # i (C[0,0])
    W[3][0] = 1.0  # l (C[1,1])
    
    # t=1: M2 = (c+d)e: A: c=1,d=1; B: e=1; C: k+=1
    U[2][1] = 1.0  # c
    U[3][1] = 1.0  # d
    V[0][1] = 1.0  # e
    W[1][1] = 1.0  # k (C[1,0])
    
    # t=2: M3 = a(f-h): A: a=1; B: f=1,h=-1; C: j+=1
    U[0][2] = 1.0  # a
    V[1][2] = 1.0  # f
    V[3][2] = -1.0 # h
    W[2][2] = 1.0  # j (C[0,1])
    
    # t=3: M4 = d(g-e): A: d=1; B: g=1,e=-1; C: i+=1, k+=1
    U[3][3] = 1.0  # d
    V[2][3] = 1.0  # g
    V[0][3] = -1.0 # e
    W[0][3] = 1.0  # i
    W[1][3] = 1.0  # k
    
    # t=4: M5 = (a+b)h: A: a=1,b=1; B: h=1; C: i-=1, j+=1
    U[0][4] = 1.0  # a
    U[1][4] = 1.0  # b
    V[3][4] = 1.0  # h
    W[0][4] = -1.0 # i
    W[2][4] = 1.0  # j
    
    # t=5: M6 = (c-a)(e+f): A: c=1,a=-1; B: e=1,f=1; C: l+=1
    U[2][5] = 1.0  # c
    U[0][5] = -1.0 # a
    V[0][5] = 1.0  # e
    V[1][5] = 1.0  # f
    W[3][5] = 1.0  # l
    
    # t=6: M7 = (b-d)(g+h): A: b=1,d=-1; B: g=1,h=1; C: i+=1, l-=1
    U[1][6] = 1.0  # b
    U[3][6] = -1.0 # d
    V[2][6] = 1.0  # g
    V[3][6] = 1.0  # h
    W[0][6] = 1.0  # i
    W[3][6] = -1.0 # l
    
    return U, V, W


def _construct_2x2x3():
    """Construct rank-11 decomposition for 2x2x3.
    
    Split the 3 columns into 2+1:
    - 2x2x2 part: Strassen, rank 7
    - 2x2x1 part: schoolbook, rank 4
    Total: 11
    """
    # Strassen 2x2x2
    U_s, V_s, W_s = _construct_strassen_2x2x2()
    
    # Schoolbook 2x2x1
    U_sc, V_sc, W_sc = _construct_schoolbook(2, 2, 1)
    
    # Combine: total rank = 7 + 4 = 11
    total_rank = 7 + 4
    U = [[0.0] * total_rank for _ in range(4)]  # 2*2
    V = [[0.0] * total_rank for _ in range(6)]  # 2*3
    W = [[0.0] * total_rank for _ in range(6)]  # 3*2
    
    # First 7 columns: 2x2x2 part (columns 0,1 of B and C)
    for c in range(7):
        for i in range(4):
            U[i][c] = U_s[i][c]
        for j in range(2):
            for k in range(2):
                V[j * 3 + k][c] = V_s[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][c] = W_s[k * 2 + i][c]
    
    # Last 4 columns: 2x2x1 part (column 2 of B and C)
    for c in range(4):
        col = 7 + c
        for i in range(4):
            U[i][col] = U_sc[i][c]
        for j in range(2):
            V[j * 3 + 2][col] = V_sc[j * 1 + 0][c]
        for i in range(2):
            W[2 * 2 + i][col] = W_sc[0 * 2 + i][c]
    
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Compute Kronecker product decomposition.
    
    Given decompositions of (n1,m1,p1) and (n2,m2,p2),
    produce decomposition of (n1*n2, m1*m2, p1*p2).
    
    Indexing:
    - U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
    - U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
    
    Result:
    - U: (n1*n2*m1*m2, r1*r2)
    - V: (m1*m2*p1*p2, r1*r2)
    - W: (p1*p2*n1*n2, r1*r2)
    
    For element (i,j,k) with i=i1*n2+i2, j=j1*m2+j2, k=k1*p2+k2:
    T[i,j,k] = T1[i1,j1,k1] * T2[i2,j2,k2]
    
    U[i1*m1+j1, t1] * U[i2*m2+j2, t2] -> U[(i1*n2+i2)*m1*m2 + (j1*m2+j2), t1*r2+t2]
    V[j1*p1+k1, t1] * V[j2*p2+k2, t2] -> V[(j1*m2+j2)*p1*p2 + (k1*p2+k2), t1*r2+t2]  
    Wait, V indexing is V[j*p+k, t]. For the product:
    V[(j1*m2+j2)*(p1*p2) + (k1*p2+k2), t1*r2+t2] = V1[j1*p1+k1, t1] * V2[j2*p2+k2, t2]
    
    Hmm, let me be more careful.
    
    For the 4x4x9 case:
    n=4=2*2, m=4=2*2, p=9=3*3
    i = i1*2 + i2, j = j1*2 + j2, k = k1*3 + k2
    where i1,j1,k1 in {0,1}, i2,j2 in {0,1}, k2 in {0,1,2}
    
    U index: i*m + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    V index: j*p + k = (j1*2+j2)*9 + (k1*3+k2) = j1*18 + j2*9 + k1*3 + k2
    W index: k*n + i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
    
    For the Kronecker product of two 2x2x3 decompositions:
    First factor: n1=2, m1=2, p1=3
    Second factor: n2=2, m2=2, p2=3
    
    U1[i1*2+j1, t1] for i1,j1 in {0,1}
    U2[i2*2+j2, t2] for i2,j2 in {0,1}
    
    U[(i1*2+i2)*4 + (j1*2+j2), t1*7+t2] = U1[i1*2+j1, t1] * U2[i2*2+j2, t2]
    
    Wait, that's not right. The Kronecker product structure means:
    U_full[i,j] = U1[i1,j1] * U2[i2,j2] where i=i1*n2+i2, j=j1*m2+j2
    
    So U_full[(i1*n2+i2)*m + (j1*m2+j2), t1*r2+t2] = U1[i1*m1+j1, t1] * U2[i2*m2+j2, t2]
    
    Similarly for V and W.
    """
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    total_rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    for t1 in range(r1):
        for t2 in range(r2):
            t = t1 * r2 + t2
            for i1 in range(n1):
                for j1 in range(m1):
                    for k1 in range(p1):
                        u1 = U1[i1 * m1 + j1][t1]
                        v1 = V1[j1 * p1 + k1][t1]
                        w1 = W1[k1 * n1 + i1][t1]
                        for i2 in range(n2):
                            for j2 in range(m2):
                                for k2 in range(p2):
                                    u2 = U2[i2 * m2 + j2][t2]
                                    v2 = V2[j2 * p2 + k2][t2]
                                    w2 = W2[k2 * n2 + i2][t2]
                                    
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    
                                    U[i * m + j][t] = u1 * u2
                                    V[j * p + k][t] = v1 * v2
                                    W[k * n + i][t] = w1 * w2
    
    return U, V, W


def _construct_4x4x9():
    """Construct rank-121 decomposition for 4x4x9 using Kronecker product of two 2x2x3."""
    start_time = time.time()
    
    # Get 2x2x3 decomposition (rank 11)
    U1, V1, W1 = _construct_2x2x3()
    
    # Kronecker product: (2x2x3) ⊗ (2x2x3) = 4x4x9
    U, V, W = _kron_decompose(U1, V1, W1, U1, V1, W1, 2, 2, 3, 2, 2, 3)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        raise TimeoutError("Kronecker construction took too long")
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook_full():
    """Fallback schoolbook for 4x4x9."""
    return _construct_schoolbook(4, 4, 9)


def construct():
    start_time = time.time()
    
    # Try the Kronecker product construction (rank 121)
    try:
        result = _construct_4x4x9()
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 10:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 144)
    U, V, W = _construct_schoolbook_full()
    return {"U": U, "V": V, "W": W}
