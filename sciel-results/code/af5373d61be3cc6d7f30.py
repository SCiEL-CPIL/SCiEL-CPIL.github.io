import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """Return (U,V,W) for 2x2*2x2 -> 2x2 using Strassen (rank 7).
    U index: i*2+j ; V index: j*2+k ; W index: k*2+i (trace convention)
    So W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # a=U[0], b=U[1], c=U[2], d=U[3]
    # x=V[0], y=V[1], u=V[2], v=V[3]
    # C00=W[0], C10=W[1], C01=W[2], C11=W[3]
    
    # M1 = (a+d)(x+u): C00+=1, C11+=1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[2][0]=1; W[0][0]=1; W[3][0]=1
    # M2 = (c+d)x: C00-=1, C10+=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[0][1]=-1; W[1][1]=1
    # M3 = a(y-u): C00+=1, C01+=1
    U[0][2]=1; V[1][2]=1; V[2][2]=-1; W[0][2]=1; W[2][2]=1
    # M4 = d(v-u): C00-=1, C01+=1
    U[3][3]=1; V[2][3]=-1; V[3][3]=1; W[0][3]=-1; W[2][3]=1
    # M5 = (a+b)u: C10+=1, C11-=1
    U[0][4]=1; U[1][4]=1; V[2][4]=1; W[1][4]=1; W[3][4]=-1
    # M6 = (c+d)(v-y): C10+=1, C11+=1
    U[2][5]=1; U[3][5]=1; V[1][5]=-1; V[3][5]=1; W[1][5]=1; W[3][5]=1
    # M7 = (a+b)(y-v): C01+=1, C11+=1
    U[0][6]=1; U[1][6]=1; V[1][6]=1; V[3][6]=-1; W[2][6]=1; W[3][6]=1
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m * m x p -> n x p. Rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rng_seed=42):
    """Verify the decomposition with random integer matrices."""
    rng = np.random.default_rng(rng_seed)
    A = rng.integers(-3, 4, size=(n, m))
    B = rng.integers(-3, 4, size=(m, p))
    C_expected = A @ B
    rank = len(U[0])
    C_decomp = np.zeros((n, p), dtype=int)
    for t in range(rank):
        a_t = np.zeros(n)
        for i in range(n):
            a_t[i] = sum(U[i*m+j][t] * A[i,j] for j in range(m))
        v_t = np.zeros(p)
        for k in range(p):
            v_t[k] = sum(V[j*p+k][t] * B[j,k] for j in range(m))
        for i in range(n):
            for k in range(p):
                C_decomp[i,k] += a_t[i] * v_t[k] * W[k*n+i][t]
    return np.array_equal(C_expected, C_decomp)

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140.0

    # Split middle dimension 5 into 2+3.
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x6 -> split B1 into three 2x2 blocks, each Strassen (rank 7). Total: 21.
    # A2*B2: 2x3 * 3x6 -> schoolbook: 36.
    # Total: 57.

    U_parts = []
    V_parts = []
    W_parts = []

    # Part 1: A1*B1 (2x2 * 2x6), split into 3 blocks of 2x2*2x2
    for block in range(3):
        sU, sV, sW = _build_strassen_2x2x2()
        U_parts.append(sU)
        V_parts.append(sV)
        W_parts.append(sW)

    # Part 2: A2*B2 (2x3 * 3x6), schoolbook
    sU, sV, sW = _build_schoolbook(2, 3, 6)
    U_parts.append(sU)
    V_parts.append(sV)
    W_parts.append(sW)

    # Concatenate
    U_rows = n * m  # 10
    V_rows = m * p  # 30
    W_rows = p * n  # 12

    U = [[0] for _ in range(U_rows)]
    V = [[0] for _ in range(V_rows)]
    W = [[0] for _ in range(W_rows)]
    # Initialize with zeros of correct length
    total_rank = sum(len(p[0]) for p in U_parts)
    U = [[0]*total_rank for _ in range(U_rows)]
    V = [[0]*total_rank for _ in range(V_rows)]
    W = [[0]*total_rank for _ in range(W_rows)]

    offset = 0
    for partU, partV, partW in zip(U_parts, V_parts, W_parts):
        r = len(partU[0])
        for i in range(U_rows):
            for t in range(r):
                U[i][offset+t] = partU[i][t]
        for i in range(V_rows):
            for t in range(r):
                V[i][offset+t] = partV[i][t]
        for i in range(W_rows):
            for t in range(r):
                W[i][offset+t] = partW[i][t]
        offset += r

    # Verify
    if not _verify(U, V, W, n, m, p, rng_seed=42):
        U, V, W = _build_schoolbook(n, m, p)
    elif not _verify(U, V, W, n, m, p, rng_seed=123):
        U, V, W = _build_schoolbook(n, m, p)

    if time.time() > deadline - 5:
        U, V, W = _build_schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
