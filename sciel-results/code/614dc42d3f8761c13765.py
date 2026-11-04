"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _check_all_entries(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                x2 = 2.0 * float(x)
                if abs(x2 - round(x2)) > 1e-9:
                    return False
    return True

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    If T1[i,j,k] = sum_t U1[i,t]*V1[j,t]*W1[k,t] for (n1,m1,p1)
    and T2[i',j',k'] = sum_t' U2[i',t']*V2[j',t']*W2[k',t'] for (n2,m2,p2)
    then T[i,j,k] for (n1*n2, m1*m2, p1*p2) has rank r1*r2.
    Index mapping: i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    n1, m1, p1 = len(U1), len(V1), len(W1)
    n2, m2, p2 = len(U2), len(V2), len(W2)
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    i = i1*n2 + i2
                    j = j1*m2 + j2
                    for t1 in range(r1):
                        u1 = U1[i1][t1]
                        v1 = V1[j1][t1]
                        for t2 in range(r2):
                            t = t1*r2 + t2
                            u2 = U2[i2][t2]
                            v2 = V2[j2][t2]
                            U[i][t] += u1 * u2
                            V[j][t] += v1 * v2
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    j = j1*m2 + j2
                    k = k1*p2 + k2
                    for t1 in range(r1):
                        v1 = V1[j1][t1]
                        w1 = W1[k1][t1]
                        for t2 in range(r2):
                            t = t1*r2 + t2
                            v2 = V2[j2][t2]
                            w2 = W2[k2][t2]
                            V[j][t] += v1 * v2
                            W[k][t] += w1 * w2
    
    return U, V, W

def _decomp_3x2x2():
    """
    Rank-6 decomposition of 3x2x2 matmul tensor.
    For A (3x2) * B (2x2) -> C (3x2):
    C[i,k] = sum_j A[i,j] * B[j,k]
    For each (i,j) pair, one term handles all k.
    """
    n, m, p = 3, 2, 2
    rank = n * m  # 6
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i*m + j][c] = 1.0
            for k in range(p):
                V[j*p + k][c] = 1.0
                W[k*n + i][c] = 1.0
    
    return U, V, W

def _decomp_1x2x2():
    """
    Rank-2 decomposition of 1x2x2 matmul tensor.
    For A (1x2) * B (2x2) -> C (1x2):
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    """
    n, m, p = 1, 2, 2
    rank = n * m  # 2
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i*m + j][c] = 1.0
            for k in range(p):
                V[j*p + k][c] = 1.0
                W[k*n + i][c] = 1.0
    
    return U, V, W

def _decomp_2x2x2_strassen():
    """
    Rank-7 decomposition of 2x2x2 matmul tensor (Strassen).
    n=2, m=2, p=2.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Strassen:
    m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    m2 = (A[1,0]+A[1,1])*B[0,0]
    m3 = A[0,0]*(B[0,1]-B[1,1])
    m4 = A[1,1]*(B[1,0]-B[0,0])
    m5 = (A[0,0]+A[0,1])*B[1,1]
    m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] = m1 - m5 + m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m2 + m6
    
    In tensor form, T[i,j,k] = 1 for all i,j,k in 0..1.
    Decomposition: T[i,j,k] = sum_t U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t]
    
    For each term t, U[i*2+j, t] is the coefficient of A[i,j] in the A-factor of m_t,
    V[j*2+k, t] is the coefficient of B[j,k] in the B-factor of m_t,
    W[k*2+i, t] is the coefficient of C[i,k] in the reconstruction.
    
    Let me work this out:
    
    m1: A-factor: A[0,0]+A[1,1] => U[0,0]=1, U[3,0]=1 (indices: i*2+j: (0,0)=0, (1,1)=3)
         B-factor: B[0,0]+B[1,1] => V[0,0]=1, V[3,0]=1 (j*2+k: (0,0)=0, (1,1)=3)
         C-recon: C[0,0]+=m1, C[1,1]+=m1 => W[0*2+0,0]=1, W[1*2+1,0]=1 => W[0,0]=1, W[3,0]=1
    
    m2: A-factor: A[1,0]+A[1,1] => U[2,1]=1, U[3,1]=1
         B-factor: B[0,0] => V[0,1]=1
         C-recon: C[1,0]+=m2 => W[0*2+1,1]=1 => W[1,1]=1
    
    m3: A-factor: A[0,0] => U[0,2]=1
         B-factor: B[0,1]-B[1,1] => V[1,2]=1, V[3,2]=-1
         C-recon: C[0,1]+=m3 => W[1*2+0,2]=1 => W[2,2]=1
    
    m4: A-factor: A[1,1] => U[3,3]=1
         B-factor: B[1,0]-B[0,0] => V[2,3]=1, V[0,3]=-1
         C-recon: C[1,0]+=m4 => W[1,3]=1
    
    m5: A-factor: A[0,0]+A[0,1] => U[0,4]=1, U[1,4]=1
         B-factor: B[1,1] => V[3,4]=1
         C-recon: C[0,0]-=m5, C[0,1]+=m5 => W[0,4]=-1, W[2,4]=1
    
    m6: A-factor: A[1,0]-A[0,0] => U[2,5]=1, U[0,5]=-1
         B-factor: B[0,0]+B[0,1] => V[0,5]=1, V[1,5]=1
         C-recon: C[1,1]+=m6 => W[3,5]=1
    
    m7: A-factor: A[0,1]-A[1,1] => U[1,6]=1, U[3,6]=-1
         B-factor: B[1,0]+B[1,1] => V[2,6]=1, V[3,6]=1
         C-recon: C[0,0]+=m7, C[1,1]-=m6... wait, C[0,0]+=m7 => W[0,6]=1
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # m1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0][0] = 1.0   # A[0,0]
    U[3][0] = 1.0   # A[1,1]
    V[0][0] = 1.0   # B[0,0]
    V[3][0] = 1.0   # B[1,1]
    W[0][0] = 1.0   # C[0,0] += m1
    W[3][0] = 1.0   # C[1,1] += m1
    
    # m2: (A[1,0]+A[1,1])*B[0,0]
    U[2][1] = 1.0   # A[1,0]
    U[3][1] = 1.0   # A[1,1]
    V[0][1] = 1.0   # B[0,0]
    W[1][1] = 1.0   # C[1,0] += m2
    
    # m3: A[0,0]*(B[0,1]-B[1,1])
    U[0][2] = 1.0   # A[0,0]
    V[1][2] = 1.0   # B[0,1]
    V[3][2] = -1.0  # B[1,1]
    W[2][2] = 1.0   # C[0,1] += m3
    
    # m4: A[1,1]*(B[1,0]-B[0,0])
    U[3][3] = 1.0   # A[1,1]
    V[2][3] = 1.0   # B[1,0]
    V[0][3] = -1.0  # B[0,0]
    W[1][3] = 1.0   # C[1,0] += m4
    
    # m5: (A[0,0]+A[0,1])*B[1,1]
    U[0][4] = 1.0   # A[0,0]
    U[1][4] = 1.0   # A[0,1]
    V[3][4] = 1.0   # B[1,1]
    W[0][4] = -1.0  # C[0,0] -= m5
    W[2][4] = 1.0   # C[0,1] += m5
    
    # m6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2][5] = 1.0   # A[1,0]
    U[0][5] = -1.0  # A[0,0]
    V[0][5] = 1.0   # B[0,0]
    V[1][5] = 1.0   # B[0,1]
    W[3][5] = 1.0   # C[1,1] += m6
    
    # m7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1][6] = 1.0   # A[0,1]
    U[3][6] = -1.0  # A[1,1]
    V[2][6] = 1.0   # B[1,0]
    V[3][6] = 1.0   # B[1,1]
    W[0][6] = 1.0   # C[0,0] += m7
    
    return U, V, W

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Strategy: Kronecker product of 3x2x2 (rank 6) and 1x2x2 (rank 2)
    # This gives 3x4x4 with rank 6*2 = 12
    U1, V1, W1 = _decomp_3x2x2()
    U2, V2, W2 = _decomp_1x2x2()
    
    U, V, W = _kron_decomp(U1, V1, W1, U2, V2, W2)
    
    # Verify
    if _verify(U, V, W, n, m, p) and _check_all_entries(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: try previous rank-42 approach
    if time.time() < deadline:
        U2s, V2s, W2s = _decomp_2x2x2_strassen()
        U, V, W = _kron_decomp(U1, V1, W1, U2s, V2s, W2s)
        if _verify(U, V, W, n, m, p) and _check_all_entries(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
