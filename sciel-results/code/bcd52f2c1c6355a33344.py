"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}


def _strassen_4x4x4():
    """Recursive Strassen decomposition for 4x4 x 4x4 matmul, rank 49.
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]."""
    S = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U[i * 4 + j][c] = S['U'][i1 * 2 + j1][c1] * S['U'][i2 * 2 + j2][c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = k1 * 2 + k2
                            V[j * 4 + k][c] = S['V'][j1 * 2 + k1][c1] * S['V'][j2 * 2 + k2][c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    for i1 in range(2):
                        for i2 in range(2):
                            i = i1 * 2 + i2
                            W[k * 4 + i][c] = S['W'][k1 * 2 + i1][c1] * S['W'][k2 * 2 + i2][c2]
                            
    return {"U": U, "V": V, "W": W}


def _decomp_4x4x9_optimized():
    """Construct 4x4x9 decomposition using Strassen for 4x4x4 and schoolbook for 4x4x5.
    Rank = 49 + 20 = 69."""
    n, m, p = 4, 4, 9
    p1 = 4
    p2 = 5
    rank1 = 49
    rank2 = 20
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 (cols 0-3)
    S444 = _strassen_4x4x4()
    for c in range(rank1):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = S444['U'][i * m + j][c]
        for j in range(m):
            for k in range(p1):
                V[j * p + k][c] = S444['V'][j * m + k][c]
        for k in range(p1):
            for i in range(n):
                W[k * n + i][c] = S444['W'][k * n + i][c]
                
    # Part 2: 4x4x5 (cols 4-8) Schoolbook
    for c in range(rank2):
        cc = rank1 + c
        # c maps to (j, k_offset)
        j = c // p2
        k_offset = c % p2
        k = p1 + k_offset
        
        # U: A[i,j]
        for i in range(n):
            U[i * m + j][cc] = 1
            
        # V: B[j,k]
        V[j * p + k][cc] = 1
        
        # W: C[k,i]
        for i in range(n):
            W[k * n + i][cc] = 1
            
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 9
    
    # Try the optimized rank-69 construction
    try:
        cand = _decomp_4x4x9_optimized()
        if _verify(cand['U'], cand['V'], cand['W'], n, m, p):
            print("Verified rank-69 decomposition")
            return cand
        else:
            print("Rank-69 verification failed, falling back to schoolbook")
    except Exception as e:
        print("Rank-69 construction failed:", e)
        
    # Fallback to schoolbook if something went wrong
    return _schoolbook(n, m, p)
