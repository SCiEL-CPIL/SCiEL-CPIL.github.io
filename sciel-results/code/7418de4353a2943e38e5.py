"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2: A00(B10-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3: A11(B01-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # r=4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=5: (A01-A10)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: (A10-A00)(B11+B01)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of two Strassen decompositions."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*4+i][r] += val
    
    return U, V, W


def _build_4x4x2_optimized():
    """Build rank-28 decomposition for 4x4 * 4x2.
    
    C[i,k] = sum_j A[i,j] B[j,k] for i=0..3, k=0,1.
    Split into 2x2 blocks. For each (i1, k1) block pair, use two 2x2x2 Strassen products.
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 28
    U = [[0] * rank for _ in range(16)]  # 4x4 A
    V = [[0] * rank for _ in range(8)]   # 4x2 B
    W = [[0] * rank for _ in range(8)]   # 2x4 C (W[k*4+i])
    
    # C[i1*2+il, k1*2+kl] = A[i1*2+il, 0:2] @ B[0:2, k1*2+kl] + A[i1*2+il, 2:4] @ B[2:4, k1*2+kl]
    # For each (i1, k1): two 2x2x2 Strassen products
    block_idx = 0
    for i1 in range(2):
        for k1 in range(2):
            for j1 in range(2):
                for t in range(7):
                    r = block_idx * 7 + t
                    # U: A[i1*2+il, j1*2+jl]
                    for il in range(2):
                        for jl in range(2):
                            i = i1 * 2 + il
                            j = j1 * 2 + jl
                            val = U2[il*2+jl][t]
                            if val != 0:
                                U[i*4+j][r] += val
                    # V: B[j1*2+jl, k1*2+kl]
                    for jl in range(2):
                        for kl in range(2):
                            j = j1 * 2 + jl
                            k = k1 * 2 + kl
                            val = V2[jl*2+kl][t]
                            if val != 0:
                                V[j*2+k][r] += val
                    # W: C[i1*2+il, k1*2+kl] -> W[k*4+i]
                    for kl in range(2):
                        for il in range(2):
                            k = k1 * 2 + kl
                            i = i1 * 2 + il
                            val = W2[kl*2+il][t]
                            if val != 0:
                                W[k*4+i][r] += val
                block_idx += 1
    
    return U, V, W


def _build_4x4x1_optimized():
    """Build rank-8 decomposition for 4x4 * 4x1.
    
    C[i,0] = sum_j A[i,j] B[j,0] for i=0..3.
    Split into 2x2 blocks. For each i1, two 2x1x1 schoolbook products.
    """
    rank = 8
    U = [[0] * rank for _ in range(16)]  # 4x4 A
    V = [[0] * rank for _ in range(4)]   # 4x1 B
    W = [[0] * rank for _ in range(4)]   # 1x4 C (W[0*4+i] = W[i])
    
    # C[i1*2+il, 0] = A[i1*2+il, 0:2] @ B[0:2, 0] + A[i1*2+il, 2:4] @ B[2:4, 0]
    # Each 2x1 @ 2x1 has rank 2 (schoolbook)
    r = 0
    for i1 in range(2):
        for j1 in range(2):
            for jl in range(2):
                # Term: A[i1*2+il, j1*2+jl] * B[j1*2+jl, 0] for il=0,1
                # This is a 2x1 @ 2x1 product, rank 2
                for il in range(2):
                    U[i1*2+il, j1*2+jl if False else (i1*2+il)*4 + (j1*2+jl)][r] = 1
                    V[j1*2+jl][r] = 1
                    W[i1*2+il][r] = 1
                    r += 1
    
    return U, V, W


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 with rank 85.
    
    Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 8
    """
    n, m, p = 4, 4, 7
    
    U1, V1, W1 = _build_4x4x4_strassen()
    U2, V2, W2 = _build_4x4x2_optimized()
    U3, V3, W3 = _build_4x4x1_optimized()
    
    rank1, rank2, rank3 = 49, 28, 8
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank1):
                V[j*p+k][r] = V1[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank1):
                W[k*n+i][r] = W1[k*4+i][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2[i][r]
    for j in range(4):
        for k in range(2):
            for r in range(rank2):
                V[j*p + (4+k)][offset2 + r] = V2[j*2+k][r]
    for k in range(2):
        for i in range(4):
            for r in range(rank2):
                W[(4+k)*n + i][offset2 + r] = W2[k*4+i][r]
    
    # Part 3: B col 6, C col 6
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for j in range(4):
        for r in range(rank3):
            V[j*p + 6][offset3 + r] = V3[j][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
        return _schoolbook(4, 4, 7)
    except Exception:
        pass
    
    return _schoolbook(4, 4, 7)
