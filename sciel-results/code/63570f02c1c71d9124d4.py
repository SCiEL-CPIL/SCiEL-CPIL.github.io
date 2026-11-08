"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a11
        [0, 0, 0, 0, 1, 0, 1],    # a12
        [0, 1, 0, 0, 0, 1, 0],    # a21
        [1, 1, 0, 1, 0, 0, -1],   # a22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b11
        [0, 0, 1, 0, 0, 1, 0],    # b12
        [1, 0, -1, 1, 0, 0, 1],   # b21
        [1, 0, 0, 0, 1, 0, 1],    # b22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # c12 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # c21 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],   # c22 (k=1,i=1)
    ]
    return U, V, W

def _schoolbook_3x3():
    """Return schoolbook decomposition for 3x3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27)
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    n = m = p = 3
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
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    """Compute tensor product of two decompositions.
    T1 for (n1,m1,p1) rank r1, T2 for (n2,m2,p2) rank r2.
    Result for (n1*n2, m1*m2, p1*p2) rank r1*r2.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 3, 3, 3
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1 * m1 + j1][c1]
                    if u1 == 0:
                        continue
                    for k1 in range(p1):
                        v1 = V1[j1 * p1 + k1][c1]
                        w1 = W1[k1 * n1 + i1][c1]
                        if v1 == 0 or w1 == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2 = U2[i2 * m2 + j2][c2]
                                if u2 == 0:
                                    continue
                                for k2 in range(p2):
                                    v2 = V2[j2 * p2 + k2][c2]
                                    w2 = W2[k2 * n2 + i2][c2]
                                    if v2 == 0 or w2 == 0:
                                        continue
                                    
                                    u_val = u1 * u2
                                    v_val = v1 * v2
                                    w_val = w1 * w2
                                    
                                    i_idx = i1 * n2 + i2
                                    j_idx = j1 * m2 + j2
                                    k_idx = k1 * p2 + k2
                                    
                                    U[i_idx * m + j_idx][c] = u_val
                                    V[j_idx * p + k_idx][c] = v_val
                                    W[k_idx * n + i_idx][c] = w_val
    
    return U, V, W

def construct():
    """Construct a rank-189 decomposition for 6x6x6 matrix multiplication
    using tensor product of Strassen (2x2x2, rank 7) and schoolbook (3x3x3, rank 27)."""
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _schoolbook_3x3()
    U, V, W = _tensor_product(U1, V1, W1, U2, V2, W2)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
