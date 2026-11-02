"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for C = A B where A is 2x2 and B is 2x10 can be viewed as
    5 independent 2x2x2 multiplication problems, one for each pair of columns in B.
    
    The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen's algorithm).
    Since the 5 blocks are independent, the total rank is 5 * 7 = 35.
    
    Interface:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices:
    U index: i*m + j (i in 0..1, j in 0..1) -> 0,1,2,3
    V index: j*p + k (j in 0..1, k in 0..9) -> 0..19
    W index: k*n + i (k in 0..9, i in 0..1) -> 0..19 (Official trace convention)
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen's 7 products for multiplying 2x2 matrices M1 and M2:
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Result M = M1 * M2:
    # M[0,0] = P1 + P4 - P5 + P7
    # M[0,1] = P3 + P5
    # M[1,0] = P2 + P4
    # M[1,1] = P1 - P2 + P3 + P6
    
    # A coefficients for each Strassen product [a, b, c, d]
    strassen_A = [
        [1, 0, 0, 1],   # P1: (a+d)
        [0, 0, 1, 1],   # P2: (c+d)
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: (a+b)
        [-1, 0, 1, 0],  # P6: (c-a)
        [0, 1, 0, -1]   # P7: (b-d)
    ]
    
    # B coefficients for each Strassen product [e, f, g, h]
    strassen_B = [
        [1, 0, 0, 1],   # P1: (e+h)
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: (f-h)
        [-1, 0, 1, 0],  # P4: (g-e)
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: (e+f)
        [0, 0, 1, 1]    # P7: (g+h)
    ]
    
    # W coefficients: contribution of each P_r to [C00, C01, C10, C11]
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_contrib = [
        [1, 0, 0, 1],   # P1: C00+1, C11+1
        [0, 0, 1, -1],  # P2: C10+1, C11-1
        [0, 1, 0, 1],   # P3: C01+1, C11+1
        [1, 0, 1, 0],   # P4: C00+1, C10+1
        [-1, 1, 0, 0],  # P5: C00-1, C01+1
        [0, 0, 0, 1],   # P6: C11+1
        [1, 0, 0, 0]    # P7: C00+1
    ]
    
    # Fill U, V, W for each of the 5 blocks and 7 Strassen products
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U: index i*m + j
            U[0][col] = strassen_A[r][0]  # a: i=0, j=0
            U[1][col] = strassen_A[r][1]  # b: i=0, j=1
            U[2][col] = strassen_A[r][2]  # c: i=1, j=0
            U[3][col] = strassen_A[r][3]  # d: i=1, j=1
            
            # Fill V: index j*p + k
            # For block b, local columns are 2*b and 2*b+1
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]  # e: j=0, k=k0
            V[0 * p + k1][col] = strassen_B[r][1]  # f: j=0, k=k1
            V[1 * p + k0][col] = strassen_B[r][2]  # g: j=1, k=k0
            V[1 * p + k1][col] = strassen_B[r][3]  # h: j=1, k=k1
            
            # Fill W: index k_global * n + i
            # C00: i=0, k_local=0 -> k_global=2*b -> idx (2*b)*2 + 0
            # C01: i=0, k_local=1 -> k_global=2*b+1 -> idx (2*b+1)*2 + 0
            # C10: i=1, k_local=0 -> k_global=2*b -> idx (2*b)*2 + 1
            # C11: i=1, k_local=1 -> k_global=2*b+1 -> idx (2*b+1)*2 + 1
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]    # C00
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]  # C01
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]    # C10
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]  # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
