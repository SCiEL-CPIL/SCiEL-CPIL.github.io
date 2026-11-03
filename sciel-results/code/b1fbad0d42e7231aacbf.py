import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _strassen_decomp():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U: (i, j) where i in 0..1, j in 0..1 -> i*2+j
    V: (j, k) where j in 0..1, k in 0..1 -> j*2+k
    W: (k, i) where k in 0..1, i in 0..1 -> k*2+i
    """
    U = [[Fraction(0)] * 7 for _ in range(4)]
    V = [[Fraction(0)] * 7 for _ in range(4)]
    W = [[Fraction(0)] * 7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C11 += M1, C22 += M1?
    # C11 = M1 - M3 + M4 + M5
    # C12 = M3 + M5 - M6 - M7
    # C21 = M2 + M4 + M6 + M7
    # C22 = M1 - M2 + M3 - M4
    
    # W indices: W[k*2+i, r]
    # C11 (i=0, k=0) -> W[0, r]
    # C12 (i=0, k=1) -> W[1, r]
    # C21 (i=1, k=0) -> W[2, r]
    # C22 (i=1, k=1) -> W[3, r]
    
    # r=0 (M1)
    W[0][0] = 1 # C11
    W[3][0] = 1 # C22
    
    # r=1 (M2 = (A21+A22)B11)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C21
    W[3][1] = -1 # C22
    
    # r=2 (M3 = A11(B12-B22))
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = -1 # C11
    W[1][2] = 1 # C12
    W[3][2] = 1 # C22
    
    # r=3 (M4 = A22(B21-B11))
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C11
    W[2][3] = 1 # C21
    W[3][3] = -1 # C22
    
    # r=4 (M5 = (A11+A12)B22)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1 # C11
    W[1][4] = 1 # C12
    
    # r=5 (M6 = (A21-A11)(B11+B12))
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = -1 # C12
    W[2][5] = 1 # C21
    
    # r=6 (M7 = (A12-A22)(B21+B22))
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = -1 # C12
    W[2][6] = 1 # C21
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 construction
    # Split m=3 into 2 and 1.
    # A1: 2x2, A2: 2x1
    # B1: 2x6, B2: 1x6
    # C = A1 B1 + A2 B2
    
    # Part 1: A2 B2 (Rank 1)
    # A2 is 2x1, B2 is 1x6.
    # C_i,k = A2_i,1 * B2_1,k
    # U2 (2, 1), V2 (1, 1), W2 (6, 1)
    # U2[i, 0] = 1, V2[0, 0] = 1, W2[k, 0] = 1
    # But we need to map to global indices.
    
    # Part 2: A1 B1 (Rank 21)
    # A1 is 2x2, B1 is 2x6.
    # B1 is 2x6, which is 3 blocks of 2x2.
    # B1 = [B1_1, B1_2, B1_3]
    # C1 = [C1_1, C1_2, C1_3]
    # Each C1_j = A1 B1_j (M_{2,2,2})
    
    total_rank = 1 + 3 * 7
    U = [[Fraction(0)] * total_rank for _ in range(n * m)] # 6 x 22
    V = [[Fraction(0)] * total_rank for _ in range(m * p)] # 18 x 22
    W = [[Fraction(0)] * total_rank for _ in range(p * n)] # 12 x 22
    
    # Fill Part 1: A2 B2
    # A2 indices: i in 0..1, j=2 (since A is 2x3, A2 is col 2)
    # A2_i,2 -> U[i*3 + 2, r]
    # B2 indices: j=2, k in 0..5
    # B2_2,k -> V[2*6 + k, r]
    # C indices: i in 0..1, k in 0..5
    # C_i,k -> W[k*2 + i, r]
    
    # r=0 is the rank 1 term
    for i in range(2):
        U[i*3 + 2][0] = Fraction(1)
    for k in range(6):
        V[2*6 + k][0] = Fraction(1)
    for i in range(2):
        for k in range(6):
            W[k*2 + i][0] = Fraction(1)
            
    # Fill Part 2: A1 B1
    # A1 indices: i in 0..1, j in 0..1
    # A1_i,j -> U[i*3 + j, r]
    # B1 indices: j in 0..1, k in 0..5
    # B1_j,k -> V[j*6 + k, r]
    # C indices: i in 0..1, k in 0..5
    # C_i,k -> W[k*2 + i, r]
    
    U_str, V_str, W_str = _strassen_decomp()
    
    for block in range(3):
        # B1_j, block*2 + k'
        # k = block*2 + k'
        for r_str in range(7):
            r_global = 1 + block * 7 + r_str
            # U
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2 + j][r_str]
                    if val != 0:
                        U[i*3 + j][r_global] += val
                        
            # V
            for j in range(2):
                for k_prime in range(2):
                    k = block * 2 + k_prime
                    val = V_str[j*2 + k_prime][r_str]
                    if val != 0:
                        V[j*6 + k][r_global] += val
                        
            # W
            for i in range(2):
                for k_prime in range(2):
                    k = block * 2 + k_prime
                    val = W_str[k_prime*2 + i][r_str]
                    if val != 0:
                        W[k*2 + i][r_global] += val

    # Convert to float
    U_float = [[float(x) for x in row] for row in U]
    V_float = [[float(x) for x in row] for row in V]
    W_float = [[float(x) for x in row] for row in W]
    
    return {"U": U_float, "V": V_float, "W": W_float}
