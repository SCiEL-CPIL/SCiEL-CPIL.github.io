import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(r, a_idx, b_idx, c_idx, val=1):
        U[a_idx, r] += val
        V[b_idx, r] += val
        W[c_idx, r] += val

    # Index mappings:
    # A[i, j] -> i*m + j
    # B[j, k] -> j*p + k
    # C[i, k] -> k*n + i  (Trace convention W[k*n+i])
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # C indices: 0(0,0), 2(0,1), 1(1,0), 3(1,1)
    # Note: C[0,1] is k=1, i=0 -> 1*2+0=2. C[1,0] is k=0, i=1 -> 0*2+1=1.
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    add_term(0, 0, 0, 0)
    add_term(0, 4, 0, 0)
    add_term(0, 0, 4, 0)
    add_term(0, 4, 4, 0)
    add_term(0, 0, 0, 3)
    add_term(0, 4, 0, 3)
    add_term(0, 0, 4, 3)
    add_term(0, 4, 4, 3)
    # Wait, Strassen uses linear combinations. The rank-1 terms are not just products of individual elements, 
    # but products of sums. My add_term function adds to specific indices.
    # For M1, U should have 1 at A00 and A11. V should have 1 at B00 and B11.
    # W should have 1 at C00 and C11.
    # So:
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Block 2: A_R (2x1) * B_BL (1x2) -> C_L (2x2)
    # A_R: A02 (idx 2), A12 (idx 5)
    # B_BL: B20 (idx 6), B21 (idx 7)
    # C_L: C00 (0), C01 (2), C10 (1), C11 (3)
    
    # r=7: A02 * B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 * B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 * B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 * B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Block 3: A_L (2x2) * B_TR (2x1) -> C_R (2x1)
    # A_L: A00 (0), A01 (1), A10 (3), A11 (4)
    # B_TR: B02 (2), B12 (5)
    # C_R: C02 (4), C12 (5)
    
    # r=11: A00 * B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 * B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A10 * B02 -> C12
    U[3, 13] = 1
    V[2, 13] = 1
    W[5, 13] = 1
    
    # r=14: A11 * B12 -> C12
    U[4, 14] = 1
    V[5, 14] = 1
    W[5, 14] = 1

    # Block 4: A_R (2x1) * B_BR (1x1) -> C_R (2x1)
    # A_R: A02 (2), A12 (5)
    # B_BR: B22 (8)
    # C_R: C02 (4), C12 (5)
    
    # r=15: A02 * B22 -> C02
    U[2, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1
    
    # r=16: A12 * B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
