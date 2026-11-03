import numpy as np

def construct():
    n, m, p = 2, 3, 4
    
    # Dimensions of U, V, W
    # U: (n*m, rank) = (6, 22)
    # V: (m*p, rank) = (12, 22)
    # W: (p*n, rank) = (8, 22)
    
    rank = 22
    U = np.zeros((6, rank))
    V = np.zeros((12, rank))
    W = np.zeros((8, rank))
    
    # Indices:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A_sub = [[A00, A01], [A10, A11]]
    # B_sub = [[B00, B01], [B10, B11]]
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping to indices:
    # A00 -> U[0, r], B00 -> V[0, r], C00 -> W[0, r]
    # A01 -> U[1, r], B01 -> V[1, r], C01 -> W[1, r]
    # A10 -> U[2, r], B10 -> V[2, r], C10 -> W[2, r]
    # A11 -> U[3, r], B11 -> V[3, r], C11 -> W[3, r]
    
    r = 0
    # M1
    U[0, r] += 1; U[3, r] += 1
    V[0, r] += 1; V[3, r] += 1
    W[0, r] += 1
    r += 1
    # M2
    U[2, r] += 1; U[3, r] += 1
    V[0, r] += 1
    W[2, r] += 1
    r += 1
    # M3
    U[0, r] += 1
    V[3, r] += 1; V[1, r] -= 1
    W[1, r] += 1
    r += 1
    # M4
    U[3, r] += 1
    V[0, r] += 1; V[2, r] -= 1
    W[0, r] += 1
    r += 1
    # M5
    U[0, r] += 1; U[1, r] += 1
    V[3, r] += 1
    W[0, r] -= 1
    r += 1
    # M6
    U[2, r] += 1; U[0, r] -= 1
    V[0, r] += 1; V[1, r] += 1
    W[3, r] += 1
    r += 1
    # M7
    U[1, r] += 1; U[3, r] -= 1
    V[2, r] += 1; V[3, r] += 1
    W[0, r] += 1
    r += 1
    
    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # B_sub = [[B02, B03], [B12, B13]]
    # Indices:
    # B02 -> V[2, r] (j=0, k=2)
    # B03 -> V[3, r] (j=0, k=3)
    # B12 -> V[6, r] (j=1, k=2)
    # B13 -> V[7, r] (j=1, k=3)
    # C02 -> W[4, r] (k=2, i=0)
    # C03 -> W[5, r] (k=3, i=0)
    # C12 -> W[6, r] (k=2, i=1)
    # C13 -> W[7, r] (k=3, i=1)
    
    # M1
    U[0, r] += 1; U[3, r] += 1
    V[2, r] += 1; V[7, r] += 1
    W[4, r] += 1
    r += 1
    # M2
    U[2, r] += 1; U[3, r] += 1
    V[2, r] += 1
    W[6, r] += 1
    r += 1
    # M3
    U[0, r] += 1
    V[7, r] += 1; V[3, r] -= 1
    W[5, r] += 1
    r += 1
    # M4
    U[3, r] += 1
    V[2, r] += 1; V[6, r] -= 1
    W[4, r] += 1
    r += 1
    # M5
    U[0, r] += 1; U[1, r] += 1
    V[7, r] += 1
    W[4, r] -= 1
    r += 1
    # M6
    U[2, r] += 1; U[0, r] -= 1
    V[2, r] += 1; V[3, r] += 1
    W[7, r] += 1
    r += 1
    # M7
    U[1, r] += 1; U[3, r] -= 1
    V[6, r] += 1; V[7, r] += 1
    W[4, r] += 1
    r += 1
    
    # Part 3: Schoolbook for A[0:2, 2:3] * B[2:3, 0:4]
    # A_sub = [[A02], [A12]] -> U[4, r], U[5, r]
    # B_sub = [[B20, B21, B22, B23]] -> V[8, r], V[9, r], V[10, r], V[11, r]
    # C00 -> W[0, r], C01 -> W[1, r], C02 -> W[4, r], C03 -> W[5, r]
    # C10 -> W[2, r], C11 -> W[3, r], C12 -> W[6, r], C13 -> W[7, r]
    
    # A02 * B20
    U[4, r] += 1
    V[8, r] += 1
    W[0, r] += 1
    r += 1
    # A02 * B21
    U[4, r] += 1
    V[9, r] += 1
    W[1, r] += 1
    r += 1
    # A02 * B22
    U[4, r] += 1
    V[10, r] += 1
    W[4, r] += 1
    r += 1
    # A02 * B23
    U[4, r] += 1
    V[11, r] += 1
    W[5, r] += 1
    r += 1
    # A12 * B20
    U[5, r] += 1
    V[8, r] += 1
    W[2, r] += 1
    r += 1
    # A12 * B21
    U[5, r] += 1
    V[9, r] += 1
    W[3, r] += 1
    r += 1
    # A12 * B22
    U[5, r] += 1
    V[10, r] += 1
    W[6, r] += 1
    r += 1
    # A12 * B23
    U[5, r] += 1
    V[11, r] += 1
    W[7, r] += 1
    r += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
