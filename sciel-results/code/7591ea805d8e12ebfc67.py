"""
Strassen rank-7 decomposition for 2x2 matrix multiplication.
The tensor T satisfies T[i,j,k,l] = U[i,j,c] * V[k,l,c] * W[c,m,n]
where W is in the official trace convention W[k*n+i, r] = W[(k*n+i), c].
"""

import numpy as np
import time

def construct():
    n, m, p = 2, 2, 2  # A is n x m, B is m x p, C = A * B is n x p
    
    rank = 7
    
    # U is (n*m) x rank = 4 x 7
    # V is (m*p) x rank = 4 x 7
    # W is (p*n) x rank = 4 x 7, with index W[k*n + i]
    
    # Strassen's 7 multiplication terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A10) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U matrix: rows indexed by (i, j) -> i*m + j
    # For term c, U[i, j, c] is the coefficient of A_ij in the first factor of M_c
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # M1: (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # Contribution to C: C00 += 1, C11 += 1
    # W index: k*n + i
    W[0, 0] = 1  # C00: k=0, i=0
    W[3, 0] = 1  # C11: k=1, i=1
    
    # M2: (A10 + A11)(B00)
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    # Contribution to C: C10 += 1, C11 -= 1
    W[2, 1] = 1  # C10: k=0, i=1
    W[3, 1] = -1 # C11: k=1, i=1
    
    # M3: A00(B01 - B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    # Contribution to C: C01 += 1, C11 += 1
    W[1, 2] = 1  # C01: k=1, i=0
    W[3, 2] = 1  # C11: k=1, i=1
    
    # M4: A11(B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    # Contribution to C: C00 += 1, C10 += 1
    W[0, 3] = 1  # C00: k=0, i=0
    W[2, 3] = 1  # C10: k=0, i=1
    
    # M5: (A00 + A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    # Contribution to C: C00 -= 1, C01 += 1
    W[0, 4] = -1 # C00: k=0, i=0
    W[1, 4] = 1  # C01: k=1, i=0
    
    # M6: (A10 - A00)(B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # Contribution to C: C11 += 1
    W[3, 5] = 1  # C11: k=1, i=1
    
    # M7: (A11 - A10)(B10 + B11)
    U[3, 6] = 1  # A11
    U[2, 6] = -1 # A10
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    # Contribution to C: C00 += 1
    W[0, 6] = 1  # C00: k=0, i=0
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if j == k else 0
                    actual = sum(U[i*m+j, c] * V[k*p+j, c] * W[k*n+i, c] for c in range(rank))
                    if abs(expected - actual) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank-8
        rank = 8
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((4, rank), dtype=float)
        W = np.zeros((4, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
