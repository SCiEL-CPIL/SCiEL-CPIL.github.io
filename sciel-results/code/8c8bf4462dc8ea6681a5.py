import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 22

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add to matrices
    def add_term(i, j, k, r, u_val, v_val, w_val):
        U[i * m + j][r] += u_val
        V[j * p + k][r] += v_val
        W[k * n + i][r] += w_val

    # Block 1: A11 (2x2) * B111 (2x2)
    # A11: i in {0,1}, j in {0,1}
    # B111: j in {0,1}, k in {0,1}
    # C11: i in {0,1}, k in {0,1}
    
    # Strassen 2x2
    # M1 = (A00+A11)(B00+B11)
    r = 0
    add_term(0, 0, 0, r, 1, 1, 1) # A00, B00, C00? No.
    # The term is U[i,j,r] V[j,k,r] W[k,i,r].
    # For M1, we need U to pick A00 and A11.
    # U[0,0,0] = 1, U[1,1,0] = 1.
    # V[0,0,0] = 1, V[1,1,0] = 1.
    # W[0,0,0] = 1, W[1,1,0] = 1?
    # C00 gets M1. C11 gets M1.
    # So W[0,0,0] = 1 (for C00), W[1,1,0] = 1 (for C11).
    # But W index is k*n+i.
    # C00 is k=0, i=0 -> index 0.
    # C11 is k=1, i=1 -> index 1*2+1 = 3.
    
    # Let's do this systematically.
    
    # M1
    r = 0
    U[0*m+0][r] += 1 # A00
    U[1*m+1][r] += 1 # A11
    V[0*p+0][r] += 1 # B00
    V[1*p+1][r] += 1 # B11
    W[0*n+0][r] += 1 # C00
    W[1*n+1][r] += 1 # C11
    
    # M2 = (A10+A11)B00
    r = 1
    U[1*m+0][r] += 1 # A10
    U[1*m+1][r] += 1 # A11
    V[0*p+0][r] += 1 # B00
    W[0*n+1][r] += 1 # C10 (k=0, i=1)
    W[1*n+1][r] += -1 # C11 (k=1, i=1) ? No, C11 = M1 - M2 + ...
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So W for M2 should contribute to C10 and C11.
    # W[0*n+1][r] += 1
    # W[1*n+1][r] += -1
    
    # M3 = A00(B01-B11)
    r = 2
    U[0*m+0][r] += 1 # A00
    V[0*p+1][r] += 1 # B01
    V[1*p+1][r] += -1 # B11
    W[0*n+0][r] += 1 # C00? No, C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # So M3 contributes to C01 and C11.
    # W[1*n+0][r] += 1 # C01 (k=1, i=0)
    # W[1*n+1][r] += 1 # C11
    
    # M4 = A11(B10-B00)
    r = 3
    U[1*m+1][r] += 1 # A11
    V[1*p+0][r] += 1 # B10
    V[0*p+0][r] += -1 # B00
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # So M4 contributes to C00 and C10.
    # W[0*n+0][r] += 1
    # W[0*n+1][r] += 1
    
    # M5 = (A00+A01)B11
    r = 4
    U[0*m+0][r] += 1 # A00
    U[0*m+1][r] += 1 # A01
    V[1*p+1][r] += 1 # B11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # So M5 contributes to C00 and C01.
    # W[0*n+0][r] += -1
    # W[1*n+0][r] += 1
    
    # M6 = (A10-A00)(B00+B01)
    r = 5
    U[1*m+0][r] += 1 # A10
    U[0*m+0][r] += -1 # A00
    V[0*p+0][r] += 1 # B00
    V[0*p+1][r] += 1 # B01
    # C11 = M1 - M2 + M3 + M6
    # So M6 contributes to C11.
    # W[1*n+1][r] += 1
    
    # M7 = (A01-A11)(B10+B11)
    r = 6
    U[0*m+1][r] += 1 # A01
    U[1*m+1][r] += -1 # A11
    V[1*p+0][r] += 1 # B10
    V[1*p+1][r] += 1 # B11
    # C00 = M1 + M4 - M5 + M7
    # So M7 contributes to C00.
    # W[0*n+0][r] += 1

    # Block 2: A11 (2x2) * B112 (2x2)
    # B112: j in {0,1}, k in {2,3}
    # C12: i in {0,1}, k in {2,3}
    
    # Same Strassen structure, but k indices are 2,3.
    # M1'
    r = 7
    U[0*m+0][r] += 1
    U[1*m+1][r] += 1
    V[0*p+2][r] += 1 # B02
    V[1*p+3][r] += 1 # B13
    W[2*n+0][r] += 1 # C02
    W[3*n+1][r] += 1 # C13
    
    # M2'
    r = 8
    U[1*m+0][r] += 1
    U[1*m+1][r] += 1
    V[0*p+2][r] += 1
    W[2*n+1][r] += 1 # C12
    W[3*n+1][r] += -1 # C13
    
    # M3'
    r = 9
    U[0*m+0][r] += 1
    V[0*p+3][r] += 1 # B03
    V[1*p+3][r] += -1 # B13
    W[3*n+0][r] += 1 # C03
    W[3*n+1][r] += 1 # C13
    
    # M4'
    r = 10
    U[1*m+1][r] += 1
    V[1*p+2][r] += 1 # B12
    V[0*p+2][r] += -1 # B02
    W[2*n+0][r] += 1 # C02
    W[2*n+1][r] += 1 # C12
    
    # M5'
    r = 11
    U[0*m+0][r] += 1
    U[0*m+1][r] += 1
    V[1*p+3][r] += 1 # B13
    W[2*n+0][r] += -1 # C02
    W[3*n+0][r] += 1 # C03
    
    # M6'
    r = 12
    U[1*m+0][r] += 1
    U[0*m+0][r] += -1
    V[0*p+2][r] += 1 # B02
    V[0*p+3][r] += 1 # B03
    W[3*n+1][r] += 1 # C13
    
    # M7'
    r = 13
    U[0*m+1][r] += 1
    U[1*m+1][r] += -1
    V[1*p+2][r] += 1 # B12
    V[1*p+3][r] += 1 # B13
    W[2*n+0][r] += 1 # C02

    # Block 3: A12 (2x1) * B21 (1x4)
    # A12: i in {0,1}, j=2
    # B21: j=2, k in {0,1,2,3}
    # C2: i in {0,1}, k in {0,1,2,3}
    
    # Schoolbook for 2x1x4
    # C[i,k] = A[i,2] * B[2,k]
    # 8 multiplications.
    
    for k in range(4):
        # M for i=0
        r = 14 + 2*k
        U[0*m+2][r] += 1 # A02
        V[2*p+k][r] += 1 # B2k
        W[k*n+0][r] += 1 # C0k
        
        # M for i=1
        r = 15 + 2*k
        U[1*m+2][r] += 1 # A12
        V[2*p+k][r] += 1 # B2k
        W[k*n+1][r] += 1 # C1k

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
