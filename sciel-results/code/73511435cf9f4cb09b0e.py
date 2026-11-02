import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Part 1: Rank 1 term for A_left * B_top
        # A indices: i in {0,1}, j=0 -> 0, 3
        # B indices: j=0, k in {k0, k1} -> k0, k1
        # C indices: k in {k0, k1}, i in {0,1} -> 2k0, 2k0+1, 2k1, 2k1+1
        
        U[0][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        V[k1][col] = 1
        W[2*k0][col] = 1
        W[2*k0+1][col] = 1
        W[2*k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # Part 2: Rank 7 terms for A_right * B_bot (Strassen)
        # A indices: i in {0,1}, j in {1,2}
        # a00=1, a01=2, a10=4, a11=5
        # B indices: j in {1,2}, k in {k0, k1}
        # b00=10+k0, b01=10+k1, b10=20+k0, b11=20+k1
        # C indices: k in {k0, k1}, i in {0,1}
        # c00=2k0, c01=2k1, c10=2k0+1, c11=2k1+1
        
        a00, a01, a10, a11 = 1, 2, 4, 5
        b00, b01, b10, b11 = 10+k0, 10+k1, 20+k0, 20+k1
        c00, c01, c10, c11 = 2*k0, 2*k1, 2*k0+1, 2*k1+1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = -1; W[c10][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
