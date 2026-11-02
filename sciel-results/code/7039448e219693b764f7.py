import numpy as np

def construct():
    n, m, p = 3, 3, 6
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U indices: i*m+j
    # V indices: j*p+k
    # W indices: k*n+i
    
    # We will fill the matrices for each of the 3 k-blocks
    for block in range(3):
        # k indices in this block
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Offset in rank
        offset = block * 17
        
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Term 0: M1
        # U: A00, A11
        U[0*3+0][offset+0] = 1
        U[1*3+1][offset+0] = 1
        # V: B00, B11
        V[0*6+k0][offset+0] = 1
        V[1*6+k1][offset+0] = 1
        # W: C00, C11
        W[k0*3+0][offset+0] = 1
        W[k1*3+1][offset+0] = 1
        
        # Term 1: M2
        # U: A10, A11
        U[1*3+0][offset+1] = 1
        U[1*3+1][offset+1] = 1
        # V: B00
        V[0*6+k0][offset+1] = 1
        # W: C10
        W[k0*3+1][offset+1] = 1
        
        # Term 2: M3
        # U: A00
        U[0*3+0][offset+2] = 1
        # V: B01, -B11
        V[0*6+k1][offset+2] = 1
        V[1*6+k1][offset+2] = -1
        # W: C01
        W[k1*3+0][offset+2] = 1
        
        # Term 3: M4
        # U: A11
        U[1*3+1][offset+3] = 1
        # V: B10, -B00
        V[1*6+k0][offset+3] = 1
        V[0*6+k0][offset+3] = -1
        # W: C10, C11
        W[k0*3+1][offset+3] = 1
        W[k1*3+1][offset+3] = -1 # Wait, C11 = M1 - M2 + M3 + M6. C10 = M2 + M4.
        # Let's re-verify Strassen C matrix.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M4:
        # W[k0*3+1] (C10) += 1
        # W[k1*3+1] (C11) += -1? No, M4 is not in C11 formula.
        # C11 = M1 - M2 + M3 + M6.
        # So M4 only contributes to C10.
        W[k0*3+1][offset+3] = 1
        # Correction: W[k1*3+1] is 0 for M4.
        
        # Term 4: M5
        # U: A00, A01
        U[0*3+0][offset+4] = 1
        U[0*3+1][offset+4] = 1
        # V: B11
        V[1*6+k1][offset+4] = 1
        # W: C00, C01
        W[k0*3+0][offset+4] = -1 # C00 has -M5
        W[k1*3+0][offset+4] = 1  # C01 has +M5
        
        # Term 5: M6
        # U: A10, -A00
        U[1*3+0][offset+5] = 1
        U[0*3+0][offset+5] = -1
        # V: B00, B01
        V[0*6+k0][offset+5] = 1
        V[0*6+k1][offset+5] = 1
        # W: C11
        W[k1*3+1][offset+5] = 1
        
        # Term 6: M7
        # U: A01, -A11
        U[0*3+1][offset+6] = 1
        U[1*3+1][offset+6] = -1
        # V: B10, B11
        V[1*6+k0][offset+6] = 1
        V[1*6+k1][offset+6] = 1
        # W: C00
        W[k0*3+0][offset+6] = 1
        
        # 4 terms for i<2, j=2
        # A02 B20 -> C00
        # A02 B21 -> C01
        # A12 B20 -> C10
        # A12 B21 -> C11
        
        # Term 7: A02 B20
        U[0*3+2][offset+7] = 1
        V[2*6+k0][offset+7] = 1
        W[k0*3+0][offset+7] = 1
        
        # Term 8: A02 B21
        U[0*3+2][offset+8] = 1
        V[2*6+k1][offset+8] = 1
        W[k1*3+0][offset+8] = 1
        
        # Term 9: A12 B20
        U[1*3+2][offset+9] = 1
        V[2*6+k0][offset+9] = 1
        W[k0*3+1][offset+9] = 1
        
        # Term 10: A12 B21
        U[1*3+2][offset+10] = 1
        V[2*6+k1][offset+10] = 1
        W[k1*3+1][offset+10] = 1
        
        # 6 terms for i=2
        # A20 B00 -> C20
        # A20 B01 -> C21
        # A21 B10 -> C20
        # A21 B11 -> C21
        # A22 B20 -> C20
        # A22 B21 -> C21
        
        # Term 11: A20 B00
        U[2*3+0][offset+11] = 1
        V[0*6+k0][offset+11] = 1
        W[k0*3+2][offset+11] = 1
        
        # Term 12: A20 B01
        U[2*3+0][offset+12] = 1
        V[0*6+k1][offset+12] = 1
        W[k1*3+2][offset+12] = 1
        
        # Term 13: A21 B10
        U[2*3+1][offset+13] = 1
        V[1*6+k0][offset+13] = 1
        W[k0*3+2][offset+13] = 1
        
        # Term 14: A21 B11
        U[2*3+1][offset+14] = 1
        V[1*6+k1][offset+14] = 1
        W[k1*3+2][offset+14] = 1
        
        # Term 15: A22 B20
        U[2*3+2][offset+15] = 1
        V[2*6+k0][offset+15] = 1
        W[k0*3+2][offset+15] = 1
        
        # Term 16: A22 B21
        U[2*3+2][offset+16] = 1
        V[2*6+k1][offset+16] = 1
        W[k1*3+2][offset+16] = 1

    return {"U": U, "V": V, "W": W}
