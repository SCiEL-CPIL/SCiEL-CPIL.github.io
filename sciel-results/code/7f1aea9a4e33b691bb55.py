import time

DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = None


def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)


def _rank35():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen block for A[0:2, 0:2] x B[0:2, 0:2]
    # Local indices i, j, k in 0..1
    # Global indices:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # M1: (A00+A11)(B00+B11)
    # u: A00=1, A11=1
    # v: B00=1, B11=1
    # w: C00=1, C01=1, C10=1, C11=-1 ? 
    # Let's use the standard Strassen w factors.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Slot 0: M1
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*4+0][0] = 1; V[1*4+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+0][0] = 1; W[0*3+1][0] = 1; W[1*3+1][0] = -1
    
    # Slot 1: M2
    # (A10+A11)B00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*4+0][1] = 1
    W[0*3+0][1] = 0; W[1*3+0][1] = 0; W[0*3+1][1] = 1; W[1*3+1][1] = -1
    
    # Slot 2: M3
    # A00(B01-B11)
    U[0*3+0][2] = 1
    V[0*4+1][2] = 1; V[1*4+1][2] = -1
    W[0*3+0][2] = 0; W[1*3+0][2] = 1; W[0*3+1][2] = 0; W[1*3+1][2] = 1
    
    # Slot 3: M4
    # A11(B10-B00)
    U[1*3+1][3] = 1
    V[1*4+0][3] = 1; V[0*4+0][3] = -1
    W[0*3+0][3] = 1; W[1*3+0][3] = 0; W[0*3+1][3] = 1; W[1*3+1][3] = 0
    
    # Slot 4: M5
    # (A00+A01)B11
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*4+1][4] = 1
    W[0*3+0][4] = -1; W[1*3+0][4] = 1; W[0*3+1][4] = 0; W[1*3+1][4] = 0
    
    # Slot 5: M6
    # (A10-A00)(B00+B01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*4+0][5] = 1; V[0*4+1][5] = 1
    W[0*3+0][5] = 0; W[1*3+0][5] = 0; W[0*3+1][5] = 0; W[1*3+1][5] = 1
    
    # Slot 6: M7
    # (A01-A11)(B10+B11)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*4+0][6] = 1; V[1*4+1][6] = 1
    W[0*3+0][6] = 1; W[1*3+0][6] = 0; W[0*3+1][6] = 0; W[1*3+1][6] = 0

    # Schoolbook for A[0:2, 2] x B[2, 0:2]
    # A02 B20 -> C00
    # A02 B21 -> C01
    # A12 B20 -> C10
    # A12 B21 -> C11
    
    # Slot 7
    U[0*3+2][7] = 1
    V[2*4+0][7] = 1
    W[0*3+0][7] = 1
    
    # Slot 8
    U[0*3+2][8] = 1
    V[2*4+1][8] = 1
    W[1*3+0][8] = 1
    
    # Slot 9
    U[1*3+2][9] = 1
    V[2*4+0][9] = 1
    W[0*3+1][9] = 1
    
    # Slot 10
    U[1*3+2][10] = 1
    V[2*4+1][10] = 1
    W[1*3+1][10] = 1

    # Schoolbook for the rest
    # C02, C03, C12, C13, C20, C21, C22, C23
    
    c = 11
    for i in range(3):
        for k in range(4):
            # Skip the 2x2 block handled above
            if i < 2 and k < 2:
                continue
            for j in range(3):
                U[i*3+j][c] = 1
                V[j*4+k][c] = 1
                W[k*3+i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    global _START
    _START = time.time()
    return _rank35()
