DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    def add(i, j, k, cu=1, cv=1, cw=1):
        nonlocal col
        U[i*m+j][col] += cu
        V[j*p+k][col] += cv
        W[k*n+i][col] += cw
        col += 1

    # Strassen 2x2 (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    add(0,0,0, 1, 1, 1)
    add(0,0,1, 1, 1, 1)
    add(1,1,0, 1, 1, 1)
    add(1,1,1, 1, 1, 1)
    
    # M2 = (A10+A11)B00
    add(1,0,0, 1, 1, 1)
    add(1,1,0, 1, 1, 1)
    
    # M3 = A00(B10-B11)
    add(0,0,0, 1, 1, 1)
    add(0,0,1, 1, 1, -1)
    
    # M4 = A11(B00-B01)
    add(1,1,0, 1, 1, 1)
    add(1,1,1, 1, 1, -1)
    
    # M5 = (A00+A01)B11
    add(0,0,1, 1, 1, 1)
    add(0,1,1, 1, 1, 1)
    
    # M6 = (A10-A00)(B00+B01)
    add(1,0,0, 1, 1, 1)
    add(1,0,1, 1, 1, 1)
    add(0,0,0, -1, 1, 1)
    add(0,0,1, -1, 1, 1)
    
    # M7 = (A01-A11)(B10+B11)
    add(0,1,0, 1, 1, 1)
    add(0,1,1, 1, 1, 1)
    add(1,1,0, -1, 1, 1)
    add(1,1,1, -1, 1, 1)
    
    # Border terms
    # C02 = A00 B02 + A01 B12 + A02 B22
    add(0,0,2, 1, 1, 1)
    add(0,1,2, 1, 1, 1)
    add(0,2,2, 1, 1, 1)
    
    # C12 = A10 B02 + A11 B12 + A12 B22
    add(1,0,2, 1, 1, 1)
    add(1,1,2, 1, 1, 1)
    add(1,2,2, 1, 1, 1)
    
    # C20 = A20 B00 + A21 B10 + A22 B20
    add(2,0,0, 1, 1, 1)
    add(2,1,0, 1, 1, 1)
    add(2,2,0, 1, 1, 1)
    
    # C21 = A20 B01 + A21 B11 + A22 B21
    add(2,0,1, 1, 1, 1)
    add(2,1,1, 1, 1, 1)
    add(2,2,1, 1, 1, 1)
    
    # C22 = A20 B02 + A21 B12 + A22 B22
    add(2,0,2, 1, 1, 1)
    add(2,1,2, 1, 1, 1)
    add(2,2,2, 1, 1, 1)
    
    # Corrections
    # C00 += A02 B20
    add(0,2,0, 1, 1, 1)
    # C01 += A02 B21
    add(0,2,1, 1, 1, 1)
    # C10 += A12 B20
    add(1,2,0, 1, 1, 1)
    # C11 += A12 B21
    add(1,2,1, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}
