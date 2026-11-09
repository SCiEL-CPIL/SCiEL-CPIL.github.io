import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen terms
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Helper to add a term
    # i: row of A (1 or 2)
    # J: col of A (1..4)
    # K: col of C (1..8)
    # coeff: coefficient for W
    def add_term(i, J, K, coeff):
        nonlocal col
        U[(i-1)*m + (J-1)][col] = 1
        V[(J-1)*p + (K-1)][col] = 1
        W[(K-1)*n + (i-1)][col] = coeff
        col += 1
        
    # M1
    # A11 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q, 1)
    # A11 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q+4, 1)
    # A22 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q, 1)
    # A22 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q+4, 1)
            
    # M2
    # A21 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j, q, 1)
    # A22 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q, 1)
            
    # M3
    # A11 B12
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q+4, 1)
    # A11 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q+4, -1)
            
    # M4
    # A22 B21
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q, 1)
    # A22 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q, -1)
            
    # M5
    # A11 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q+4, 1)
    # A12 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j+2, q+4, 1)
            
    # M6
    # A21 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j, q, 1)
    # A21 B12
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j, q+4, 1)
    # A11 B11
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q, -1)
    # A11 B12
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j, q+4, -1)
            
    # M7
    # A12 B21
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j+2, q, 1)
    # A12 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(1, j+2, q+4, 1)
    # A22 B21
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q, -1)
    # A22 B22
    for j in range(1, 3):
        for q in range(1, 5):
            add_term(2, j+2, q+4, -1)
            
    return {"U": U, "V": V, "W": W}
