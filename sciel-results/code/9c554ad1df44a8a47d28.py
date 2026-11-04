import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, r, c, val):
        mat[r][c] = val

    # Block L (k=0, 1)
    # Outer Product Part 1 (4 terms)
    # 1. A00 B00 -> C00
    set_val(U, 0, 0, 1)
    set_val(V, 0, 0, 1)
    set_val(W, 0, 0, 1)
    
    # 2. A00 B01 -> C01
    set_val(U, 0, 1, 1)
    set_val(V, 1, 1, 1)
    set_val(W, 2, 1, 1)
    
    # 3. A10 B00 -> C10
    set_val(U, 3, 2, 1)
    set_val(V, 0, 2, 1)
    set_val(W, 1, 2, 1)
    
    # 4. A10 B01 -> C11
    set_val(U, 3, 3, 1)
    set_val(V, 1, 3, 1)
    set_val(W, 3, 3, 1)
    
    # Strassen Part 1 (7 terms)
    # M1 = (A01+A12)(B10+B21)
    set_val(U, 1, 4, 1)
    set_val(U, 5, 4, 1)
    set_val(V, 4, 4, 1)
    set_val(V, 9, 4, 1)
    set_val(W, 0, 4, 1)
    set_val(W, 3, 4, 1)
    
    # M2 = (A11+A12)B10
    set_val(U, 4, 5, 1)
    set_val(U, 5, 5, 1)
    set_val(V, 4, 5, 1)
    set_val(W, 1, 5, 1)
    set_val(W, 3, 5, -1)
    
    # M3 = A01(B11-B21)
    set_val(U, 1, 6, 1)
    set_val(V, 5, 6, 1)
    set_val(V, 9, 6, -1)
    set_val(W, 2, 6, 1)
    set_val(W, 3, 6, 1)
    
    # M4 = A12(B20-B10)
    set_val(U, 5, 7, 1)
    set_val(V, 8, 7, 1)
    set_val(V, 4, 7, -1)
    set_val(W, 0, 7, 1)
    set_val(W, 1, 7, 1)
    
    # M5 = (A01+A02)B21
    set_val(U, 1, 8, 1)
    set_val(U, 2, 8, 1)
    set_val(V, 9, 8, 1)
    set_val(W, 0, 8, -1)
    set_val(W, 2, 8, 1)
    
    # M6 = (A11-A01)(B10+B11)
    set_val(U, 4, 9, 1)
    set_val(U, 1, 9, -1)
    set_val(V, 4, 9, 1)
    set_val(V, 5, 9, 1)
    set_val(W, 3, 9, 1)
    
    # M7 = (A02-A12)(B20+B21)
    set_val(U, 2, 10, 1)
    set_val(U, 5, 10, -1)
    set_val(V, 8, 10, 1)
    set_val(V, 9, 10, 1)
    set_val(W, 0, 10, 1)

    # Block R (k=2, 3)
    # Outer Product Part 2 (4 terms)
    # 1. A00 B02 -> C02
    set_val(U, 0, 11, 1)
    set_val(V, 6, 11, 1)
    set_val(W, 4, 11, 1)
    
    # 2. A00 B03 -> C03
    set_val(U, 0, 12, 1)
    set_val(V, 7, 12, 1)
    set_val(W, 6, 12, 1)
    
    # 3. A10 B02 -> C12
    set_val(U, 3, 13, 1)
    set_val(V, 6, 13, 1)
    set_val(W, 5, 13, 1)
    
    # 4. A10 B03 -> C13
    set_val(U, 3, 14, 1)
    set_val(V, 7, 14, 1)
    set_val(W, 7, 14, 1)
    
    # Strassen Part 2 (7 terms)
    # M1' = (A01+A12)(B12+B23)
    set_val(U, 1, 15, 1)
    set_val(U, 5, 15, 1)
    set_val(V, 10, 15, 1)
    set_val(V, 15, 15, 1)
    set_val(W, 4, 15, 1)
    set_val(W, 7, 15, 1)
    
    # M2' = (A11+A12)B12
    set_val(U, 4, 16, 1)
    set_val(U, 5, 16, 1)
    set_val(V, 10, 16, 1)
    set_val(W, 5, 16, 1)
    set_val(W, 7, 16, -1)
    
    # M3' = A01(B13-B23)
    set_val(U, 1, 17, 1)
    set_val(V, 11, 17, 1)
    set_val(V, 15, 17, -1)
    set_val(W, 6, 17, 1)
    set_val(W, 7, 17, 1)
    
    # M4' = A12(B22-B12)
    set_val(U, 5, 18, 1)
    set_val(V, 14, 18, 1)
    set_val(V, 10, 18, -1)
    set_val(W, 4, 18, 1)
    set_val(W, 5, 18, 1)
    
    # M5' = (A01+A02)B23
    set_val(U, 1, 19, 1)
    set_val(U, 2, 19, 1)
    set_val(V, 15, 19, 1)
    set_val(W, 4, 19, -1)
    set_val(W, 6, 19, 1)
    
    # M6' = (A11-A01)(B12+B13)
    set_val(U, 4, 20, 1)
    set_val(U, 1, 20, -1)
    set_val(V, 10, 20, 1)
    set_val(V, 11, 20, 1)
    set_val(W, 7, 20, 1)
    
    # M7' = (A02-A12)(B22+B23)
    set_val(U, 2, 21, 1)
    set_val(U, 5, 21, -1)
    set_val(V, 14, 21, 1)
    set_val(V, 15, 21, 1)
    set_val(W, 4, 21, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
