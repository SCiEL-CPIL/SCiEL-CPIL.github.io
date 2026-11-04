import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Part 1: A1 B1 (2x1x3)
    # A1 is col 0 of A. B1 is row 0 of B.
    # C[i,k] += A[i,0] B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: A2 B2 (2x2x3)
    # A2 is cols 1,2 of A. B2 is rows 1,2 of B.
    # Split B2 into B21 (cols 0,1) and B22 (col 2).
    
    # Part 2a: A2 B21 (2x2x2) - Strassen
    # Local indices i', j' in {0,1} map to global i, j+1.
    # Local k' in {0,1} map to global k.
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, c):
        # u_coeffs: dict {global_idx: coeff}
        # v_coeffs: dict {global_idx: coeff}
        # w_coeffs: dict {global_idx: coeff}
        for idx, coeff in u_coeffs.items():
            U[idx, c] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx, c] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx, c] += coeff
            
    # Strassen products
    # p1 = (a00+a11)(b00+b11)
    # a00: i=0, j=1 -> idx 0*3+1=1
    # a11: i=1, j=2 -> idx 1*3+2=5
    # b00: j=1, k=0 -> idx 1*3+0=3
    # b11: j=2, k=1 -> idx 2*3+1=7
    # c00: i=0, k=0 -> idx 0*2+0=0
    # c11: i=1, k=1 -> idx 1*2+1=3
    
    add_strassen_term({1:1, 5:1}, {3:1, 7:1}, {0:1, 3:1}, col)
    col += 1
    
    # p2 = (a10+a11)b00
    # a10: i=1, j=1 -> idx 4
    # a11: i=1, j=2 -> idx 5
    # b00: j=1, k=0 -> idx 3
    # c10: i=1, k=0 -> idx 2
    add_strassen_term({4:1, 5:1}, {3:1}, {2:1}, col)
    col += 1
    
    # p3 = a00(b01-b11)
    # a00: idx 1
    # b01: j=1, k=1 -> idx 4
    # b11: j=2, k=1 -> idx 7
    # c01: i=0, k=1 -> idx 1
    add_strassen_term({1:1}, {4:1, 7:-1}, {1:1}, col)
    col += 1
    
    # p4 = a11(b10-b00)
    # a11: idx 5
    # b10: j=2, k=0 -> idx 6
    # b00: j=1, k=0 -> idx 3
    # c00: idx 0, c10: idx 2
    add_strassen_term({5:1}, {6:1, 3:-1}, {0:1, 2:1}, col)
    col += 1
    
    # p5 = (a00+a01)b11
    # a00: idx 1
    # a01: i=0, j=2 -> idx 2
    # b11: idx 7
    # c00: idx 0, c01: idx 1
    add_strassen_term({1:1, 2:1}, {7:1}, {0:-1, 1:1}, col)
    col += 1
    
    # p6 = (a10-a00)(b00+b01)
    # a10: idx 4
    # a00: idx 1
    # b00: idx 3
    # b01: idx 4
    # c11: idx 3
    add_strassen_term({4:1, 1:-1}, {3:1, 4:1}, {3:1}, col)
    col += 1
    
    # p7 = (a01-a11)(b10+b11)
    # a01: idx 2
    # a11: idx 5
    # b10: idx 6
    # b11: idx 7
    # c00: idx 0, c11: idx 3
    add_strassen_term({2:1, 5:-1}, {6:1, 7:1}, {0:1, 3:1}, col)
    col += 1
    
    # Part 2b: A2 B22 (2x2x1)
    # B22 is col 2 of B2. Global k=2.
    # A2 is cols 1,2.
    # C[i,2] += A[i,1]B[1,2] + A[i,2]B[2,2]
    
    # c02: i=0, k=2 -> idx 2*2+0=4
    # c12: i=1, k=2 -> idx 2*2+1=5
    
    # Term 1: A[0,1] B[1,2]
    U[1, col] = 1
    V[1*3+2, col] = 1 # V[5]
    W[4, col] = 1
    col += 1
    
    # Term 2: A[0,2] B[2,2]
    U[2, col] = 1
    V[2*3+2, col] = 1 # V[8]
    W[4, col] = 1
    col += 1
    
    # Term 3: A[1,1] B[1,2]
    U[4, col] = 1
    V[5, col] = 1
    W[5, col] = 1
    col += 1
    
    # Term 4: A[1,2] B[2,2]
    U[5, col] = 1
    V[8, col] = 1
    W[5, col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
