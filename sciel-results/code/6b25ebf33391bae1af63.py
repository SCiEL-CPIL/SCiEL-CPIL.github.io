import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    # We will build the decomposition manually
    
    # Helper to add a rank-1 term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for this term
    # We store terms as lists of (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))

    # 1. Decompose 2x2x2 using Strassen
    # A is 2x2, B is 2x2
    # Indices: A(i,j), B(j,k), C(i,k)
    # i in 0..1, j in 0..1, k in 0..1
    
    def add_strassen_2x2x2(i0, j0, k0):
        # i0, j0, k0 are offsets in the global matrices
        # A indices: (i0+i, j0+j)
        # B indices: (j0+j, k0+k)
        # C indices: (i0+i, k0+k)
        
        # M1 = (A00+A11)(B00+B11)
        add_term((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), 1)
        add_term((i0+1)*m + (j0+1), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), 1)
        add_term((i0+0)*m + (j0+0), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1)
        
        # M2 = (A10+A11)B00
        add_term((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1)
        add_term((i0+1)*m + (j0+1), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1)
        
        # M3 = A00(B10-B11)
        add_term((i0+0)*m + (j0+0), (j0+1)*p + (k0+0), (k0+0)*n + (i0+0), 1)
        add_term((i0+0)*m + (j0+0), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), -1)
        
        # M4 = A11(B01-B00)
        add_term((i0+1)*m + (j0+1), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), 1)
        add_term((i0+1)*m + (j0+1), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), -1)
        
        # M5 = (A00+A01)B11
        add_term((i0+0)*m + (j0+0), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1)
        add_term((i0+0)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1)
        
        # M6 = (A10-A00)(B00+B01)
        add_term((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1)
        add_term((i0+1)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), 1)
        add_term((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), -1)
        add_term((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), -1)
        
        # M7 = (A01-A11)(B10+B11)
        add_term((i0+0)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+0), 1)
        add_term((i0+0)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+0), -1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+0), -1)

    # 2. Decompose 2x2x1 (Standard)
    def add_std_2x2x1(i0, j0, k0):
        # C00 = A00 B00 + A01 B10
        add_term((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), 1)
        add_term((i0+0)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+0), 1)
        # C10 = A10 B00 + A11 B10
        add_term((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+1), 1)

    # 3. Decompose 1x2x7 (Standard)
    # A is 1x2, B is 2x7, C is 1x7
    def add_std_1x2x7(i0, j0, k0):
        # i0 is row offset (0), j0 is col offset (0..1), k0 is col offset (0..6)
        # C0k = A00 B0k + A01 B1k
        for k in range(7):
            add_term((i0+0)*m + (j0+0), (j0+0)*p + (k0+k), (k0+k)*n + (i0+0), 1)
            add_term((i0+0)*m + (j0+1), (j0+1)*p + (k0+k), (k0+k)*n + (i0+0), 1)

    # 4. Decompose 2x1x7 (Standard)
    # A is 2x1, B is 1x7, C is 2x7
    def add_std_2x1x7(i0, j0, k0):
        # i0 row offset, j0 col offset (0), k0 col offset
        for k in range(7):
            add_term((i0+0)*m + (j0+0), (j0+0)*p + (k0+k), (k0+k)*n + (i0+0), 1)
            add_term((i0+1)*m + (j0+0), (j0+0)*p + (k0+k), (k0+k)*n + (i0+1), 1)

    # 5. Decompose 1x3x7 (Standard)
    def add_std_1x3x7(i0, j0, k0):
        for k in range(7):
            for j in range(3):
                add_term((i0+0)*m + (j0+j), (j0+j)*p + (k0+k), (k0+k)*n + (i0+0), 1)

    # 6. Decompose 2x3x7
    # Split 3 into 2+1
    # A 2x3 = [A11 (2x2), A12 (2x1)]
    # B 3x7 = [B11 (2x7); B12 (1x7)]
    def add_2x3x7(i0, j0, k0):
        # A11 B11: 2x2x7
        # Split 7 into 2+2+2+1
        add_strassen_2x2x2(i0, j0, k0)
        add_strassen_2x2x2(i0, j0, k0+2)
        add_strassen_2x2x2(i0, j0, k0+4)
        add_std_2x2x1(i0, j0, k0+6)
        
        # A12 B12: 2x1x7
        add_std_2x1x7(i0, j0+2, k0)

    # 7. Decompose 2x2x7
    def add_2x2x7(i0, j0, k0):
        add_strassen_2x2x2(i0, j0, k0)
        add_strassen_2x2x2(i0, j0, k0+2)
        add_strassen_2x2x2(i0, j0, k0+4)
        add_std_2x2x1(i0, j0, k0+6)

    # Main construction
    # A is 3x5, B is 5x7
    # Split 5 into 2+3
    # A1 (3x2), A2 (3x3)
    # B1 (2x7), B2 (3x7)
    
    # A1 B1: 3x2x7
    # Split A1 into 2x2 and 1x2
    # A11 (2x2) B11 (2x7)
    add_2x2x7(0, 0, 0)
    # A12 (1x2) B12 (2x7)
    add_std_1x2x7(2, 0, 0)
    
    # A2 B2: 3x3x7
    # Split A2 into 2x3 and 1x3
    # A21 (2x3) B21 (3x7)
    add_2x3x7(0, 2, 0)
    # A22 (1x3) B22 (3x7)
    add_std_1x3x7(2, 2, 0)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][r] = coeff
        V[v_idx][r] = coeff
        W[w_idx][r] = coeff

    return {"U": U, "V": V, "W": W}
