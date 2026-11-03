import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U: (6, rank), V: (24, rank), W: (16, rank)
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u, v, w):
        nonlocal col
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w
        col += 1

    # A indices: i in {0,1}, j in {0,1,2}. idx = i*m + j
    # B indices: j in {0,1,2}, k in {0..7}. idx = j*p + k
    # C indices: i in {0,1}, k in {0..7}. idx = k*n + i
    
    for b in range(2):
        k_base = 4 * b
        
        # Part 1: A1 * B1 (Rank 8)
        # A1 is col 0 of A. B1 is row 0 of B.
        for i in range(2):
            for k in range(4):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                u[i * m + 0] = 1
                v[0 * p + k_base + k] = 1
                w[(k_base + k) * n + i] = 1
                
                add_term(u, v, w)
                
        # Part 2: A2 * B2a (Rank 7)
        # A2 is cols 1,2 of A. B2a is rows 1,2 of B, cols 0,1 of block.
        # a=A01, b=A02, c=A11, d=A12
        # e=B10, f=B11, g=B20, h=B21
        # Indices in A: 01->1, 02->2, 11->4, 12->5
        # Indices in B: 10->8+k_b, 11->8+k_b+1, 20->16+k_b, 21->16+k_b+1
        # Indices in C: 00->2(k_b)+0, 01->2(k_b+1)+0, 10->2(k_b)+1, 11->2(k_b+1)+1
        
        def get_A(idx):
            vec = np.zeros(n * m)
            vec[idx] = 1
            return vec
            
        def get_B(idx):
            vec = np.zeros(m * p)
            vec[idx] = 1
            return vec
            
        def get_C(idx):
            vec = np.zeros(p * n)
            vec[idx] = 1
            return vec
            
        A01 = 1 # i=0, j=1
        A02 = 2 # i=0, j=2
        A11 = 4 # i=1, j=1
        A12 = 5 # i=1, j=2
        
        B10 = 8 + k_base
        B11 = 8 + k_base + 1
        B20 = 16 + k_base
        B21 = 16 + k_base + 1
        
        C00 = 2 * k_base + 0
        C01 = 2 * (k_base + 1) + 0
        C10 = 2 * k_base + 1
        C11 = 2 * (k_base + 1) + 1
        
        # M1 = (A01+A12)(B10+B20)
        u = get_A(A01) + get_A(A12)
        v = get_B(B10) + get_B(B20)
        w = get_C(C00) + get_C(C11)
        add_term(u, v, w)
        
        # M2 = (A11+A12)B10
        u = get_A(A11) + get_A(A12)
        v = get_B(B10)
        w = -get_C(C11) + get_C(C10)
        add_term(u, v, w)
        
        # M3 = A01(B11-B20)
        u = get_A(A01)
        v = get_B(B11) - get_B(B20)
        w = get_C(C01) + get_C(C11)
        add_term(u, v, w)
        
        # M4 = A12(B20-B10)
        u = get_A(A12)
        v = get_B(B20) - get_B(B10)
        w = get_C(C00) + get_C(C10)
        add_term(u, v, w)
        
        # M5 = (A01+A02)B20
        u = get_A(A01) + get_A(A02)
        v = get_B(B20)
        w = -get_C(C00) + get_C(C01)
        add_term(u, v, w)
        
        # M6 = (A11-A01)(B10+B11)
        u = get_A(A11) - get_A(A01)
        v = get_B(B10) + get_B(B11)
        w = get_C(C11)
        add_term(u, v, w)
        
        # M7 = (A02-A12)(B20+B21)
        u = get_A(A02) - get_A(A12)
        v = get_B(B20) + get_B(B21)
        w = get_C(C00)
        add_term(u, v, w)
        
        # Part 3: A2 * B2b (Rank 7)
        # B2b is rows 1,2 of B, cols 2,3 of block.
        # e=B12, f=B13, g=B22, h=B23
        # C00->C02, C01->C03, C10->C12, C11->C13
        
        B12 = 8 + k_base + 2
        B13 = 8 + k_base + 3
        B22 = 16 + k_base + 2
        B23 = 16 + k_base + 3
        
        C02 = 2 * (k_base + 2) + 0
        C03 = 2 * (k_base + 3) + 0
        C12 = 2 * (k_base + 2) + 1
        C13 = 2 * (k_base + 3) + 1
        
        # M1 = (A01+A12)(B12+B22)
        u = get_A(A01) + get_A(A12)
        v = get_B(B12) + get_B(B22)
        w = get_C(C02) + get_C(C13)
        add_term(u, v, w)
        
        # M2 = (A11+A12)B12
        u = get_A(A11) + get_A(A12)
        v = get_B(B12)
        w = -get_C(C13) + get_C(C12)
        add_term(u, v, w)
        
        # M3 = A01(B13-B22)
        u = get_A(A01)
        v = get_B(B13) - get_B(B22)
        w = get_C(C03) + get_C(C13)
        add_term(u, v, w)
        
        # M4 = A12(B22-B12)
        u = get_A(A12)
        v = get_B(B22) - get_B(B12)
        w = get_C(C02) + get_C(C12)
        add_term(u, v, w)
        
        # M5 = (A01+A02)B22
        u = get_A(A01) + get_A(A02)
        v = get_B(B22)
        w = -get_C(C02) + get_C(C03)
        add_term(u, v, w)
        
        # M6 = (A11-A01)(B12+B13)
        u = get_A(A11) - get_A(A01)
        v = get_B(B12) + get_B(B13)
        w = get_C(C13)
        add_term(u, v, w)
        
        # M7 = (A02-A12)(B22+B23)
        u = get_A(A02) - get_A(A12)
        v = get_B(B22) + get_B(B23)
        w = get_C(C02)
        add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
