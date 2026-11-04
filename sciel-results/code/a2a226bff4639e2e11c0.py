import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = w_val
        col += 1

    # Helper for Strassen block
    # A is 2x2 at (i0, j0) to (i1, j1)
    # B is 2x2 at (j0, k0) to (j1, k1)
    # C is 2x2 at (i0, k0) to (i1, k1)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        add_term(i0*m+j0, j0*p+k0, k0*n+i0, 1)
        add_term(i0*m+j0, j0*p+k0, k1*n+i1, 1)
        add_term(i1*m+j1, j1*p+k1, k0*n+i0, 1)
        add_term(i1*m+j1, j1*p+k1, k1*n+i1, 1)
        
        # M2 = (A10 + A11)B00
        # C10 += M2, C11 -= M2
        add_term(i1*m+j0, j0*p+k0, k0*n+i1, 1)
        add_term(i1*m+j1, j1*p+k0, k0*n+i1, 1)
        add_term(i1*m+j0, j0*p+k0, k1*n+i1, -1)
        add_term(i1*m+j1, j1*p+k0, k1*n+i1, -1)
        
        # M3 = A00(B01 - B11)
        # C01 += M3, C11 += M3
        add_term(i0*m+j0, j0*p+k1, k1*n+i0, 1)
        add_term(i0*m+j0, j1*p+k1, k1*n+i0, -1)
        add_term(i0*m+j0, j0*p+k1, k1*n+i1, 1)
        add_term(i0*m+j0, j1*p+k1, k1*n+i1, -1)
        
        # M4 = A11(B10 - B00)
        # C00 += M4, C10 += M4
        add_term(i1*m+j1, j1*p+k0, k0*n+i0, 1)
        add_term(i1*m+j1, j0*p+k0, k0*n+i0, -1)
        add_term(i1*m+j1, j1*p+k0, k0*n+i1, 1)
        add_term(i1*m+j1, j0*p+k0, k0*n+i1, -1)
        
        # M5 = (A00 + A01)B11
        # C00 -= M5, C01 += M5
        add_term(i0*m+j0, j1*p+k1, k0*n+i0, -1)
        add_term(i0*m+j1, j1*p+k1, k0*n+i0, -1)
        add_term(i0*m+j0, j1*p+k1, k1*n+i0, 1)
        add_term(i0*m+j1, j1*p+k1, k1*n+i0, 1)
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += M6
        add_term(i1*m+j0, j0*p+k0, k1*n+i1, 1)
        add_term(i1*m+j0, j0*p+k1, k1*n+i1, 1)
        add_term(i0*m+j0, j0*p+k0, k1*n+i1, -1)
        add_term(i0*m+j0, j0*p+k1, k1*n+i1, -1)
        
        # M7 = (A01 - A11)(B10 + B11)
        # C00 += M7
        add_term(i0*m+j1, j1*p+k0, k0*n+i0, 1)
        add_term(i0*m+j1, j1*p+k1, k0*n+i0, 1)
        add_term(i1*m+j1, j1*p+k0, k0*n+i0, -1)
        add_term(i1*m+j1, j1*p+k1, k0*n+i0, -1)

    # Helper for schoolbook block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term(i*m+j, j*p+k, k*n+i, 1)

    # Part 1: A1 (2x2) * B11 (2x2)
    # A1: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Part 2: A1 (2x2) * B12a (2x2)
    # B12a: j in {0,1}, k in {2,3}
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Part 3: A1 (2x2) * B12b (2x1)
    # B12b: j in {0,1}, k in {4}
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Part 4: A2 (2x1) * B2 (1x5)
    # A2: i in {0,1}, j in {2}
    # B2: j in {2}, k in {0,1,2,3,4}
    add_schoolbook(0, 2, 2, 3, 0, 5)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
