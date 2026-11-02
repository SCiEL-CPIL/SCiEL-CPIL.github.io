import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # Local indices 0,1 for i, j, k
        # Global i: i_offset + 0, i_offset + 1
        # Global j: j_offset + 0, j_offset + 1
        # Global k: k_offset + 0, k_offset + 1
        
        # Term 1: m1 = (a+d)(e+h) -> C11, C22
        U[i_offset*m + j_offset][col] = 1
        U[i_offset*m + j_offset+1][col] = 1
        V[j_offset*p + k_offset][col] = 1
        V[j_offset*p + k_offset+1][col] = 1
        W[k_offset*n + i_offset][col] = 1
        W[(k_offset+1)*n + i_offset+1][col] = 1
        col += 1
        
        # Term 2: m2 = (c+d)e -> C21, C22
        U[i_offset*m + j_offset+1][col] = 1
        U[(i_offset+1)*m + j_offset+1][col] = 1
        V[j_offset*p + k_offset][col] = 1
        W[(k_offset+1)*n + i_offset+1][col] = -1
        W[k_offset*n + i_offset+1][col] = 1
        col += 1
        
        # Term 3: m3 = a(f-h) -> C12, C22
        U[i_offset*m + j_offset][col] = 1
        V[j_offset*p + k_offset+1][col] = 1
        V[j_offset*p + k_offset][col] = -1
        W[k_offset*n + i_offset+1][col] = 1
        W[(k_offset+1)*n + i_offset+1][col] = 1
        col += 1
        
        # Term 4: m4 = d(g-e) -> C11, C21
        U[(i_offset+1)*m + j_offset+1][col] = 1
        V[j_offset*p + k_offset+1][col] = 1
        V[j_offset*p + k_offset][col] = -1
        W[k_offset*n + i_offset][col] = 1
        W[(k_offset+1)*n + i_offset+1][col] = 1
        col += 1
        
        # Term 5: m5 = (a+b)h -> C11, C12
        U[i_offset*m + j_offset][col] = 1
        U[i_offset*m + j_offset+1][col] = 1
        V[j_offset*p + k_offset+1][col] = 1
        W[k_offset*n + i_offset][col] = -1
        W[k_offset*n + i_offset+1][col] = 1
        col += 1
        
        # Term 6: m6 = (c-a)(e+f) -> C22
        U[(i_offset+1)*m + j_offset+1][col] = 1
        U[i_offset*m + j_offset][col] = -1
        V[j_offset*p + k_offset][col] = 1
        V[j_offset*p + k_offset+1][col] = 1
        W[(k_offset+1)*n + i_offset+1][col] = 1
        col += 1
        
        # Term 7: m7 = (b-d)(g+h) -> C11
        U[i_offset*m + j_offset+1][col] = 1
        U[(i_offset+1)*m + j_offset+1][col] = -1
        V[j_offset*p + k_offset][col] = 1
        V[j_offset*p + k_offset+1][col] = 1
        W[k_offset*n + i_offset][col] = 1
        col += 1

    def add_2x2x1(i_offset, j_offset, k_offset):
        nonlocal col
        # A is 2x2 (i, j), B is 2x1 (j, k)
        # C is 2x1 (i, k)
        # C_i = A_i0 B_0k + A_i1 B_1k
        
        # Term 1: A_i0 B_0k
        U[i_offset*m + j_offset][col] = 1
        V[j_offset*p + k_offset][col] = 1
        W[k_offset*n + i_offset][col] = 1
        col += 1
        
        # Term 2: A_i1 B_1k
        U[i_offset*m + j_offset+1][col] = 1
        V[(j_offset+1)*p + k_offset][col] = 1
        W[k_offset*n + i_offset][col] = 1
        col += 1
        
        # Term 3: A_(i+1)0 B_0k
        U[(i_offset+1)*m + j_offset][col] = 1
        V[j_offset*p + k_offset][col] = 1
        W[k_offset*n + i_offset+1][col] = 1
        col += 1
        
        # Term 4: A_(i+1)1 B_1k
        U[(i_offset+1)*m + j_offset+1][col] = 1
        V[(j_offset+1)*p + k_offset][col] = 1
        W[k_offset*n + i_offset+1][col] = 1
        col += 1

    def add_2x1x5(i_offset, j_offset):
        nonlocal col
        # A is 2x1 (i, j), B is 1x5 (j, k)
        # C is 2x5 (i, k)
        # C_ik = A_ij B_jk
        
        # Term 1: Outer product
        for i in range(2):
            U[(i_offset+i)*m + j_offset][col] = 1
        for k in range(5):
            V[j_offset*p + k][col] = 1
        for i in range(2):
            for k in range(5):
                W[k*n + i_offset+i][col] = 1
        col += 1

    # Part 1: j in {0, 1}
    # Sub-part 1: k in {0, 1} -> Strassen
    add_strassen(0, 0, 0)
    # Sub-part 2: k in {2, 3} -> Strassen
    add_strassen(0, 0, 2)
    # Sub-part 3: k in {4} -> 2x2x1
    add_2x2x1(0, 0, 4)
    
    # Part 2: j in {2, 3}
    # Sub-part 1: k in {0, 1} -> Strassen
    add_strassen(0, 2, 0)
    # Sub-part 2: k in {2, 3} -> Strassen
    add_strassen(0, 2, 2)
    # Sub-part 3: k in {4} -> 2x2x1
    add_2x2x1(0, 2, 4)
    
    # Part 3: j in {4}
    # Sub-part: 2x1x5
    add_2x1x5(0, 4)
    
    return {"U": U, "V": V, "W": W}
