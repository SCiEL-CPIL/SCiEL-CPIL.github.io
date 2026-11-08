import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    r = 0
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal r
        for idx, val in u_entries:
            U[idx][r] = val
        for idx, val in v_entries:
            V[idx][r] = val
        for idx, val in w_entries:
            W[idx][r] = val
        r += 1
    
    # Block 1: Column 2 of A times Row 2 of B (outer product)
    # A[:,2] * B[2,:] -> contributes to C[:,2]... wait no
    # A is 2x3, B is 3x5. C[i,k] = sum_j A[i,j]*B[j,k]
    # j=2 term: A[i,2]*B[2,k] for all i,k
    for i in range(n):
        for k in range(p):
            add_term([(i*m+2, 1)], [(2*p+k, 1)], [(k*n+i, 1)])
    
    # Block 2: Columns 0,1 of A times Row 0,1 of B, Column 4
    # j=0,1 terms for k=4: A[i,j]*B[j,4]
    for i in range(n):
        for j in range(2):
            add_term([(i*m+j, 1)], [(j*p+4, 1)], [(4*n+i, 1)])
    
    # Block 3: Columns 0,1 of A times Rows 0,1 of B, Columns 0,1 (Strassen)
    # This computes a 2x2 matrix product: A12 * B12 where A12 is 2x2, B12 is 2x2
    # Result contributes to C[:, 0:2]
    # Strassen's 7 multiplications for 2x2*2x2
    
    # Let A12 = [[a00, a01], [a10, a11]] (rows i=0,1, cols j=0,1)
    # Let B12 = [[b00, b01], [b10, b11]] (rows j=0,1, cols k=0,1)
    # Result C12 = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11)(b00+b11)
    add_term([(0*m+0, 1), (1*m+1, 1)], [(0*p+0, 1), (1*p+1, 1)], [(0*n+0, 1), (1*n+1, 1)])
    
    # M2 = (a10+a11)(b00)
    add_term([(1*m+0, 1), (1*m+1, 1)], [(0*p+0, 1)], [(0*n+1, 1), (1*n+1, -1)])
    
    # M3 = (a00)(b01-b11)
    add_term([(0*m+0, 1)], [(0*p+1, 1), (1*p+1, -1)], [(1*n+0, 1), (1*n+1, 1)])
    
    # M4 = (a11)(b10-b00)
    add_term([(1*m+1, 1)], [(1*p+0, 1), (0*p+0, -1)], [(0*n+0, 1), (0*n+1, 1)])
    
    # M5 = (a00+a01)(b11)
    add_term([(0*m+0, 1), (0*m+1, 1)], [(1*p+1, 1)], [(0*n+0, -1), (1*n+0, 1)])
    
    # M6 = (a10-a00)(b00+b01)
    add_term([(1*m+0, 1), (0*m+0, -1)], [(0*p+0, 1), (0*p+1, 1)], [(1*n+1, 1)])
    
    # M7 = (a01-a11)(b10+b11)
    add_term([(0*m+1, 1), (1*m+1, -1)], [(1*p+0, 1), (1*p+1, 1)], [(0*n+0, 1)])
    
    # Block 4: Columns 0,1 of A times Rows 0,1 of B, Columns 2,3 (Strassen)
    # A12 * B12' where B12' has cols k=2,3
    # B12' = [[b02, b03], [b12, b13]]
    
    # M1 = (a00+a11)(b02+b13)
    add_term([(0*m+0, 1), (1*m+1, 1)], [(0*p+2, 1), (1*p+3, 1)], [(2*n+0, 1), (3*n+1, 1)])
    
    # M2 = (a10+a11)(b02)
    add_term([(1*m+0, 1), (1*m+1, 1)], [(0*p+2, 1)], [(2*n+1, 1), (3*n+1, -1)])
    
    # M3 = (a00)(b03-b13)
    add_term([(0*m+0, 1)], [(0*p+3, 1), (1*p+3, -1)], [(3*n+0, 1), (3*n+1, 1)])
    
    # M4 = (a11)(b12-b02)
    add_term([(1*m+1, 1)], [(1*p+2, 1), (0*p+2, -1)], [(2*n+0, 1), (2*n+1, 1)])
    
    # M5 = (a00+a01)(b13)
    add_term([(0*m+0, 1), (0*m+1, 1)], [(1*p+3, 1)], [(2*n+0, -1), (3*n+0, 1)])
    
    # M6 = (a10-a00)(b02+b03)
    add_term([(1*m+0, 1), (0*m+0, -1)], [(0*p+2, 1), (0*p+3, 1)], [(3*n+1, 1)])
    
    # M7 = (a01-a11)(b12+b13)
    add_term([(0*m+1, 1), (1*m+1, -1)], [(1*p+2, 1), (1*p+3, 1)], [(2*n+0, 1)])
    
    return {"U": U, "V": V, "W": W}
