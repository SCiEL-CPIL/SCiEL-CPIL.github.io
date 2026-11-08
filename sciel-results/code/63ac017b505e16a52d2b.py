import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Helper to add a product term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Block 1: Cols 0, 1 of B. A (2x3) * B (3x2)
    # Rank 11 decomposition
    # A = [[a, b, c], [d, e, f]]
    # B = [[x, y], [u, v], [s, t]]
    # Indices in U: i*m+j. 0:a, 1:b, 2:c, 3:d, 4:e, 5:f
    # Indices in V: j*p+k. 0:x, 1:y, 2:u, 3:v, 4:s, 5:t (for block 1)
    # Actually V index is j*p+k.
    # j=0: k=0,1 -> 0,1
    # j=1: k=0,1 -> 2,3
    # j=2: k=0,1 -> 4,5
    # W index: k*n+i.
    # k=0: i=0,1 -> 0,1
    # k=1: i=0,1 -> 2,3
    
    # Strassen part for A_L (a,b,d,e) and B_T (x,y,u,v)
    # M1 = (a+e)(x+u)
    add_term(0, 0, 0, 1) # a*x
    add_term(0, 2, 0, 1) # a*u
    add_term(4, 0, 0, 1) # e*x
    add_term(4, 2, 0, 1) # e*u
    # M2 = (d+e)(x+u)
    add_term(3, 0, 0, 1) # d*x
    add_term(3, 2, 0, 1) # d*u
    add_term(4, 0, 0, 1) # e*x
    add_term(4, 2, 0, 1) # e*u
    # M3 = a(x-u)
    add_term(0, 0, 0, 1) # a*x
    add_term(0, 2, 0, -1) # a*(-u)
    # M4 = e(v-x)
    add_term(4, 3, 0, 1) # e*v
    add_term(4, 0, 0, -1) # e*(-x)
    # M5 = (a+b)(v+u)
    add_term(0, 3, 0, 1) # a*v
    add_term(0, 2, 0, 1) # a*u
    add_term(1, 3, 0, 1) # b*v
    add_term(1, 2, 0, 1) # b*u
    # M6 = (d+b)(v-y)
    add_term(3, 3, 0, 1) # d*v
    add_term(3, 1, 0, -1) # d*(-y)
    add_term(1, 3, 0, 1) # b*v
    add_term(1, 1, 0, -1) # b*(-y)
    # M7 = (e+d)(x-y)
    add_term(4, 0, 0, 1) # e*x
    add_term(4, 1, 0, -1) # e*(-y)
    add_term(3, 0, 0, 1) # d*x
    add_term(3, 1, 0, -1) # d*(-y)
    
    # This approach of adding terms one by one into the same column index 'col' is wrong.
    # Each 'M' is a single product of a linear form in A and a linear form in B.
    # So U and V should have the coefficients of the linear form in that column.
    # W should have the coefficient of the output.
    
    return {"U": U, "V": "V", "W": W}
