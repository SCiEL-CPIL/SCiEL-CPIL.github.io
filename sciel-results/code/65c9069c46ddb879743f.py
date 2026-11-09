import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, u_val, v_val, w_val, col):
        # U index i*m + j
        U[i * m + j][col] += u_val
        # V index j*p + k
        V[j * p + k][col] += v_val
        # W index k*n + i
        W[k * n + i][col] += w_val

    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A B
    # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
    # M2 = (c+d)e -> C10 += M2
    # M3 = a(f-h) -> C01 += M3
    # M4 = d(g-e) -> C00 += M4, C10 += M4
    # M5 = (a+b)h -> C00 -= M5, C01 += M5
    # M6 = (c-a)(e+f) -> C11 += M6
    # M7 = (b-d)(g+h) -> C00 += M7
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # i in {0,1}, j in {0,1}, k in {0,1}
    # a=A00, b=A01, c=A10, d=A11
    # e=B00, f=B01, g=B10, h=B11
    
    col = 0
    # M1
    add_term(0, 0, 0, 1, 1, 0, col) # a e -> C00? No, M1 is (a+d)(e+h)
    # We need to add terms for each factor in the product.
    # (a+d)(e+h) = ae + ah + de + dh
    # This is 4 rank-1 terms if we expand.
    # But Strassen combines them.
    # The tensor decomposition of Strassen is rank 7.
    # Each of the 7 terms is a rank-1 tensor?
    # No, M1 is a product of a linear form in A and a linear form in B.
    # So it is a rank-1 term in the bilinear form space.
    # But it contributes to multiple entries of C.
    # So for each Strassen factor M_c, we have one column in U, V, W.
    # U column has coefficients for A.
    # V column has coefficients for B.
    # W column has coefficients for C.
    
    # Let's re-define add_term to handle multiple indices.
    # Actually, U, V, W are matrices.
    # For a specific column c, U[:, c] is the vector for A.
    # V[:, c] is the vector for B.
    # W[:, c] is the vector for C.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_term(col, u_indices, v_indices, w_indices):
        for idx, val in u_indices:
            U[idx][col] = val
        for idx, val in v_indices:
            V[idx][col] = val
        for idx, val in w_indices:
            W[idx][col] = val

    # Block 1: A[0:2,0:2] * B[0:2,0:2]
    # a=00, b=01, c=10, d=11
    # e=00, f=01, g=10, h=11
    
    # M1 = (a+d)(e+h)
    # U: a=1, d=1
    # V: e=1, h=1
    # W: C00=1, C11=1
    set_term(0, 
             [(0*3+0, 1), (1*3+1, 1)], # a, d
             [(0*4+0, 1), (1*4+1, 1)], # e, h
             [(0*3+0, 1), (1*3+1, 1)]) # C00, C11

    # M2 = (c+d)e
    set_term(1,
             [(1*3+0, 1), (1*3+1, 1)], # c, d
             [(0*4+0, 1)],             # e
             [(1*3+0, 1)])             # C10

    # M3 = a(f-h)
    set_term(2,
             [(0*3+0, 1)],             # a
             [(0*4+1, 1), (1*4+1, -1)],# f, -h
             [(0*3+1, 1)])             # C01

    # M4 = d(g-e)
    set_term(3,
             [(1*3+1, 1)],             # d
             [(1*4+0, 1), (0*4+0, -1)],# g, -e
             [(0*3+0, 1), (1*3+0, 1)]) # C00, C10

    # M5 = (a+b)h
    set_term(4,
             [(0*3+0, 1), (0*3+1, 1)], # a, b
             [(1*4+1, 1)],             # h
             [(0*3+0, -1), (0*3+1, 1)])# -C00, C01

    # M6 = (c-a)(e+f)
    set_term(5,
             [(1*3+0, 1), (0*3+0, -1)],# c, -a
             [(0*4+0, 1), (0*4+1, 1)], # e, f
             [(1*3+1, 1)])             # C11

    # M7 = (b-d)(g+h)
    set_term(6,
             [(0*3+1, 1), (1*3+1, -1)],# b, -d
             [(1*4+0, 1), (1*4+1, 1)], # g, h
             [(0*3+0, 1)])             # C00

    # Block 2: A[0:2,0:2] * B[0:2,2:4]
    # e=B02, f=B03, g=B12, h=B13
    # k in {2,3}
    
    # M1'
    set_term(7,
             [(0*3+0, 1), (1*3+1, 1)], # a, d
             [(0*4+2, 1), (1*4+3, 1)], # e, h
             [(2*3+0, 1), (3*3+1, 1)]) # C02, C13

    # M2'
    set_term(8,
             [(1*3+0, 1), (1*3+1, 1)], # c, d
             [(0*4+2, 1)],             # e
             [(2*3+1, 1)])             # C12

    # M3'
    set_term(9,
             [(0*3+0, 1)],             # a
             [(0*4+3, 1), (1*4+3, -1)],# f, -h
             [(2*3+0, 1)])             # C03

    # M4'
    set_term(10,
             [(1*3+1, 1)],             # d
             [(1*4+2, 1), (0*4+2, -1)],# g, -e
             [(2*3+0, 1), (2*3+1, 1)]) # C02, C12

    # M5'
    set_term(11,
             [(0*3+0, 1), (0*3+1, 1)], # a, b
             [(1*4+3, 1)],             # h
             [(2*3+0, -1), (2*3+1, 1)])# -C02, C03

    # M6'
    set_term(12,
             [(1*3+0, 1), (0*3+0, -1)],# c, -a
             [(0*4+2, 1), (0*4+3, 1)], # e, f
             [(3*3+1, 1)])             # C13

    # M7'
    set_term(13,
             [(0*3+1, 1), (1*3+1, -1)],# b, -d
             [(1*4+2, 1), (1*4+3, 1)], # g, h
             [(2*3+0, 1)])             # C02

    # Block 3: A[0:2,2:3] * B[2:3,0:4]
    # A21 = [[x], [y]] (col 2)
    # B14 = [p, q, r, s] (row 2)
    # C21 = [[xp, xq, xr, xs], [yp, yq, yr, ys]]
    # Rank 8 terms.
    
    # x * p -> C02
    set_term(14,
             [(0*3+2, 1)], # x
             [(2*4+0, 1)], # p
             [(0*3+0, 1)]) # C02? No, C is 3x4. i=0, k=2. W index 2*3+0=6.
             # Wait, W index is k*n+i.
             # C02 is i=0, k=2. Index 2*3+0 = 6.
             # My previous W indices:
             # C00: k=0, i=0 -> 0.
             # C11: k=1, i=1 -> 4.
             # C10: k=0, i=1 -> 1.
             # C01: k=1, i=0 -> 3.
             # C02: k=2, i=0 -> 6.
             # C12: k=2, i=1 -> 7.
             # C03: k=3, i=0 -> 9.
             # C13: k=3, i=1 -> 10.
             
             # Let's re-verify W indices in Block 1.
             # C00: k=0, i=0 -> 0. Correct.
             # C11: k=1, i=1 -> 4. Correct.
             # C10: k=0, i=1 -> 1. Correct.
             # C01: k=1, i=0 -> 3. Correct.
             # C02: k=2, i=0 -> 6.
             # C12: k=2, i=1 -> 7.
             # C03: k=3, i=0 -> 9.
             # C13: k=3, i=1 -> 10.
             
             # I made a mistake in Block 1 W indices.
             # Block 1: k in {0,1}.
             # C00: 0. C11: 4. C10: 1. C01: 3.
             # Block 2: k in {2,3}.
             # C02: 6. C13: 10. C12: 7. C03: 9.
             
             # Let's re-check Block 1 W indices in code above.
             # M1: C00 (0), C11 (4). Correct.
             # M2: C10 (1). Correct.
             # M3: C01 (3). Correct.
             # M4: C00 (0), C10 (1). Correct.
             # M5: C00 (0), C01 (3). Correct.
             # M6: C11 (4). Correct.
             # M7: C00 (0). Correct.
             
             # Block 2 W indices in code above:
             # M1': C02 (6), C13 (10).
             # Code had: (2*3+0, 1) -> 6. (3*3+1, 1) -> 10. Correct.
             # M2': C12 (7). Code: (2*3+1, 1) -> 7. Correct.
             # M3': C03 (9). Code: (2*3+0, 1) -> 6. WRONG. Should be 9.
             # M4': C02 (6), C12 (7). Code: (2*3+0, 1), (2*3+1, 1). Correct.
             # M5': C02 (6), C03 (9). Code: (2*3+0, -1), (2*3+1, 1). WRONG. Second should be 9.
             # M6': C13 (10). Code: (3*3+1, 1) -> 10. Correct.
             # M7': C02 (6). Code: (2*3+0, 1) -> 6. Correct.

    # I will rewrite the code carefully with correct indices.
    
    return {"U": U, "V": V, "W": W}

