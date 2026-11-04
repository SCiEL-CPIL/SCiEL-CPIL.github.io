import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    # Schoolbook rank is 36. We aim for 34.
    
    # We will build the decomposition term by term.
    # U: (9, R), V: (12, R), W: (12, R)
    # We don't know R in advance, so we use lists of tuples or dynamic arrays.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict or list of 9 values for A (3x3)
        v_vals: dict or list of 12 values for B (3x4)
        w_vals: dict or list of 12 values for C (3x4)
        """
        # Convert to list of Fractions
        U_list = [Fraction(u_vals.get((i,j), 0)) for i in range(n) for j in range(m)]
        V_list = [Fraction(v_vals.get((j,k), 0)) for j in range(m) for k in range(p)]
        W_list = [Fraction(w_vals.get((k,i), 0)) for k in range(p) for i in range(n)]
        
        U_terms.append(U_list)
        V_terms.append(V_list)
        W_terms.append(W_list)

    # Block indices:
    # A: 0,1 (rows), 2 (row); 0,1 (cols), 2 (col)
    # B: 0,1 (rows), 2 (row); 0,1 (cols), 2,3 (cols)
    # C: 0,1 (rows), 2 (row); 0,1 (cols), 2,3 (cols)
    
    # 1. C11 = A11 B11 + A12 B21
    # A11 B11: 2x2x2 Strassen
    # Strassen 2x2x2 rank 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need to map these to the full 3x3 and 3x4 matrices.
    # A11 corresponds to A[0:2, 0:2]
    # B11 corresponds to B[0:2, 0:2]
    # C11 corresponds to C[0:2, 0:2]
    
    def get_A_sub(i1, i2, j1, j2):
        d = {}
        for i in range(i1, i2):
            for j in range(j1, j2):
                d[(i,j)] = 1
        return d
        
    def get_B_sub(i1, i2, j1, j2):
        d = {}
        for i in range(i1, i2):
            for j in range(j1, j2):
                d[(i,j)] = 1
        return d
        
    def get_C_sub(i1, i2, j1, j2):
        d = {}
        for i in range(i1, i2):
            for j in range(j1, j2):
                d[(i,j)] = 1
        return d

    # Strassen terms for A11 B11
    # A11 is A[0:2, 0:2], B11 is B[0:2, 0:2], C11 is C[0:2, 0:2]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # A: A00(0,0), A11(1,1)
    # B: B00(0,0), B11(1,1)
    # C: C00(0,0), C11(1,1)
    
    # We will add 7 terms for A11 B11
    # Term 1: M1
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (1,1): 1}
    w = {(0,0): 1, (1,1): 1}
    add_term(u, v, w)
    
    # Term 2: M2 = (A10+A11)B00 -> C10
    u = {(1,0): 1, (1,1): 1}
    v = {(0,0): 1}
    w = {(1,0): 1}
    add_term(u, v, w)
    
    # Term 3: M3 = A00(B10-B11) -> C01, C11
    u = {(0,0): 1}
    v = {(1,0): 1, (1,1): -1}
    w = {(0,1): 1, (1,1): 1}
    add_term(u, v, w)
    
    # Term 4: M4 = A11(B01-B10) -> C00, C10
    u = {(1,1): 1}
    v = {(0,1): 1, (1,0): -1}
    w = {(0,0): 1, (1,0): 1}
    add_term(u, v, w)
    
    # Term 5: M5 = (A00+A10)B11 -> C00, C01
    u = {(0,0): 1, (1,0): 1}
    v = {(1,1): 1}
    w = {(0,0): -1, (0,1): 1}
    add_term(u, v, w)
    
    # Term 6: M6 = (A01-A00)(B00+B10) -> C11
    u = {(0,1): 1, (0,0): -1}
    v = {(0,0): 1, (1,0): 1}
    w = {(1,1): 1}
    add_term(u, v, w)
    
    # Term 7: M7 = (A10-A11)(B01+B11) -> C00
    u = {(1,0): 1, (1,1): -1}
    v = {(0,1): 1, (1,1): 1}
    w = {(0,0): 1}
    add_term(u, v, w)
    
    # A12 B21: 2x1x2 (Rank 4)
    # A12 is A[0:2, 2], B21 is B[2, 0:2], C12 is C[0:2, 0:2]? No.
    # C11 = A11 B11 + A12 B21.
    # A12 is 2x1 (rows 0,1; col 2).
    # B21 is 1x2 (row 2; cols 0,1).
    # Product is 2x2 (rows 0,1; cols 0,1). This adds to C11.
    # Terms:
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    
    add_term({(0,2): 1}, {(2,0): 1}, {(0,0): 1})
    add_term({(0,2): 1}, {(2,1): 1}, {(0,1): 1})
    add_term({(1,2): 1}, {(2,0): 1}, {(1,0): 1})
    add_term({(1,2): 1}, {(2,1): 1}, {(1,1): 1})
    
    # 2. C12 = A11 B12 + A12 B22
    # A11 B12: 2x2x2 (Rank 7)
    # A11 is A[0:2, 0:2], B12 is B[0:2, 2:4], C12 is C[0:2, 2:4]
    
    # M1: (A00+A11)(B02+B13)
    u = {(0,0): 1, (1,1): 1}
    v = {(0,2): 1, (1,3): 1}
    w = {(2,0): 1, (3,1): 1}
    add_term(u, v, w)
    
    # M2: (A10+A11)B02
    u = {(1,0): 1, (1,1): 1}
    v = {(0,2): 1}
    w = {(2,1): 1}
    add_term(u, v, w)
    
    # M3: A00(B12-B13)
    u = {(0,0): 1}
    v = {(1,2): 1, (1,3): -1}
    w = {(3,0): 1, (3,1): 1}
    add_term(u, v, w)
    
    # M4: A11(B03-B12)
    u = {(1,1): 1}
    v = {(0,3): 1, (1,2): -1}
    w = {(2,0): 1, (2,1): 1}
    add_term(u, v, w)
    
    # M5: (A00+A10)B13
    u = {(0,0): 1, (1,0): 1}
    v = {(1,3): 1}
    w = {(2,0): -1, (3,0): 1}
    add_term(u, v, w)
    
    # M6: (A01-A00)(B02+B12)
    u = {(0,1): 1, (0,0): -1}
    v = {(0,2): 1, (1,2): 1}
    w = {(3,1): 1}
    add_term(u, v, w)
    
    # M7: (A10-A11)(B03+B13)
    u = {(1,0): 1, (1,1): -1}
    v = {(0,3): 1, (1,3): 1}
    w = {(2,0): 1}
    add_term(u, v, w)
    
    # A12 B22: 2x1x2 (Rank 4)
    # A12 is A[0:2, 2], B22 is B[2, 2:4], C12 is C[0:2, 2:4]
    add_term({(0,2): 1}, {(2,2): 1}, {(2,0): 1})
    add_term({(0,2): 1}, {(2,3): 1}, {(3,0): 1})
    add_term({(1,2): 1}, {(2,2): 1}, {(2,1): 1})
    add_term({(1,2): 1}, {(2,3): 1}, {(3,1): 1})
    
    # 3. C21 = A21 B11 + A22 B21
    # A21 B11: 1x2x2 (Rank 4)
    # A21 is A[2, 0:2], B11 is B[0:2, 0:2], C21 is C[2, 0:2]
    add_term({(2,0): 1}, {(0,0): 1}, {(0,2): 1})
    add_term({(2,0): 1}, {(0,1): 1}, {(1,2): 1})
    add_term({(2,1): 1}, {(1,0): 1}, {(0,2): 1})
    add_term({(2,1): 1}, {(1,1): 1}, {(1,2): 1})
    
    # A22 B21: 1x1x2 (Rank 2)
    # A22 is A[2,2], B21 is B[2, 0:2], C21 is C[2, 0:2]
    add_term({(2,2): 1}, {(2,0): 1}, {(0,2): 1})
    add_term({(2,2): 1}, {(2,1): 1}, {(1,2): 1})
    
    # 4. C22 = A21 B12 + A22 B22
    # A21 B12: 1x2x2 (Rank 4)
    # A21 is A[2, 0:2], B12 is B[0:2, 2:4], C22 is C[2, 2:4]
    add_term({(2,0): 1}, {(0,2): 1}, {(2,2): 1})
    add_term({(2,0): 1}, {(0,3): 1}, {(3,2): 1})
    add_term({(2,1): 1}, {(1,2): 1}, {(2,2): 1})
    add_term({(2,1): 1}, {(1,3): 1}, {(3,2): 1})
    
    # A22 B22: 1x1x2 (Rank 2)
    # A22 is A[2,2], B22 is B[2, 2:4], C22 is C[2, 2:4]
    add_term({(2,2): 1}, {(2,2): 1}, {(2,2): 1})
    add_term({(2,2): 1}, {(2,3): 1}, {(3,2): 1})
    
    # Total rank: 7 + 4 + 7 + 4 + 4 + 2 + 4 + 2 = 34
    
    rank = len(U_terms)
    
    # Convert to numpy arrays
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    # Verify
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    if verify(U, V, W, rank):
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
    else:
        # Fallback to schoolbook
        U_s = np.zeros((n*m, n*m*p), dtype=object)
        V_s = np.zeros((m*p, n*m*p), dtype=object)
        W_s = np.zeros((p*n, n*m*p), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {"U": [[float(x) for x in row] for row in U_s], 
                "V": [[float(x) for x in row] for row in V_s], 
                "W": [[float(x) for x in row] for row in W_s]}
