import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # A indices: (i1, j1), (i1, j2), (i2, j1), (i2, j2)
    # B indices: (j1, k1), (j1, k2), (j2, k1), (j2, k2)
    # C indices: (i1, k1), (i1, k2), (i2, k1), (i2, k2)
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (a+d)(e+h)
        # a: U[i1*m+j1], d: U[i2*m+j2]
        # e: V[j1*p+k1], h: V[j2*p+k2]
        # C11: W[k1*n+i1], C12: W[k2*n+i1], C21: W[k1*n+i2], C22: W[k2*n+i2]
        
        # We need to expand (a+d)(e+h) = ae + ah + de + dh
        # This is not a single multiplication in the tensor sense unless we allow sums in U/V.
        # The tensor decomposition requires U, V, W to be sparse with entries 1, -1, 0.
        # Strassen works by changing the basis of the inputs and outputs.
        # So we need to modify U and V to reflect the sums.
        
        # This is tricky to implement inline without pre-computing the full U, V, W for the block.
        # I will explicitly write out the 7 terms for each block.
        
        # Term 1: (A11+A22)(B11+B22)
        # U entries: 1 at (i1,j1), 1 at (i2,j2)
        # V entries: 1 at (j1,k1), 1 at (j2,k2)
        # W entries: 1 at (k1,i1), 1 at (k2,i1), 1 at (k1,i2), 1 at (k2,i2)
        # But we need to subtract/add to get the correct C.
        
        # It is easier to just add the 7 Strassen terms as separate rank-1 tensors.
        # Each term is a product of a linear combination of A and a linear combination of B.
        # The linearity allows us to just add the corresponding rows/cols to U and V.
        
        # M1: (a+d)(e+h)
        U[i1*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i1][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] += 1
        
        # M2: (c+d)e
        U[i2*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1
        
        # M3: a(f-h)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1
        V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1
        
        # M4: d(g-e)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1
        V[j1*p+k1][col] += -1
        W[k1*n+i2][col] += 1
        
        # M5: (a+b)h
        U[i1*m+j1][col] += 1
        U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k2*n+i1][col] += 1
        
        # M6: (c-a)(e+f)
        U[i2*m+j1][col] += 1
        U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1
        V[j1*p+k2][col] += 1
        W[k1*n+i2][col] += -1
        W[k2*n+i2][col] += -1
        
        # M7: (b-d)(g+h)
        U[i1*m+j2][col] += 1
        U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i1][col] += 1
        
        col += 7

    # Helper to add schoolbook 2x1x6
    # A: (i1, j1), (i2, j1)
    # B: (j1, k1..k6)
    # C: (i1, k1..k6), (i2, k1..k6)
    def add_2x1x6(i1, i2, j1, k_start):
        nonlocal col
        for k in range(6):
            U[i1*m+j1][col] += 1
            U[i2*m+j1][col] += 1
            V[j1*p+(k_start+k)][col] += 1
            W[(k_start+k)*n+i1][col] += 1
            W[(k_start+k)*n+i2][col] += 1
            col += 1

    # A1 B1: 2x2x6
    # A1: rows 0,1; cols 0,1
    # B1: rows 0,1; cols 0..5
    # Split B1 into 3 blocks of 2x2
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 0, 1, 4, 5)
    
    # A2 B2: 2x3x6
    # A2: rows 0,1; cols 2,3,4
    # B2: rows 2,3,4; cols 0..5
    # Split A2 into 2x1 (col 2) and 2x2 (cols 3,4)
    # Split B2 into 1x6 (row 2) and 2x6 (rows 3,4)
    
    # A21 B21: 2x1x6
    # A21: rows 0,1; col 2
    # B21: row 2; cols 0..5
    add_2x1x6(0, 1, 2, 0)
    
    # A22 B22: 2x2x6
    # A22: rows 0,1; cols 3,4
    # B22: rows 3,4; cols 0..5
    # Split B22 into 3 blocks of 2x2
    add_strassen(0, 1, 3, 4, 0, 1)
    add_strassen(0, 1, 3, 4, 2, 3)
    add_strassen(0, 1, 3, 4, 4, 5)
    
    return {"U": U, "V": V, "W": W}
