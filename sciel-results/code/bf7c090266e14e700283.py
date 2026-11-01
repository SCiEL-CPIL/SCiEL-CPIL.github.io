"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    # 1. A11 (2x2) * B1 (2x4) -> 2 Strassen blocks (2x2 * 2x2) -> 14 mults
    # 2. A12 (2x1) * B2 (1x4) -> Schoolbook -> 8 mults
    # 3. A21 (1x2) * B1 (2x4) -> Schoolbook -> 8 mults
    # 4. A22 (1x1) * B2 (1x4) -> Schoolbook -> 4 mults
    # Total 34
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # A indices: i0, i0+1; j0, j0+1
        # B indices: j0, j0+1; k0, k0+1
        # C indices: i0, i0+1; k0, k0+1
        
        # m1 = (a+d)(e+h)
        # C11 += m1, C22 += m1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # m2 = (c+d)e
        # C21 += m2, C22 -= m2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # m3 = a(f-h)
        # C12 += m3, C22 += m3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + k0][col] += -1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # m4 = d(g-e)
        # C11 += m4, C21 += m4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # m5 = (a+b)h
        # C11 -= m5, C12 += m5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        # C22 += m6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        # C11 += m7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m + j][col] += 1
        V[j*p + k][col] += 1
        W[k*n + i][col] += 1
        col += 1

    # Block 1: A11 (i in 0,1; j in 0,1) * B1 (j in 0,1; k in 0,3)
    # Split B1 into k=0,1 and k=2,3
    add_strassen(0, 0, 0) # A11 * B1a
    add_strassen(0, 0, 2) # A11 * B1b
    
    # Block 2: A12 (i in 0,1; j=2) * B2 (j=2; k in 0,3)
    for i in range(2):
        for k in range(4):
            add_schoolbook(i, 2, k)
            
    # Block 3: A21 (i=2; j in 0,1) * B1 (j in 0,1; k in 0,3)
    for j in range(2):
        for k in range(4):
            add_schoolbook(2, j, k)
            
    # Block 4: A22 (i=2; j=2) * B2 (j=2; k in 0,3)
    for k in range(4):
        add_schoolbook(2, 2, k)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
