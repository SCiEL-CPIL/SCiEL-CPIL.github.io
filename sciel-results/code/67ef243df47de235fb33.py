"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    R = 56
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition
    # X is A[i:i+2, j:j+2]
    # Y is B[j:j+2, k:k+2]
    # Contributes to C[i:i+2, k:k+2]
    def add_strassen(i, j, k):
        nonlocal col
        # X indices in A: (i+u, j+v) for u,v in 0..1
        # Y indices in B: (j+v, k+w) for v,w in 0..1
        # C indices: (i+u, k+w)
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[i*m + j][col] = 1
        U[(i+1)*m + (j+1)][col] = 1
        V[j*p + k][col] = 1
        V[(j+1)*p + (k+1)][col] = 1
        # P00 += M1, P11 += M1
        W[k*n + i][col] = 1
        W[(k+1)*n + (i+1)][col] = 1
        col += 1
        
        # M2 = (X10 + X11)Y00
        U[(i+1)*m + j][col] = 1
        U[(i+1)*m + (j+1)][col] = 1
        V[j*p + k][col] = 1
        # P10 += M2
        W[k*n + (i+1)][col] = 1
        # P11 -= M2
        W[(k+1)*n + (i+1)][col] = -1
        col += 1
        
        # M3 = X00(Y01 - Y11)
        U[i*m + j][col] = 1
        V[j*p + (k+1)][col] = 1
        V[(j+1)*p + (k+1)][col] = -1
        # P01 += M3
        W[(k+1)*n + i][col] = 1
        # P11 += M3
        W[(k+1)*n + (i+1)][col] = 1
        col += 1
        
        # M4 = X11(Y10 - Y00)
        U[(i+1)*m + (j+1)][col] = 1
        V[(j+1)*p + k][col] = 1
        V[j*p + k][col] = -1
        # P00 += M4
        W[k*n + i][col] = 1
        # P10 += M4
        W[k*n + (i+1)][col] = 1
        col += 1
        
        # M5 = (X00 + X01)Y11
        U[i*m + j][col] = 1
        U[i*m + (j+1)][col] = 1
        V[(j+1)*p + (k+1)][col] = 1
        # P00 -= M5
        W[k*n + i][col] = -1
        # P01 += M5
        W[(k+1)*n + i][col] = 1
        col += 1
        
        # M6 = (X10 - X00)(Y00 + Y01)
        U[(i+1)*m + j][col] = 1
        U[i*m + j][col] = -1
        V[j*p + k][col] = 1
        V[j*p + (k+1)][col] = 1
        # P11 += M6
        W[(k+1)*n + (i+1)][col] = 1
        col += 1
        
        # M7 = (X01 - X11)(Y10 + Y11)
        U[i*m + (j+1)][col] = 1
        U[(i+1)*m + (j+1)][col] = -1
        V[(j+1)*p + k][col] = 1
        V[(j+1)*p + (k+1)][col] = 1
        # P00 += M7
        W[k*n + i][col] = 1
        # P11 += M7
        W[(k+1)*n + (i+1)][col] = 1
        col += 1

    # 4 blocks for A_L B_T
    for q in range(4):
        add_strassen(0, 0, 2*q)
        
    # 4 blocks for A_R B_B
    for q in range(4):
        add_strassen(0, 2, 2*q)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
