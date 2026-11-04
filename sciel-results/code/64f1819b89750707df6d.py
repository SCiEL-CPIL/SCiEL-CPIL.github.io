"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for g in range(5):  # 5 column groups of 2
        # Strassen block: A[:,0:2] (2x2) * B[0:2, 2g:2g+2] (2x2)
        # X indices in U: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        # Y indices in V: B[0,2g]=2g, B[0,2g+1]=2g+1, B[1,2g]=10+2g, B[1,2g+1]=11+2g
        # Z indices in W: C[0,2g]=4g, C[0,2g+1]=4g+2, C[1,2g]=4g+1, C[1,2g+1]=4g+3
        
        # M1 = (X00+X11)(Y00+Y11) -> Z00, Z11
        U[0][col] = 1; U[4][col] = 1
        V[2*g][col] = 1; V[11+2*g][col] = 1
        W[4*g][col] = 1; W[4*g+3][col] = 1
        col += 1
        
        # M2 = (X10+X11)Y00 -> Z10
        U[3][col] = 1; U[4][col] = 1
        V[2*g][col] = 1
        W[4*g+1][col] = 1
        col += 1
        
        # M3 = X00(Y01-Y11) -> Z01
        U[0][col] = 1
        V[2*g+1][col] = 1; V[11+2*g][col] = -1
        W[4*g+2][col] = 1
        col += 1
        
        # M4 = X11(Y10-Y00) -> Z00
        U[4][col] = 1
        V[10+2*g][col] = 1; V[2*g][col] = -1
        W[4*g][col] = 1
        col += 1
        
        # M5 = (X00+X01)Y11 -> Z11
        U[0][col] = 1; U[1][col] = 1
        V[11+2*g][col] = 1
        W[4*g+3][col] = 1
        col += 1
        
        # M6 = (X10-X00)(Y00+Y01) -> Z11
        U[3][col] = 1; U[0][col] = -1
        V[2*g][col] = 1; V[2*g+1][col] = 1
        W[4*g+3][col] = 1
        col += 1
        
        # M7 = (X01-X11)(Y10+Y11) -> Z00, Z11
        U[1][col] = 1; U[4][col] = -1
        V[10+2*g][col] = 1; V[11+2*g][col] = 1
        W[4*g][col] = 1; W[4*g+3][col] = 1
        col += 1
    
    # Schoolbook: A[:,2] (2x1) * B[2:3,:] (1x10)
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1
            V[20+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
