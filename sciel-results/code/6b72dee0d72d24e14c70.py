"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Block structure:
    # m=5 split into [2, 3], p=6 split into [2, 2, 2]
    # 3 blocks of 2x2x2 (Strassen, rank 7) + 3 blocks of 2x3x2 (rank 11) = 54
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_2x2x2(ai, aj, bi, bk, wi, wk):
        nonlocal col
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[(ai+0)*m + (aj+0)][col] = 1
        U[(ai+1)*m + (aj+1)][col] = 1
        V[(bi+0)*p + (bk+0)][col] = 1
        V[(bi+1)*p + (bk+1)][col] = 1
        W[(wk+0)*n + (wi+0)][col] = 1
        W[(wk+1)*n + (wi+1)][col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        U[(ai+1)*m + (aj+0)][col] = 1
        U[(ai+1)*m + (aj+1)][col] = 1
        V[(bi+0)*p + (bk+0)][col] = 1
        W[(wk+0)*n + (wi+1)][col] = 1
        col += 1
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        U[(ai+0)*m + (aj+0)][col] = 1
        V[(bi+0)*p + (bk+1)][col] = 1
        V[(bi+1)*p + (bk+1)][col] = -1
        W[(wk+1)*n + (wi+0)][col] = 1
        col += 1
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        U[(ai+1)*m + (aj+1)][col] = 1
        V[(bi+1)*p + (bk+0)][col] = 1
        V[(bi+0)*p + (bk+0)][col] = -1
        W[(wk+0)*n + (wi+0)][col] = 1
        W[(wk+0)*n + (wi+1)][col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        U[(ai+0)*m + (aj+0)][col] = 1
        U[(ai+0)*m + (aj+1)][col] = 1
        V[(bi+1)*p + (bk+1)][col] = 1
        W[(wk+1)*n + (wi+0)][col] = 1
        W[(wk+0)*n + (wi+0)][col] = -1
        col += 1
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[(ai+1)*m + (aj+0)][col] = 1
        U[(ai+0)*m + (aj+0)][col] = -1
        V[(bi+0)*p + (bk+0)][col] = 1
        V[(bi+0)*p + (bk+1)][col] = 1
        W[(wk+1)*n + (wi+1)][col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[(ai+0)*m + (aj+1)][col] = 1
        U[(ai+1)*m + (aj+1)][col] = -1
        V[(bi+1)*p + (bk+0)][col] = 1
        V[(bi+1)*p + (bk+1)][col] = 1
        W[(wk+0)*n + (wi+0)][col] = 1
        W[(wk+1)*n + (wi+1)][col] = -1
        col += 1
    
    def mul_2x1x2(ai, aj, bi, bk, wi, wk):
        nonlocal col
        for i in range(2):
            for k in range(2):
                U[(ai+i)*m + (aj+0)][col] = 1
                V[(bi+0)*p + (bk+k)][col] = 1
                W[(wk+k)*n + (wi+i)][col] = 1
                col += 1
    
    # Build blocks
    # (0,j): A[0:2,0:2] * B[0:2,2j:2j+2] -> C[0:2,2j:2j+2]
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    strassen_2x2x2(0, 0, 0, 4, 0, 4)
    
    # (1,j): A[0:2,2:5] * B[2:5,2j:2j+2] -> C[0:2,2j:2j+2]
    # Split m_part=3 into 1+2
    for j in range(3):
        bk = 2*j
        mul_2x1x2(0, 2, 2, bk, 0, bk)
        strassen_2x2x2(0, 3, 3, bk, 0, bk)
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
