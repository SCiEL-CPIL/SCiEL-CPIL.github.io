"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen-like decomposition for each pair of columns
    # 3 pairs, each with 11 multiplications = 33 total
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    for k_pair in range(0, p, 2):
        k1, k2 = k_pair, k_pair + 1
        # Strassen multiplications for 2x2 subproblem (A rows 0,1; B rows 0,1)
        # M1 = (A[0,0]+A[1,1])*(B[0,k1]+B[1,k2])
        U[0*3+0][col] = 1  # A[0,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*6+k1][col] = 1  # B[0,k1]
        V[1*6+k2][col] = 1  # B[1,k2]
        # W contributions: C[0,k1] += M1, C[1,k2] += M1
        W[k1*2+0][col] = 1  # C[0,k1]
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,k1]
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*6+k1][col] = 1
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
        
        # M3 = A[0,0]*(B[1,k1]-B[1,k2])
        U[0*3+0][col] = 1
        V[1*6+k1][col] = 1
        V[1*6+k2][col] = -1
        W[k2*2+0][col] = 1  # C[0,k2]
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
        
        # M4 = A[1,1]*(B[0,k2]-B[0,k1])
        U[1*3+1][col] = 1
        V[0*6+k2][col] = 1
        V[0*6+k1][col] = -1
        W[k1*2+0][col] = 1  # C[0,k1]
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,k2]
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*6+k2][col] = 1
        W[k1*2+0][col] = -1  # C[0,k1]
        W[k2*2+0][col] = 1  # C[0,k2]
        col += 1
        
        # M6 = (A[1,0]-A[0,0])*(B[0,k1]+B[0,k2])
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*6+k1][col] = 1
        V[0*6+k2][col] = 1
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[1,k1]+B[1,k2])
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*6+k1][col] = 1
        V[1*6+k2][col] = 1
        W[k1*2+0][col] = 1  # C[0,k1]
        col += 1
        
        # Additional: A[0,2]*B[2,k1]
        U[0*3+2][col] = 1
        V[2*6+k1][col] = 1
        W[k1*2+0][col] = 1  # C[0,k1]
        col += 1
        
        # Additional: A[1,2]*B[2,k1]
        U[1*3+2][col] = 1
        V[2*6+k1][col] = 1
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
        
        # Additional: A[0,2]*B[2,k2]
        U[0*3+2][col] = 1
        V[2*6+k2][col] = 1
        W[k2*2+0][col] = 1  # C[0,k2]
        col += 1
        
        # Additional: A[1,2]*B[2,k2]
        U[1*3+2][col] = 1
        V[2*6+k2][col] = 1
        W[k2*2+1][col] = 1  # C[1,k2]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
