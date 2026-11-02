"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    num_blocks = p // m  # 4 blocks of 2x2
    
    # Each 2x2x2x2 Strassen uses 7 multiplications
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    col = 0
    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1 = (A00+A11)(B[0,k0]+B[1,k1])
        # Contributes: C[0,k0]+1, C[1,k1]+1
        U[0][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # P2 = (A10+A11) * B[0,k0]
        # Contributes: C[1,k0]+1
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P3 = A00 * (B[1,k0]-B[1,k1])
        # Contributes: C[0,k1]+1, C[1,k1]-1
        U[0][col] = 1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # P4 = A11 * (B[0,k1]-B[0,k0])
        # Contributes: C[0,k0]-1, C[1,k0]+1
        U[3][col] = 1
        V[0 * p + k1][col] = 1; V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = -1; W[k0 * n + 1][col] = 1
        col += 1
        
        # P5 = (A00+A01) * B[1,k1]
        # Contributes: C[0,k1]+1
        U[0][col] = 1; U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6 = (A10-A00) * (B[0,k0]+B[1,k0])
        # Contributes: C[0,k0]+1
        U[2][col] = 1; U[0][col] = -1
        V[0 * p + k0][col] = 1; V[1 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P7 = (A01-A11) * (B[0,k1]+B[1,k1])
        # Contributes: C[1,k1]+1
        U[1][col] = 1; U[3][col] = -1
        V[0 * p + k1][col] = 1; V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
