"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 blocks * 7 Strassen factors = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (n x m)
    # A[i, j] -> i * m + j
    # i in {0, 1}, j in {0, 1}
    
    # Indices for B (m x p)
    # B[j, k] -> j * p + k
    # j in {0, 1}, k in {0, ..., 13}
    
    # Indices for C (p x n)
    # C[k, i] -> k * n + i
    # k in {0, ..., 13}, i in {0, 1}
    
    num_blocks = p // 2  # 7 blocks of size 2
    
    for l in range(num_blocks):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Strassen factors s = 0..6
        for s in range(7):
            r = 7 * l + s
            
            # U coefficients (for A)
            # V coefficients (for B)
            # W coefficients (for C)
            
            if s == 0:
                # M1 = (A10 + A11) * B0,k0
                U[1 * m + 0][r] = 1  # A10
                U[1 * m + 1][r] = 1  # A11
                V[0 * p + k0][r] = 1 # B0,k0
                
                W[k0 * n + 0][r] = 1  # C0,k0
                W[k1 * n + 1][r] = 1  # C1,k1
                
            elif s == 1:
                # M2 = (A10 + A11) * (B1,k0 + B1,k1)
                U[1 * m + 0][r] = 1  # A10
                U[1 * m + 1][r] = 1  # A11
                V[1 * p + k0][r] = 1 # B1,k0
                V[1 * p + k1][r] = 1 # B1,k1
                
                W[k0 * n + 1][r] = 1  # C1,k0
                W[k1 * n + 1][r] = -1 # C1,k1
                
            elif s == 2:
                # M3 = A00 * (B0,k1 - B1,k1)
                U[0 * m + 0][r] = 1  # A00
                V[0 * p + k1][r] = 1 # B0,k1
                V[1 * p + k1][r] = -1 # B1,k1
                
                W[k1 * n + 0][r] = 1  # C0,k1
                W[k1 * n + 1][r] = 1  # C1,k1
                
            elif s == 3:
                # M4 = A11 * (B0,k0 + B1,k0)
                U[1 * m + 1][r] = 1  # A11
                V[0 * p + k0][r] = 1 # B0,k0
                V[1 * p + k0][r] = 1 # B1,k0
                
                W[k0 * n + 0][r] = 1  # C0,k0
                W[k0 * n + 1][r] = 1  # C1,k0
                
            elif s == 4:
                # M5 = (A00 + A10) * B1,k1
                U[0 * m + 0][r] = 1  # A00
                U[1 * m + 0][r] = 1  # A10
                V[1 * p + k1][r] = 1 # B1,k1
                
                W[k0 * n + 0][r] = -1 # C0,k0
                W[k1 * n + 0][r] = 1  # C0,k1
                
            elif s == 5:
                # M6 = (A01 - A11) * (B1,k0 + B1,k1)
                U[0 * m + 1][r] = 1  # A01
                U[1 * m + 1][r] = -1 # A11
                V[1 * p + k0][r] = 1 # B1,k0
                V[1 * p + k1][r] = 1 # B1,k1
                
                W[k1 * n + 1][r] = 1  # C1,k1
                
            elif s == 6:
                # M7 = (A00 - A10) * (B0,k0 + B0,k1)
                U[0 * m + 0][r] = 1  # A00
                U[1 * m + 0][r] = -1 # A10
                V[0 * p + k0][r] = 1 # B0,k0
                V[0 * p + k1][r] = 1 # B0,k1
                
                W[k0 * n + 0][r] = 1  # C0,k0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
