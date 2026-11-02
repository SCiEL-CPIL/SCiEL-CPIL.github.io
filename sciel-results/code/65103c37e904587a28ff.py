"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 46: 6 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 4)
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: j*p + k
    # W indices: k*n + i
    
    # Strassen blocks for k = 0, 2, 4, 6, 8, 10
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        b00_idx = 0 * p + k0
        b10_idx = 1 * p + k0
        b01_idx = 0 * p + k1
        b11_idx = 1 * p + k1
        
        # W indices
        w00_idx = k0 * n + 0
        w01_idx = k1 * n + 0
        w10_idx = k0 * n + 1
        w11_idx = k1 * n + 1
        
        # U, V, W contributions for M1..M7
        # M1 = (A00+A11)(B00+B11)
        col = 7 * b
        U[0][col] += 1; U[3][col] += 1
        V[b00_idx][col] += 1; V[b11_idx][col] += 1
        W[w00_idx][col] += 1
        W[w11_idx][col] += 1
        
        # M2 = (A10+A11)B00
        col = 7 * b + 1
        U[2][col] += 1; U[3][col] += 1
        V[b00_idx][col] += 1
        W[w10_idx][col] += 1
        W[w11_idx][col] -= 1
        
        # M3 = A00(B10-B11)
        col = 7 * b + 2
        U[0][col] += 1
        V[b10_idx][col] += 1; V[b11_idx][col] -= 1
        W[w01_idx][col] += 1
        W[w11_idx][col] += 1
        
        # M4 = A11(B01-B00)
        col = 7 * b + 3
        U[3][col] += 1
        V[b01_idx][col] += 1; V[b00_idx][col] -= 1
        W[w00_idx][col] += 1
        W[w10_idx][col] += 1
        
        # M5 = (A00+A01)B11
        col = 7 * b + 4
        U[0][col] += 1; U[1][col] += 1
        V[b11_idx][col] += 1
        W[w00_idx][col] -= 1
        W[w01_idx][col] += 1
        
        # M6 = (A10-A00)(B00+B10)
        col = 7 * b + 5
        U[2][col] += 1; U[0][col] -= 1
        V[b00_idx][col] += 1; V[b10_idx][col] += 1
        W[w11_idx][col] += 1
        
        # M7 = (A01-A11)(B01+B11)
        col = 7 * b + 6
        U[1][col] += 1; U[3][col] -= 1
        V[b01_idx][col] += 1; V[b11_idx][col] += 1
        W[w00_idx][col] += 1
        
    # Schoolbook block for k = 12
    k = 12
    b0_idx = 0 * p + k
    b1_idx = 1 * p + k
    w0_idx = k * n + 0
    w1_idx = k * n + 1
    
    # M8 = A00 B0
    col = 42
    U[0][col] += 1
    V[b0_idx][col] += 1
    W[w0_idx][col] += 1
    
    # M9 = A01 B1
    col = 43
    U[1][col] += 1
    V[b1_idx][col] += 1
    W[w0_idx][col] += 1
    
    # M10 = A10 B0
    col = 44
    U[2][col] += 1
    V[b0_idx][col] += 1
    W[w1_idx][col] += 1
    
    # M11 = A11 B1
    col = 45
    U[3][col] += 1
    V[b1_idx][col] += 1
    W[w1_idx][col] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
