"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x5)
    # A1 indices: i in {0,1}, j=0 -> U[i*3+0]
    # B1 indices: j=0, k in {0..4} -> V[0*5+k]
    # C indices: i in {0,1}, k in {0..4} -> W[k*2+i]
    for i in range(2):
        for k in range(5):
            U[i * 3 + 0][col] = 1
            V[0 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x5)
    # A2 indices: i in {0,1}, j in {1,2} -> U[i*3+j]
    # B2 indices: j in {1,2}, k in {0..4} -> V[j*5+k]
    # C indices: i in {0,1}, k in {0..4} -> W[k*2+i]
    
    # Helper to add a Strassen 2x2 * 2x2 block
    # A2 is fixed indices: 
    # A00 -> U[0*3+1]=U[1], A01 -> U[0*3+2]=U[2]
    # A10 -> U[1*3+1]=U[4], A11 -> U[1*3+2]=U[5]
    # B_block is specified by k_offset
    # B00 -> V[1*5+k0], B01 -> V[1*5+k1]
    # B10 -> V[2*5+k0], B11 -> V[2*5+k1]
    # C_block is specified by k_offset
    # C00 -> W[k0*2+0], C01 -> W[k1*2+0]
    # C10 -> W[k0*2+1], C11 -> W[k1*2+1]
    
    def add_strassen_block(k0, k1):
        nonlocal col
        # M1 = (A00 + A11)(B10 + B01)
        U[1][col] += 1; U[5][col] += 1
        V[1*5+k0][col] += 1; V[1*5+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[4][col] += 1; U[5][col] += 1
        V[1*5+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[1][col] += 1
        V[1*5+k1][col] += 1; V[2*5+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[5][col] += 1
        V[2*5+k0][col] += 1; V[1*5+k0][col] -= 1
        W[k0*2+0][col] -= 1; W[k0*2+1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[1][col] += 1; U[2][col] += 1
        V[2*5+k1][col] += 1
        W[k1*2+0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[4][col] += 1; U[1][col] -= 1
        V[1*5+k0][col] += 1; V[1*5+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        
        # M7 = (A01 - A10) (B10 + B11)
        U[2][col] += 1; U[4][col] -= 1
        V[2*5+k0][col] += 1; V[2*5+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

    # Block 2a: k in {0, 1}
    add_strassen_block(0, 1)
    
    # Block 2b: k in {2, 3}
    add_strassen_block(2, 3)
    
    # Block 2c: k=4 (2x2 * 2x1)
    # Terms:
    # 1. A00 B04 -> C04
    U[1][col] = 1
    V[1*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # 2. A01 B14 -> C04
    U[2][col] = 1
    V[2*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # 3. A10 B04 -> C14
    U[4][col] = 1
    V[1*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # 4. A11 B14 -> C14
    U[5][col] = 1
    V[2*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
