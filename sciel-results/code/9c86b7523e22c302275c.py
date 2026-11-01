"""Rewritten Strassen-based decomposition for 2x2x7 matrix multiplication."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 25
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 25
    V = [[0] * rank for _ in range(m * p)]  # 14 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    # Strassen decomposition for a 2x2 * 2x2 block
    # Indices in U: i*m+j. i in {0,1}, j in {0,1}
    # Indices in V: j*p+k. j in {0,1}, k in {0,1} relative to block start
    # Indices in W: k*n+i. k in {0,1} relative to block start, i in {0,1}
    
    def add_strassen_block(offset_k, start_rank):
        # k0 = offset_k, k1 = offset_k + 1
        k0 = offset_k
        k1 = offset_k + 1
        
        # r = start_rank + 0: M1 = A00 * (B0_k1 - B1_k1)
        r = start_rank + 0
        U[0][r] = 1  # A00
        V[0 * p + k1][r] = 1   # B0_k1
        V[1 * p + k1][r] = -1  # B1_k1
        W[k1 * n + 0][r] = 1   # C0_k1
        W[k1 * n + 1][r] = 1   # C1_k1
        
        # r = start_rank + 1: M2 = (A00 + A01) * B1_k1
        r = start_rank + 1
        U[0][r] = 1  # A00
        U[1][r] = 1  # A01
        V[1 * p + k1][r] = 1  # B1_k1
        W[k0 * n + 0][r] = -1 # C0_k0
        W[k1 * n + 0][r] = 1  # C0_k1
        
        # r = start_rank + 2: M3 = (A10 - A11) * B0_k0
        r = start_rank + 2
        U[2][r] = 1  # A10
        U[3][r] = -1 # A11
        V[0 * p + k0][r] = 1  # B0_k0
        W[k0 * n + 1][r] = 1  # C1_k0
        W[k1 * n + 1][r] = 1  # C1_k1
        
        # r = start_rank + 3: M4 = A11 * (B1_k0 - B0_k0)
        r = start_rank + 3
        U[3][r] = 1  # A11
        V[1 * p + k0][r] = 1  # B1_k0
        V[0 * p + k0][r] = -1 # B0_k0
        W[k0 * n + 0][r] = 1  # C0_k0
        W[k1 * n + 1][r] = -1 # C1_k1
        
        # r = start_rank + 4: M5 = (A00 + A11) * (B0_k0 + B1_k0)
        r = start_rank + 4
        U[0][r] = 1  # A00
        U[3][r] = 1  # A11
        V[0 * p + k0][r] = 1  # B0_k0
        V[1 * p + k0][r] = 1  # B1_k0
        W[k0 * n + 0][r] = 1  # C0_k0
        W[k0 * n + 1][r] = 1  # C1_k0
        
        # r = start_rank + 5: M6 = (A01 - A11) * (B1_k0 + B1_k1)
        r = start_rank + 5
        U[1][r] = 1  # A01
        U[3][r] = -1 # A11
        V[1 * p + k0][r] = 1  # B1_k0
        V[1 * p + k1][r] = 1  # B1_k1
        W[k0 * n + 0][r] = 1  # C0_k0
        
        # r = start_rank + 6: M7 = (A10 + A00) * (B0_k1 - B1_k1)
        r = start_rank + 6
        U[2][r] = 1  # A10
        U[0][r] = 1  # A00
        V[0 * p + k1][r] = 1  # B0_k1
        V[1 * p + k1][r] = -1 # B1_k1
        W[k1 * n + 1][r] = 1  # C1_k1

    # Block 1: columns 0, 1
    add_strassen_block(0, 0)
    # Block 2: columns 2, 3
    add_strassen_block(2, 7)
    # Block 3: columns 4, 5
    add_strassen_block(4, 14)
    
    # Block 4: single column 6
    # Rank 4 schoolbook for A (2x2) * B_col (2x1)
    # C0_k6 = A00*B0_k6 + A01*B1_k6
    # C1_k6 = A10*B0_k6 + A11*B1_k6
    
    k6 = 6
    start_rank = 21
    
    # r = 21: A00 * B0_k6 -> C0_k6
    r = start_rank + 0
    U[0][r] = 1
    V[0 * p + k6][r] = 1
    W[k6 * n + 0][r] = 1
    
    # r = 22: A01 * B1_k6 -> C0_k6
    r = start_rank + 1
    U[1][r] = 1
    V[1 * p + k6][r] = 1
    W[k6 * n + 0][r] = 1
    
    # r = 23: A10 * B0_k6 -> C1_k6
    r = start_rank + 2
    U[2][r] = 1
    V[0 * p + k6][r] = 1
    W[k6 * n + 1][r] = 1
    
    # r = 24: A11 * B1_k6 -> C1_k6
    r = start_rank + 3
    U[3][r] = 1
    V[1 * p + k6][r] = 1
    W[k6 * n + 1][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
