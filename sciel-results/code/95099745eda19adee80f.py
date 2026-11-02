import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    deadline = time.time() + 110  # safety margin before 120s limit
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen's 7 intermediate products for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_pairs):
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        # M1 = (A00 + A11) * (B00 + B11)
        c = base + 0
        U[0 * m + 0][c] = 1   # A00
        U[1 * m + 1][c] = 1   # A11
        V[0 * p + k0][c] = 1  # B[0, k0]
        V[1 * p + k1][c] = 1  # B[1, k1]
        W[k0 * n + 0][c] = 1  # C[0, k0]
        W[k1 * n + 1][c] = 1  # C[1, k1]
        
        # M2 = (A10 + A11) * B00
        c = base + 1
        U[1 * m + 0][c] = 1   # A10
        U[1 * m + 1][c] = 1   # A11
        V[0 * p + k0][c] = 1  # B[0, k0]
        W[k0 * n + 1][c] = 1  # C[1, k0]
        W[k1 * n + 1][c] = -1 # C[1, k1]
        
        # M3 = A00 * (B01 - B11)
        c = base + 2
        U[0 * m + 0][c] = 1   # A00
        V[0 * p + k1][c] = 1  # B[0, k1]
        V[1 * p + k1][c] = -1 # B[1, k1]
        W[k1 * n + 0][c] = 1  # C[0, k1]
        W[k1 * n + 1][c] = 1  # C[1, k1]
        
        # M4 = A11 * (B10 - B00)
        c = base + 3
        U[1 * m + 1][c] = 1   # A11
        V[1 * p + k0][c] = 1  # B[1, k0]
        V[0 * p + k0][c] = -1 # B[0, k0]
        W[k0 * n + 0][c] = 1  # C[0, k0]
        W[k0 * n + 1][c] = 1  # C[1, k0]
        
        # M5 = (A00 + A01) * B11
        c = base + 4
        U[0 * m + 0][c] = 1   # A00
        U[0 * m + 1][c] = 1   # A01
        V[1 * p + k1][c] = 1  # B[1, k1]
        W[k0 * n + 0][c] = -1 # C[0, k0]
        W[k1 * n + 0][c] = 1  # C[0, k1]
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[1 * m + 0][c] = 1   # A10
        U[0 * m + 0][c] = -1  # A00
        V[0 * p + k0][c] = 1  # B[0, k0]
        V[0 * p + k1][c] = 1  # B[0, k1]
        W[k1 * n + 1][c] = 1  # C[1, k1]
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[0 * m + 1][c] = 1   # A01
        U[1 * m + 1][c] = -1  # A11
        V[1 * p + k0][c] = 1  # B[1, k0]
        V[1 * p + k1][c] = 1  # B[1, k1]
        W[k0 * n + 0][c] = 1  # C[0, k0]
        
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
