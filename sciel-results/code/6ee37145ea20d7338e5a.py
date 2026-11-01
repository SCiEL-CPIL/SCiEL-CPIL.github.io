"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-55 decomposition using Strassen's algorithm on 2x2 blocks.
    
    We split the middle dimension m=3 into j in {0,1} (Strassen) and j=2 (standard).
    For j in {0,1}: split the k dimension into pairs (2k, 2k+1) for k=0..4.
    For each pair, use Strassen (7 multiplications) for A_{2x2} * B_{2x2}.
    For j=2: standard outer product A_{:,2} * B_{2,:} (20 multiplications).
    
    Total rank = 5 * 7 + 20 = 55.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    rank = 55
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen block for j in {0,1}, k pairs (2k, 2k+1)
    # For each pair (k0, k1) where k0=2k, k1=2k+1:
    # A indices: i in {0,1}, j in {0,1} -> U[i*m + j, col]
    # B indices: j in {0,1}, k in {k0, k1} -> V[j*p + k, col]
    # C indices: k in {k0, k1}, i in {0,1} -> W[k*n + i, col]
    
    for k in range(p // 2):
        k0 = 2 * k
        k1 = 2 * k + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        W[k0 * n + 0][col] += 1   # C00
        W[k0 * n + 1][col] += 1   # C11 contribution... wait
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me redo this more carefully.
        # For block (k0, k1):
        # C[i][k0] and C[i][k1] for i in {0,1}
        # C00 (i=0, k=k0) = M1 + M4 - M5 + M7
        # C01 (i=0, k=k1) = M3 + M5
        # C10 (i=1, k=k0) = M2 + M4
        # C11 (i=1, k=k1) = M1 - M2 + M3 + M6
        
        # W index: k*n + i
        # W[k0*n + 0] = C[0][k0] = M1 + M4 - M5 + M7
        # W[k1*n + 0] = C[0][k1] = M3 + M5
        # W[k0*n + 1] = C[1][k0] = M2 + M4
        # W[k1*n + 1] = C[1][k1] = M1 - M2 + M3 + M6
        
        # M1: (A00 + A11)(B00 + B11)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2: (A10 + A11)B00
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3: A00(B01 - B11)
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4: A11(B10 - B00)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5: (A00 + A01)B11
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7: (A01 - A11)(B10 + B11)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    # Standard multiplication for j=2: C[i][k] += A[i][2] * B[2][k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
