import time

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Constructs a rank-56 tensor decomposition for 2x2 * 2x16 matrix multiplication
    by applying the Strassen algorithm to each pair of columns (k, k+1) of B.
    
    For each pair (k, k+1), the 2x2 * 2x2 multiplication uses 7 scalar multiplications.
    The 2x16 B matrix is divided into 8 such pairs, giving 8 * 7 = 56 total multiplications.
    
    The output factor W follows the convention W[k*n + i, r] where k is the column index
    of C (0..15) and i is the row index of C (0..1).
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_pairs = p // 2     # 8 pairs
    rank = num_pairs * 7   # 56
    
    # U: shape (n*m, rank) = (4, 56)
    # V: shape (m*p, rank) = (32, 56)
    # W: shape (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A10) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Note: I'm using a slightly different but equivalent set of 7 multipliers.
    # Let me use the standard Strassen formulation:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A10) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for pair_idx in range(num_pairs):
        k0 = 2 * pair_idx        # first column of B in this pair
        k1 = 2 * pair_idx + 1    # second column of B in this pair
        base = pair_idx * 7      # starting index in the rank dimension
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00 and A11
        U[0][base + 0] = 1   # A[0,0]
        U[3][base + 0] = 1   # A[1,1]
        # V: B[0, k0] and B[1, k1]
        V[0 * p + k0][base + 0] = 1
        V[1 * p + k1][base + 0] = 1
        # W: C[0, k0] += M1, C[1, k1] += M1
        W[k0 * n + 0][base + 0] = 1
        W[k1 * n + 1][base + 0] = 1
        
        # M2 = (A10 + A11) * B00
        U[2][base + 1] = 1   # A[1,0]
        U[3][base + 1] = 1   # A[1,1]
        V[0 * p + k0][base + 1] = 1
        # W: C[1, k0] += M2, C[1, k1] -= M2
        W[k0 * n + 1][base + 1] = 1
        W[k1 * n + 1][base + 1] = -1
        
        # M3 = A00 * (B10 - B00)
        U[0][base + 2] = 1   # A[0,0]
        V[1 * p + k0][base + 2] = 1
        V[0 * p + k0][base + 2] = -1
        # W: C[0, k1] += M3, C[1, k1] += M3
        W[k1 * n + 0][base + 2] = 1
        W[k1 * n + 1][base + 2] = 1
        
        # M4 = A11 * (B01 - B11)
        U[3][base + 3] = 1   # A[1,1]
        V[0 * p + k1][base + 3] = 1
        V[1 * p + k1][base + 3] = -1
        # W: C[0, k0] += M4, C[1, k0] += M4
        W[k0 * n + 0][base + 3] = 1
        W[k0 * n + 1][base + 3] = 1
        
        # M5 = (A00 + A01) * B11
        U[0][base + 4] = 1   # A[0,0]
        U[1][base + 4] = 1   # A[0,1]
        V[1 * p + k1][base + 4] = 1
        # W: C[0, k0] -= M5, C[0, k1] += M5
        W[k0 * n + 0][base + 4] = -1
        W[k1 * n + 0][base + 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][base + 5] = 1   # A[1,0]
        U[0][base + 5] = -1  # A[0,0]
        V[0 * p + k0][base + 5] = 1
        V[0 * p + k1][base + 5] = 1
        # W: C[1, k1] += M6
        W[k1 * n + 1][base + 5] = 1
        
        # M7 = (A01 - A10) * (B10 + B11)
        U[1][base + 6] = 1   # A[0,1]
        U[2][base + 6] = -1  # A[1,0]
        V[1 * p + k0][base + 6] = 1
        V[1 * p + k1][base + 6] = 1
        # W: C[0, k0] += M7
        W[k0 * n + 0][base + 6] = 1
    
    return {"U": U, "V": V, "W": W}
