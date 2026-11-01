"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    The multiplication tensor is a direct sum of 6 independent 2x2x2 multiplication tensors 
    (grouping columns in pairs). The minimal rank for 2x2x2 is 7 (Strassen).
    Total rank = 6 * 7 = 42.
    
    Interface:
    U: shape (n*m, rank) = (4, 42)
    V: shape (m*p, rank) = (24, 42)
    W: shape (p*n, rank) = (24, 42)
    
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    t_start = time.time()
    deadline = t_start + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen's algorithm coefficients for one 2x2 block
    # Local indices for A, B, C are 0..3 (row-major: 00, 01, 10, 11)
    
    U_s = np.zeros((4, 7), dtype=np.float64)
    V_s = np.zeros((4, 7), dtype=np.float64)
    W_s = np.zeros((4, 7), dtype=np.float64)
    
    # s=0: M1 = (A00+A11)*(B00+B11) -> C00, C11
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # s=1: M2 = (A10+A11)*B00 -> C10, C11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1  # C10 +1, C11 -1
    
    # s=2: M3 = A00*(B01-B11) -> C01, C11
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1   # C01 +1, C11 +1
    
    # s=3: M4 = A11*(B10-B00) -> C00, C10
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1   # C00 +1, C10 +1
    
    # s=4: M5 = (A00+A01)*B11 -> C00, C01
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1  # C00 -1, C01 +1
    
    # s=5: M6 = (A10-A00)*(B00+B01) -> C11
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1                  # C11 +1
    
    # s=6: M7 = (A01-A11)*(B10+B11) -> C00
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1                  # C00 +1

    # Verify Strassen reconstruction locally just in case (fast, small)
    # C = sum_s U_s V_s W_s^T ? No, it's a tensor.
    # Let's just trust the standard Strassen coefficients which are well-known.
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # U is the same for all blocks, so we add the same U_s to the slice
        # Actually, U is (n*m, rank). The rank dimension is split into blocks of 7.
        # So U[:, base:base+7] should be U_s for all t?
        # Wait. The rank is 42. Each block uses 7 rank units.
        # U is (4, 42).
        # For block t, the columns base..base+6 of U are used.
        # Since the A matrix is the same for all blocks, U[:, base:base+7] = U_s.
        U[:, base:base+7] = U_s
        
        # V mapping:
        # V shape (24, 42). Rows are j*p + k.
        # Block t uses B columns 2t, 2t+1.
        # V rows involved:
        # j=0, k=2t -> idx 0*p + 2t = 2t
        # j=0, k=2t+1 -> idx 0*p + 2t+1 = 2t+1
        # j=1, k=2t -> idx 1*p + 2t = 12 + 2t
        # j=1, k=2t+1 -> idx 1*p + 2t+1 = 12 + 2t+1
        
        V[0*p + k0, base:base+7] = V_s[0, :]
        V[0*p + k1, base:base+7] = V_s[1, :]
        V[1*p + k0, base:base+7] = V_s[2, :]
        V[1*p + k1, base:base+7] = V_s[3, :]
        
        # W mapping:
        # W shape (24, 42). Rows are k*n + i.
        # Block t uses C rows 2t, 2t+1.
        # W rows involved:
        # k=2t, i=0 -> idx 2t*n + 0 = 4t
        # k=2t, i=1 -> idx 2t*n + 1 = 4t+1
        # k=2t+1, i=0 -> idx (2t+1)*n + 0 = 4t+2
        # k=2t+1, i=1 -> idx (2t+1)*n + 1 = 4t+3
        
        W[k0*n + 0, base:base+7] = W_s[0, :]
        W[k0*n + 1, base:base+7] = W_s[2, :]  # Local C10 is index 2
        W[k1*n + 0, base:base+7] = W_s[1, :]  # Local C01 is index 1
        W[k1*n + 1, base:base+7] = W_s[3, :]  # Local C11 is index 3
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
