import time
import random
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    rank = 49  # Rank-49 recursive Strassen
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2 tensor
    # Indices for 2x2 blocks: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # 7 products
    # 1: (A00+A11)(B00+B11) -> M1
    # 2: (A10+A11)B00       -> M2
    # 3: A00(B10-B11)       -> M3
    # 4: A11(B01-B00)       -> M4
    # 5: (A00+A01)B11       -> M5
    # 6: (A10-A00)(B00+B01) -> M6
    # 7: (A01-A11)(B10+B11) -> M7
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A side): 2x2 -> 4 rows (flattened row-major), 4 cols (blocks)
    # Actually, U has shape (n*m, rank) = (16, 49).
    # For 2x2 case, n=2, m=2. Rows 0..3.
    # For 4x4, we use Kronecker product structure.
    # Rank 49 = 7 * 7.
    # This corresponds to two levels of Strassen.
    # Level 1: 7 products.
    # Level 2: Each product is a 2x2 multiplication, decomposed into 7 sub-products.
    # So indices k from 0 to 48.
    # k = k1 * 7 + k2, where k1 is the level-1 product index (0..6) and k2 is the level-2 product index (0..6).
    
    # Let's define the coefficients for the 2x2 case first.
    # U_2x2: shape (4, 7)
    # Rows: A00, A01, A10, A11
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00: M1, M3, M5
        [0, 0, 0, 0, 1, 1, -1], # A01: M5, M6, M7
        [0, 1, 0, 0, 0, -1, 1], # A10: M2, M6, M7
        [1, 1, 0, 1, 0, 0, 0]  # A11: M1, M2, M4
    ])
    
    # V_2x2: shape (4, 7)
    # Rows: B00, B01, B10, B11
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00: M1, M2, M4, M6
        [1, 0, 0, 1, 0, 1, 0], # B01: M1, M4, M6
        [1, 0, 1, 0, 0, 0, 1], # B10: M1, M3, M7
        [1, 0, -1, -1, 1, 0, 1] # B11: M1, M3, M4, M5, M7
    ])
    
    # W_2x2: shape (4, 7)
    # Rows: C00, C01, C10, C11
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0], # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0], # C10: M2, M4
        [1, -1, 1, 0, 0, 1, 0]  # C11: M1, M2, M3, M6
    ])
    
    # Now construct 4x4 factors using Kronecker products.
    # U_4x4 = U_2x2 \u2295 U_2x2 ?
    # Let's check indices.
    # A is 4x4. Flattened row-major: A00, A01, A02, A03, A10, ..., A33.
    # Block Aij is 2x2.
    # A00 = [[A00, A01], [A10, A11]]
    # A01 = [[A02, A03], [A12, A13]]
    # A10 = [[A20, A21], [A30, A31]]
    # A11 = [[A22, A23], [A32, A33]]
    
    # The Kronecker product U_2x2 \u2295 U_2x2 will have shape (4*4, 7*7) = (16, 49).
    # The rows are indexed by (i_block, i_local).
    # The columns are indexed by (k_block, k_local).
    
    U_4x4 = np.kron(U_2x2, U_2x2)
    V_4x4 = np.kron(V_2x2, V_2x2)
    W_4x4 = np.kron(W_2x2, W_2x2)
    
    # Convert to list of lists
    for i in range(16):
        for k in range(49):
            U[i][k] = float(U_4x4[i][k])
            V[i][k] = float(V_4x4[i][k])
            W[i][k] = float(W_4x4[i][k])
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
