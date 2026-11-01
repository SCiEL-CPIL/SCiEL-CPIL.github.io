import time
import sys

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The rank-35 decomposition is based on 5 independent Strassen 2x2x2 blocks.
    U shape: (4, 35)
    V shape: (20, 35)
    W shape: (20, 35)
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, but numpy would be faster for verification.
    # The interface expects lists of lists or arrays. The seed program uses lists.
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices (local 2x2): 0:00, 1:01, 2:10, 3:11
    
    strassen_A = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1]  # P7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: e+h
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: f-h
        [-1, 0, 1, 0], # P4: g-e
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: e+f
        [0, 0, 1, 1]   # P7: g+h
    ]
    
    # W contributions to local 2x2 C matrix
    # C indices: 0:00, 1:01, 2:10, 3:11
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    # P1 = (a+d)(e+h) -> C00 +1, C11 +1
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    # P2 = (c+d)e -> C10 +1, C11 -1
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    # P3 = a(f-h) -> C01 +1, C11 +1
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    # P4 = d(g-e) -> C00 +1, C10 +1
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    # P5 = (a+b)h -> C00 -1, C01 +1
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    # P6 = (c-a)(e+f) -> C11 +1
    w_contrib[5][3] = 1
    
    # P7 = (b-d)(g+h) -> C00 +1
    w_contrib[6][0] = 1
    
    # Fill U, V, W
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # U: A coefficients
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # V: B coefficients
            # B is 2x10. Block b corresponds to columns 2*b and 2*b+1.
            # V index: j*p + k.
            # j=0, k=2*b -> 0*p + 2*b = 2*b
            # j=0, k=2*b+1 -> 2*b+1
            # j=1, k=2*b -> 1*p + 2*b = 10 + 2*b
            # j=1, k=2*b+1 -> 10 + 2*b + 1
            
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # W: C coefficients
            # W index: k*n + i.
            # C is 2x10. Block b corresponds to columns 2*b and 2*b+1 of C?
            # Wait, C is p x n = 10 x 2.
            # C[k, i].
            # For block b, the output columns of the multiplication A(2x2) * B_block(2x2) are C[:, 2*b] and C[:, 2*b+1]?
            # No, B is 2x10. A is 2x2.
            # C = A * B. C is 2x10.
            # The problem statement says: "The output factor follows the official trace convention W[k*n+i,r]."
            # And "W: shape (p*n, rank)".
            # So W index is k*n + i.
            # k is the row of C (0..9), i is the column of C (0..1).
            # In the block decomposition, we are computing C[:, 2*b] and C[:, 2*b+1]?
            # No, B's columns are 0..9.
            # C's columns are 0..9.
            # C[:, k] = A * B[:, k].
            # So for each k (0..9), we compute a 2x1 vector.
            # My previous block decomposition grouped B into 5 blocks of 2 columns.
            # Block b consists of B[:, 2*b] and B[:, 2*b+1].
            # The result is C[:, 2*b] and C[:, 2*b+1].
            # So k_global = 2*b + k_local.
            # i is the column index of C (0 or 1).
            
            # W index = k_global * n + i = (2*b + k_local) * 2 + i.
            
            # Local C indices in w_contrib:
            # 0: C00 (i=0, k_local=0)
            # 1: C01 (i=0, k_local=1) -- Wait, C01 usually means row 0, col 1.
            # In our notation C[i, k].
            # w_contrib is for the 2x2 block result M = A * B_block.
            # M is 2x2. M[i, k_local].
            # So M[0,0] corresponds to C[0, 2*b].
            # M[0,1] corresponds to C[0, 2*b+1].
            # M[1,0] corresponds to C[1, 2*b].
            # M[1,1] corresponds to C[1, 2*b+1].
            
            # w_contrib[r] is [M00, M01, M10, M11].
            
            # M00: i=0, k_local=0 -> k_global=2*b. W_idx = 2*b*2 + 0.
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            
            # M01: i=0, k_local=1 -> k_global=2*b+1. W_idx = (2*b+1)*2 + 0.
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            
            # M10: i=1, k_local=0 -> k_global=2*b. W_idx = 2*b*2 + 1.
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            
            # M11: i=1, k_local=1 -> k_global=2*b+1. W_idx = (2*b+1)*2 + 1.
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}
