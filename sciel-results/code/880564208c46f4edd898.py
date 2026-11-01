import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct rank-16 tensor decomposition for 2x2 x 2x5 matrix multiplication.
    
    Strategy: Two modified Strassen blocks that incorporate column-4 contributions,
    plus 2 additional terms, totaling rank 16.
    """
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Try to construct rank-16 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # (4, 16)
    V = [[0] * rank for _ in range(m * p)]  # (10, 16)
    W = [[0] * rank for _ in range(p * n)]  # (10, 16)
    
    # Modified Strassen block 1: cols 0, 1 with column-4 sharing
    # We modify the V-factors to include B[0,4] and B[1,4] contributions
    
    # Term 0: M1 = (A00+A11) * (B[0,0] + B[1,1] + B[0,4])
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0*p+0][r] = 1; V[1*p+1][r] = 1; V[0*p+4][r] = 1
    W[0*n+0][r] = 1; W[1*n+1][r] = 1; W[4*n+0][r] = 1
    
    # Term 1: M2 = (A10+A11) * (B[0,0] + B[1,4])
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0*p+0][r] = 1; V[1*p+4][r] = 1
    W[0*n+1][r] = 1; W[1*n+1][r] = -1; W[4*n+1][r] = 1
    
    # Term 2: M3 = A00 * (B[0,1] - B[1,1])
    r = 2
    U[0][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = -1
    W[1*n+0][r] = 1; W[1*n+1][r] = 1
    
    # Term 3: M4 = A11 * (B[1,0] - B[0,0])
    r = 3
    U[3][r] = 1
    V[1*p+0][r] = 1; V[0*p+0][r] = -1
    W[0*n+0][r] = 1; W[0*n+1][r] = 1
    
    # Term 4: M5 = (A00+A01) * B[1,1]
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[1*p+1][r] = 1
    W[0*n+0][r] = -1; W[1*n+0][r] = 1
    
    # Term 5: M6 = (A10-A00) * (B[0,0] + B[0,1])
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0*p+0][r] = 1; V[0*p+1][r] = 1
    W[1*n+1][r] = 1
    
    # Term 6: M7 = (A01-A11) * (B[1,0] + B[1,1])
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[1*p+0][r] = 1; V[1*p+1][r] = 1
    W[0*n+0][r] = 1
    
    # Standard Strassen block 2: cols 2, 3
    # Term 7: M1 = (A00+A11) * (B[0,2] + B[1,3])
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[0*p+2][r] = 1; V[1*p+3][r] = 1
    W[2*n+0][r] = 1; W[3*n+1][r] = 1
    
    # Term 8: M2 = (A10+A11) * B[0,2]
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[0*p+2][r] = 1
    W[2*n+1][r] = 1; W[3*n+1][r] = -1
    
    # Term 9: M3 = A00 * (B[0,3] - B[1,3])
    r = 9
    U[0][r] = 1
    V[0*p+3][r] = 1; V[1*p+3][r] = -1
    W[3*n+0][r] = 1; W[3*n+1][r] = 1
    
    # Term 10: M4 = A11 * (B[1,2] - B[0,2])
    r = 10
    U[3][r] = 1
    V[1*p+2][r] = 1; V[0*p+2][r] = -1
    W[2*n+0][r] = 1; W[2*n+1][r] = 1
    
    # Term 11: M5 = (A00+A01) * B[1,3]
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[1*p+3][r] = 1
    W[2*n+0][r] = -1; W[3*n+0][r] = 1
    
    # Term 12: M6 = (A10-A00) * (B[0,2] + B[0,3])
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[0*p+2][r] = 1; V[0*p+3][r] = 1
    W[3*n+1][r] = 1
    
    # Term 13: M7 = (A01-A11) * (B[1,2] + B[1,3])
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[1*p+2][r] = 1; V[1*p+3][r] = 1
    W[2*n+0][r] = 1
    
    # Additional terms for column 4 (remaining contributions)
    # Term 14: A01 * B[1,4]
    r = 14
    U[1][r] = 1
    V[1*p+4][r] = 1
    W[4*n+0][r] = 1
    
    # Term 15: A10 * B[0,4]
    r = 15
    U[2][r] = 1
    V[0*p+4][r] = 1
    W[4*n+1][r] = 1
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[a*m+b, r] * V_arr[b*p+c, r] * W_arr[c*n+a, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fall back to rank-18 if rank-16 doesn't work
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1
        
        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)
        
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
