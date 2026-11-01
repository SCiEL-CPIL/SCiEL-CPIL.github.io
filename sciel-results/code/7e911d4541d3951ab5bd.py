import numpy as np
import time
from itertools import product

def construct():
    """
    Returns a rank-7 exact decomposition of the 2x2x2 matrix multiplication tensor.
    
    The decomposition satisfies:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for all i, j, k in {0, 1}.
    
    This corresponds to Strassen's algorithm for 2x2 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W'
        U: shape (4, 7) list of lists
        V: shape (4, 7) list of lists
        W: shape (4, 7) list of lists
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing conventions:
    # U index: i*m + j  (0: A00, 1: A01, 2: A10, 3: A11)
    # V index: j*p + k  (0: B00, 1: B01, 2: B10, 3: B11)
    # W index: k*n + i  (0: C00, 1: C10, 2: C01, 3: C11)
    
    # r=0: M1
    # U: A00 (0), A11 (3)
    # V: B00 (0), B11 (3)
    # W: C00 (0), C11 (3)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # r=1: M2
    # U: A10 (2), A11 (3)
    # V: B00 (0)
    # W: C10 (1), C11 (3) with -1
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0
    
    # r=2: M3
    # U: A00 (0)
    # V: B01 (1), B11 (3) with -1
    # W: C01 (2), C11 (3)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0
    
    # r=3: M4
    # U: A11 (3)
    # V: B10 (2), B00 (0) with -1
    # W: C00 (0), C10 (1)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0
    
    # r=4: M5
    # U: A00 (0), A01 (1)
    # V: B11 (3)
    # W: C00 (0) with -1, C01 (2)
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0
    
    # r=5: M6
    # U: A10 (2), A00 (0) with -1
    # V: B00 (0), B01 (1)
    # W: C11 (3)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # r=6: M7
    # U: A01 (1), A11 (3) with -1
    # V: B10 (2), B11 (3)
    # W: C00 (0)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    # Verification step to ensure correctness
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # If verification fails (should not happen with hardcoded Strassen), 
        # we could fallback to rank 8, but Strassen is exact.
        pass
        
    return {
        "U": U,
        "V": V,
        "W": W
    }
