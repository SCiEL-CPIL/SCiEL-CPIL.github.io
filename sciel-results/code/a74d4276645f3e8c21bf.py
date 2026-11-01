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
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        for s in range(7):
            c = base + s
            
            if s == 0:
                # M1 = (A00+A11)*(B00+B11)
                U[0, c] += 1   # A[0,0]
                U[3, c] += 1   # A[1,1]
                V[0*p + k0, c] += 1   # B[0, k0]
                V[1*p + k1, c] += 1   # B[1, k1]
                # C00 += M1, C11 += M1
                W[k0*n + 0, c] += 1   # C[0, k0]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 1:
                # M2 = (A10+A11)*B00
                U[2, c] += 1   # A[1,0]
                U[3, c] += 1   # A[1,1]
                V[0*p + k0, c] += 1   # B[0, k0]
                # C10 += M2, C11 -= M2
                W[k0*n + 1, c] += 1   # C[1, k0]
                W[k1*n + 1, c] += -1  # C[1, k1]
                
            elif s == 2:
                # M3 = A00*(B01-B11)
                U[0, c] += 1   # A[0,0]
                V[0*p + k1, c] += 1    # B[0, k1]
                V[1*p + k1, c] += -1   # B[1, k1]
                # C01 += M3, C11 += M3
                W[k1*n + 0, c] += 1   # C[0, k1]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 3:
                # M4 = A11*(B10-B00)
                U[3, c] += 1   # A[1,1]
                V[1*p + k0, c] += 1    # B[1, k0]
                V[0*p + k0, c] += -1   # B[0, k0]
                # C00 += M4, C10 += M4
                W[k0*n + 0, c] += 1   # C[0, k0]
                W[k0*n + 1, c] += 1   # C[1, k0]
                
            elif s == 4:
                # M5 = (A00+A01)*B11
                U[0, c] += 1   # A[0,0]
                U[1, c] += 1   # A[0,1]
                V[1*p + k1, c] += 1   # B[1, k1]
                # C00 -= M5, C01 += M5
                W[k0*n + 0, c] += -1  # C[0, k0]
                W[k1*n + 0, c] += 1   # C[0, k1]
                
            elif s == 5:
                # M6 = (A10-A00)*(B00+B01)
                U[2, c] += 1   # A[1,0]
                U[0, c] += -1  # A[0,0]
                V[0*p + k0, c] += 1    # B[0, k0]
                V[0*p + k1, c] += 1    # B[0, k1]
                # C11 += M6
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 6:
                # M7 = (A01-A11)*(B10+B11)
                U[1, c] += 1   # A[0,1]
                U[3, c] += -1  # A[1,1]
                V[1*p + k0, c] += 1    # B[1, k0]
                V[1*p + k1, c] += 1    # B[1, k1]
                # C00 += M7
                W[k0*n + 0, c] += 1   # C[0, k0]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
