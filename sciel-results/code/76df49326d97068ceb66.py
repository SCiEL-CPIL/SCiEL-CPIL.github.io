import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm on 6 disjoint 2x2 blocks.
    Rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS
    num_pairs = p // 2
    rank_base = num_pairs * 7
    
    U = np.zeros((n * m, rank_base))
    V = np.zeros((m * p, rank_base))
    W = np.zeros((p * n, rank_base))
    
    # Strassen coefficients for 2x2x2
    # U indices: 00, 01, 10, 11
    # V indices: 00, 01, 10, 11
    # W indices: 00, 10, 01, 11 (k*n + i) -> 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    for t in range(num_pairs):
        base = t * 7
        # k indices for this block: 2*t, 2*t+1
        k0 = 2*t
        k1 = 2*t+1
        
        # s=0: M1
        c = base
        U[0, c] = 1; U[3, c] = 1
        V[0*m+k0, c] = 1; V[1*m+k1, c] = 1
        W[k0*n+0, c] = 1; W[k1*n+1, c] = 1
        
        # s=1: M2
        c = base + 1
        U[2, c] = 1; U[3, c] = 1
        V[0*m+k0, c] = 1
        W[k0*n+1, c] = 1; W[k1*n+1, c] = -1
        
        # s=2: M3
        c = base + 2
        U[0, c] = 1
        V[1*m+k0, c] = 1; V[1*m+k1, c] = -1
        W[k1*n+0, c] = 1; W[k1*n+1, c] = 1
        
        # s=3: M4
        c = base + 3
        U[3, c] = 1
        V[1*m+k0, c] = 1; V[0*m+k0, c] = -1
        W[k0*n+0, c] = 1; W[k0*n+1, c] = 1
        
        # s=4: M5
        c = base + 4
        U[0, c] = 1; U[1, c] = 1
        V[1*m+k1, c] = 1
        W[k0*n+0, c] = -1; W[k1*n+0, c] = 1
        
        # s=5: M6
        c = base + 5
        U[2, c] = 1; U[0, c] = -1
        V[0*m+k0, c] = 1; V[1*m+k0, c] = 1
        W[k1*n+1, c] = 1
        
        # s=6: M7
        c = base + 6
        U[1, c] = 1; U[3, c] = -1
        V[1*m+k0, c] = 1; V[1*m+k1, c] = 1
        W[k0*n+0, c] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
