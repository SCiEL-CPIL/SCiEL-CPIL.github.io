"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 7 Strassen rank-7 terms per 2x2 block applied to 6 column pairs.
    
    Decomposition satisfies:
      For all i in [0,2), j in [0,2), k in [0,12):
        sum_c U[i*2+j, c] * V[j*12+k, c] * W[k*2+i, c] = 1
    """
    t_start = time.time()
    deadline = t_start + 100  # 50s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen terms for 2x2x2:
    # u: (idx, val) for A[i,j] with idx = i*2+j
    # v: (j, k_local, val) for B[j,k_local] 
    # w: (k_local, i, val) for C[k_local,i]
    #
    # M1 = (A00+A11)*(B00+B11)  ->  C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00        ->  C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11)        ->  C01 += M3, C11 += M3
    # M4 = A11*(B10-B00)        ->  C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11        ->  C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01)  ->  C11 += M6
    # M7 = (A01-A11)*(B10+B11)  ->  C00 += M7
    
    # Verify Strassen algebraically:
    # C00 = M1 + M4 - M5 + M7
    #   = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 
    #     + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #   = A00*B00 + A01*B10  ✓
    #
    # C01 = M3 + M5
    #   = A00(B01-B11) + (A00+A01)B11
    #   = A00*B01 - A00*B11 + A00*B11 + A01*B11
    #   = A00*B01 + A01*B11  ✓
    #
    # C10 = M2 + M4
    #   = (A10+A11)B00 + A11(B10-B00)
    #   = A10*B00 + A11*B00 + A11*B10 - A11*B00
    #   = A10*B00 + A11*B10  ✓
    #
    # C11 = M1 - M2 + M3 + M6
    #   = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11
    #     + A10*B00 + A10*B01 - A00*B00 - A00*B01
    #   = A10*B01 + A11*B11  ✓
    
    terms = [
        # s=0: M1 = (A00+A11)*(B00+B11)
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1), (1, 1, 1)]
        },
        # s=1: M2 = (A10+A11)*B00
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 1, 1), (1, 1, -1)]
        },
        # s=2: M3 = A00*(B01-B11)
        {
            'u': [(0, 1)],
            'v': [(0, 1, 1), (1, 1, -1)],
            'w': [(1, 0, 1), (1, 1, 1)]
        },
        # s=3: M4 = A11*(B10-B00)
        {
            'u': [(3, 1)],
            'v': [(1, 0, 1), (0, 0, -1)],
            'w': [(0, 0, 1), (0, 1, 1)]
        },
        # s=4: M5 = (A00+A01)*B11
        {
            'u': [(0, 1), (1, 1)],
            'v': [(1, 1, 1)],
            'w': [(0, 0, -1), (1, 0, 1)]
        },
        # s=5: M6 = (A10-A00)*(B00+B01)
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(1, 1, 1)]
        },
        # s=6: M7 = (A01-A11)*(B10+B11)
        {
            'u': [(1, 1), (3, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1)]
        }
    ]
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        for s in range(7):
            c = base + s
            term = terms[s]
            
            # Update U
            for u_idx, val in term['u']:
                U[u_idx, c] += val
                
            # Update V
            for j, k_local, val in term['v']:
                global_k = k0 + k_local
                V[j * p + global_k, c] += val
                
            # Update W
            for k_local, i, val in term['w']:
                global_k = k0 + k_local
                W[global_k * n + i, c] += val

    # Convert to lists as expected by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
