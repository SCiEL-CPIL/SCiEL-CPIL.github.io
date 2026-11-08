import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Build the rank 39 solution explicitly
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1
        
    def add_strassen_block(k_start):
        # Handles 2x2 x 2 block at columns k_start, k_start+1
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # Strassen 7 terms
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        add_term([0, 4], [b00, b11], [c00, c11])
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        # M3 = A00(B01 - B11) -> C01 += M3, C11 -= M3
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        # M4 = A11(B10 - B00) -> C00 -= M4, C10 += M4
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        # M5 = (A00 + A01)B11 -> C00 += M5, C01 += M5
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1 # Wait, Strassen M5 is (A00+A01)B11. C00 += M5, C01 += M5.
        # Correction: Standard Strassen:
        # M5 = (A00 + A01) B11 -> C00 += M5, C01 += M5
        # My previous line added to c00 and c01. But I set W[c00] = -1?
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # Let's restart the Strassen block definition to be sure.
        # I will use the standard 7 products.
        
    # The previous code's Strassen implementation might have had sign errors.
    # I will use a known correct Strassen implementation for 2x2x2.
    
    # Reset U, V, W
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_222(k_start):
        nonlocal col
        # A is 2x2 (rows 0,1; cols k_start, k_start+1)
        # B is 2x2 (rows 0,1; cols k_start, k_start+1)
        # C is 2x2 (rows 0,1; cols k_start, k_start+1)
        
        # Indices in U: i*m + j. Here i=0,1. j=0,1.
        # But wait, the block is at j=0,1.
        # U indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4.
        # V indices: j*p+k. j=0,1. k=k_start, k_start+1.
        # V indices: 0*7+k_start, 0*7+k_start+1, 1*7+k_start, 1*7+k_start+1.
        # W indices: k*n+i. k=k_start, k_start+1. i=0,1.
        # W indices: k_start*2+0, k_start*2+1, (k_start+1)*2+0, (k_start+1)*2+1.
        
        u00 = 0; u01 = 1; u10 = 3; u11 = 4
        v00 = 0*7+k_start; v01 = 0*7+k_start+1
        v10 = 1*7+k_start; v11 = 1*7+k_start+1
        w00 = k_start*2+0; w10 = k_start*2+1
        w01 = (k_start+1)*2+0; w11 = (k_start+1)*2+1
        
        # M1 = (A00+A11)(B00+B11)
        add_term([u00, u11], [v00, v11], [w00, w11])
        # M2 = (A10+A11)B00
        add_term([u10, u11], [v00], [w10, w11])
        W[w11][col-1] = -1
        # M3 = A00(B01-B11)
        add_term([u00], [v01, v11], [w01, w11])
        V[v11][col-1] = -1
        # M4 = A11(B10-B00)
        add_term([u11], [v10, v00], [w00, w10])
        V[v00][col-1] = -1
        # M5 = (A00+A01)B11
        add_term([u00, u01], [v11], [w00, w01])
        W[w00][col-1] = -1 # C00 = M1 + M4 - M5 + M7. So M5 is subtracted.
        # M6 = (A10-A00)(B00+B01)
        add_term([u10, u00], [v00, v01], [w10, w01])
        U[u00][col-1] = -1
        # M7 = (A01-A11)(B10+B11)
        add_term([u01, u11], [v10, v11], [w10, w11])
        U[u11][col-1] = -1
        
    # Apply Strassen to k=0,1; k=2,3; k=4,5
    add_strassen_222(0)
    add_strassen_222(2)
    add_strassen_222(4)
    
    # Handle k=6 (2x2 x 1)
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    # 4 terms
    u00 = 0; u01 = 1; u10 = 3; u11 = 4
    v06 = 0*7+6; v16 = 1*7+6
    w06 = 6*2+0; w16 = 6*2+1
    
    add_term([u00], [v06], [w06])
    add_term([u01], [v16], [w06])
    add_term([u10], [v06], [w16])
    add_term([u11], [v16], [w16])
    
    # Handle j=2 (2x1 x 7)
    # C0k = A02 B2k
    # C1k = A12 B2k
    # 14 terms
    for k in range(7):
        u02 = 0*3+2
        u12 = 1*3+2
        v2k = 2*7+k
        w0k = k*2+0
        w1k = k*2+1
        
        add_term([u02], [v2k], [w0k])
        add_term([u12], [v2k], [w1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
