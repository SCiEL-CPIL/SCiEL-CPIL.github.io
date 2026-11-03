"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Rank calculation:
    # Strip j=4: Rank 1
    # Core j=0..3: 
    #   Partition p=6 into 3 blocks of 2.
    #   Each block requires 4 Strassen 2x2x2 blocks.
    #   Total 12 Strassen blocks.
    #   Rank = 1 + 12 * 7 = 85.
    
    R = 85
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    def add_strassen(i_base, j_base, k_base, r_offset):
        """
        Adds Strassen decomposition for 2x2x2 block.
        i_base: start row of A (0 or 2)
        j_base: start col of A (0 or 2)
        k_base: start col of B (0, 2, or 4)
        r_offset: start column in U, V, W
        """
        # Local indices 0, 1
        # Global i = i_base + li
        # Global j = j_base + lj
        # Global k = k_base + lk
        
        # Strassen products M1..M7
        # M1 = (A00+A11)(B00+B11)
        # A part: A00, A11
        # B part: B00, B11
        # C part: C00 (+1), C11 (+1)
        
        def set_U(li, lj, val):
            U[(i_base + li) * m + (j_base + lj), r_offset] += val
        def set_V(lj, lk, val):
            V[(j_base + lj) * p + (k_base + lk), r_offset] += val
        def set_W(lk, li, val):
            W[(k_base + lk) * n + (i_base + li), r_offset] += val
            
        # M1
        set_U(0, 0, 1); set_U(1, 1, 1)
        set_V(0, 0, 1); set_V(1, 1, 1)
        set_W(0, 0, 1); set_W(1, 1, 1)
        
        r_offset += 1
        
        # M2 = (A10+A11)B00
        # A: A10, A11
        # B: B00
        # C: C10 (+1)
        set_U(1, 0, 1); set_U(1, 1, 1)
        set_V(0, 0, 1)
        set_W(0, 1, 1)
        
        r_offset += 1
        
        # M3 = A00(B10-B11)
        # A: A00
        # B: B10, B11
        # C: C01 (+1)
        set_U(0, 0, 1)
        set_V(1, 0, 1); set_V(1, 1, -1)
        set_W(1, 0, 1)
        
        r_offset += 1
        
        # M4 = A11(B00-B10)
        # A: A11
        # B: B00, B10
        # C: C00 (+1), C10 (+1)
        set_U(1, 1, 1)
        set_V(0, 0, 1); set_V(1, 0, -1)
        set_W(0, 0, 1); set_W(0, 1, 1)
        
        r_offset += 1
        
        # M5 = (A00+A01)B11
        # A: A00, A01
        # B: B11
        # C: C00 (-1), C01 (+1)
        set_U(0, 0, 1); set_U(0, 1, 1)
        set_V(1, 1, 1)
        set_W(0, 0, -1); set_W(1, 0, 1)
        
        r_offset += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A: A10, A00
        # B: B00, B01
        # C: C11 (+1)
        set_U(1, 0, 1); set_U(0, 0, -1)
        set_V(0, 0, 1); set_V(0, 1, 1)
        set_W(1, 1, 1)
        
        r_offset += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A: A01, A11
        # B: B10, B11
        # C: C00 (+1)
        set_U(0, 1, 1); set_U(1, 1, -1)
        set_V(1, 0, 1); set_V(1, 1, 1)
        set_W(0, 0, 1)
        
        r_offset += 1
        
        return r_offset

    # 1. Handle Strip j=4 (Rank 1)
    r = 0
    for i in range(n):
        U[i * m + 4, r] = 1.0
    for k in range(p):
        V[4 * p + k, r] = 1.0
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1.0
    r += 1
    
    # 2. Handle Core j=0..3
    # Partition k into 3 blocks of 2
    for k_base in [0, 2, 4]:
        # Partition i into 2 blocks of 2
        for i_base in [0, 2]:
            # Term 1: A[i_base:i_base+2, 0:2] * B[0:2, k_base:k_base+2]
            r = add_strassen(i_base, 0, k_base, r)
            # Term 2: A[i_base:i_base+2, 2:4] * B[2:4, k_base:k_base+2]
            r = add_strassen(i_base, 2, k_base, r)
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
