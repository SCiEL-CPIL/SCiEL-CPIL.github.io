"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm on 2x2 blocks
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps definitions for 2x2 matrices
    # U rows: 00, 01, 10, 11
    # V rows: 00, 01, 10, 11 (within block)
    # W rows: 00, 01, 10, 11 (within block)
    
    # s: 0..6
    # U_coeffs[s] is a list of (row_idx, val)
    # V_coeffs[s] is a list of (row_idx_offset, val) where row_idx is relative to block start
    # W_coeffs[s] is a list of (row_idx_offset, val) where row_idx is relative to block start
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices in block: 00->0, 01->1, 10->2, 11->3 (relative)
    # C indices in block: 00->0, 01->1, 10->2, 11->3 (relative)
    
    # M1 = (A00+A11)(B00+B01+B10+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_def = [
        [(0, 1), (3, 1)],       # M1
        [(2, 1), (3, 1)],       # M2
        [(0, 1)],               # M3
        [(3, 1)],               # M4
        [(0, 1), (1, 1)],       # M5
        [(2, 1), (0, -1)],      # M6
        [(1, 1), (3, -1)]       # M7
    ]
    
    # V indices relative to block start (j*p + k)
    # Block b covers k = 2b, 2b+1
    # B00 is j=0, k=2b -> idx 2b
    # B01 is j=0, k=2b+1 -> idx 2b+1
    # B10 is j=1, k=2b -> idx 14+2b
    # B11 is j=1, k=2b+1 -> idx 14+2b+1
    
    V_def = [
        [(0, 1), (1, 1), (14, 1), (15, 1)], # M1: B00, B01, B10, B11 (offsets from 2b)
        [(0, 1)],                            # M2: B00
        [(1, 1), (15, -1)],                  # M3: B01 - B11
        [(14, 1), (0, -1)],                  # M4: B10 - B00
        [(15, 1)],                           # M5: B11
        [(0, 1), (1, 1)],                    # M6: B00 + B01
        [(14, 1), (15, 1)]                   # M7: B10 + B11
    ]
    
    # W indices relative to block start (k*n + i)
    # Block b covers k = 2b, 2b+1
    # C00 is k=2b, i=0 -> idx 2*2b + 0 = 4b
    # C01 is k=2b+1, i=0 -> idx 2*(2b+1) + 0 = 4b+2
    # C10 is k=2b, i=1 -> idx 2*2b + 1 = 4b+1
    # C11 is k=2b+1, i=1 -> idx 2*(2b+1) + 1 = 4b+3
    
    # Offsets from 4b:
    # C00 -> 0
    # C01 -> 2
    # C10 -> 1
    # C11 -> 3
    
    W_def = [
        [(0, 1), (3, 1)],   # M1: C00 +1, C11 +1
        [(1, 1), (3, -1)],  # M2: C10 +1, C11 -1
        [(2, 1), (3, 1)],   # M3: C01 +1, C11 +1
        [(0, 1), (1, 1)],   # M4: C00 +1, C10 +1
        [(0, -1), (2, 1)],  # M5: C00 -1, C01 +1
        [(3, 1)],           # M6: C11 +1
        [(0, 1)]            # M7: C00 +1
    ]
    
    for b in range(7):
        base_v = 2 * b
        base_w = 4 * b
        for s in range(7):
            r = 7 * b + s
            
            for u_idx, val in U_def[s]:
                U[u_idx][r] = val
                
            for v_offset, val in V_def[s]:
                V[base_v + v_offset][r] = val
                
            for w_offset, val in W_def[s]:
                W[base_w + w_offset][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
