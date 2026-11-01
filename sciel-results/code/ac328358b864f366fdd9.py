import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # C = A * B
    # 7 products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U[i*m + j, r] corresponds to coefficient of A_ij in product t
    # V[j*p + k, r] corresponds to coefficient of B_jk in product t
    # W[k*n + i, r] corresponds to coefficient of product t in C_ik
    
    terms = [
        # t=0: P1
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # t=1: P2
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}}, # Wait, C10 uses P2, C11 uses -P2
        # Correction:
        # C10 = P2 + P4. So W[1,0] += 1.
        # C11 = P1 - P2 + P3 + P6. So W[1,1] += -1.
        # Let's re-verify the W mapping in the previous code.
        # Previous code:
        # t=1: W: {(0,1): 1, (1,1): -1} -> This implies C01 gets P2? No.
        # C01 = P3 + P5. P2 is not in C01.
        # C10 = P2 + P4.
        # C11 = P1 - P2 + P3 + P6.
        # So for P2 (t=1):
        # W[1, 0] += 1 (C10)
        # W[1, 1] += -1 (C11)
        # The previous code had: 'W': {(0,1): 1, (1,1): -1}
        # Index (0,1) is C_{0,1}? Or C_{1,0}?
        # W index is k*n + i.
        # k=0, i=1 -> W[1]. This is C_{1,0}? No, W[k*n+i].
        # If k=0, i=1, index is 1. This corresponds to C_{1,0} if i is row, k is col?
        # Standard: C[i][k].
        # W[k*n + i].
        # If n=2.
        # k=0, i=0 -> W[0] -> C[0,0]
        # k=0, i=1 -> W[1] -> C[1,0]
        # k=1, i=0 -> W[2] -> C[0,1]
        # k=1, i=1 -> W[3] -> C[1,1]
        
        # Let's check the previous code's W indices again.
        # t=1: 'W': {(0,1): 1, (1,1): -1}
        # (k_loc, i) -> (0,1) and (1,1).
        # k_loc=0, i=1 -> C[1, 0]?
        # k_loc=1, i=1 -> C[1, 1]?
        # So it adds to C[1,0] and C[1,1].
        # This matches C10 += P2 and C11 -= P2.
        # So the previous code was correct.
        
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}},
        # t=2: P3
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
        # t=3: P4
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        # t=4: P5
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
        # t=5: P6
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # t=6: P7
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}}
    ]
    
    num_blocks = p // 2
    for b in range(num_blocks):
        for t_idx in range(7):
            r = b * 7 + t_idx
            term = terms[t_idx]
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            for (j, k_loc), val in term['V'].items():
                k_glob = b * 2 + k_loc
                V[j * p + k_glob][r] = val
            for (k_loc, i), val in term['W'].items():
                k_glob = b * 2 + k_loc
                W[k_glob * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
