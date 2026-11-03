import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2 blocks
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float) # 8 x 56
    V = np.zeros((m * p, rank), dtype=float) # 32 x 56
    W = np.zeros((p * n, rank), dtype=float) # 16 x 56
    
    # We split the middle dimension m=4 into two groups of 2: [0,1] and [2,3]
    # We split the output dimension p=8 into four groups of 2: [0,1], [2,3], [4,5], [6,7]
    # For each group pair (mid_group, out_group), we use Strassen 2x2x2
    
    # Strassen coefficients for 2x2 multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the Strassen factors for a single 2x2x2 block
    # U factors (A side):
    # s0: A00 + A11
    # s1: A10 + A11
    # s2: A00
    # s3: A11
    # s4: A00 + A01
    # s5: A10 - A00
    # s6: A01 - A11
    
    # V factors (B side):
    # s0: B00 + B11
    # s1: B00
    # s2: B01 - B11
    # s3: B10 - B00
    # s4: B11
    # s5: B00 + B01
    # s6: B10 + B11
    
    # W factors (C side):
    # s0: C00 += 1, C11 += 1
    # s1: C10 += 1, C11 -= 1
    # s2: C01 += 1, C11 += 1
    # s3: C00 += 1, C10 += 1
    # s4: C00 -= 1, C01 += 1
    # s5: C11 += 1
    # s6: C00 += 1
    
    t = 0
    # mid_group 0 corresponds to j=0,1. mid_group 1 corresponds to j=2,3.
    for mid_g in range(2):
        j_offset = mid_g * 2
        # out_group 0..3 corresponds to k=0,1; 2,3; 4,5; 6,7
        for out_g in range(4):
            k_offset = out_g * 2
            
            for s in range(7):
                # Update U
                if s == 0:
                    U[0 * m + j_offset, t] += 1
                    U[1 * m + j_offset + 1, t] += 1
                elif s == 1:
                    U[1 * m + j_offset, t] += 1
                    U[1 * m + j_offset + 1, t] += 1
                elif s == 2:
                    U[0 * m + j_offset, t] += 1
                elif s == 3:
                    U[1 * m + j_offset + 1, t] += 1
                elif s == 4:
                    U[0 * m + j_offset, t] += 1
                    U[0 * m + j_offset + 1, t] += 1
                elif s == 5:
                    U[1 * m + j_offset, t] += 1
                    U[0 * m + j_offset, t] += -1
                elif s == 6:
                    U[0 * m + j_offset + 1, t] += 1
                    U[1 * m + j_offset + 1, t] += -1
                
                # Update V
                if s == 0:
                    V[j_offset * p + k_offset, t] += 1
                    V[(j_offset + 1) * p + k_offset + 1, t] += 1
                elif s == 1:
                    V[j_offset * p + k_offset, t] += 1
                elif s == 2:
                    V[j_offset * p + k_offset + 1, t] += 1
                    V[(j_offset + 1) * p + k_offset + 1, t] += -1
                elif s == 3:
                    V[(j_offset + 1) * p + k_offset, t] += 1
                    V[j_offset * p + k_offset, t] += -1
                elif s == 4:
                    V[(j_offset + 1) * p + k_offset + 1, t] += 1
                elif s == 5:
                    V[j_offset * p + k_offset, t] += 1
                    V[j_offset * p + k_offset + 1, t] += 1
                elif s == 6:
                    V[(j_offset + 1) * p + k_offset, t] += 1
                    V[(j_offset + 1) * p + k_offset + 1, t] += 1
                
                # Update W
                # C indices: W[k*n + i]
                # k_offset + 0 is even k, k_offset + 1 is odd k
                # i=0 is row 0, i=1 is row 1
                if s == 0:
                    W[k_offset * n + 0, t] += 1
                    W[(k_offset + 1) * n + 1, t] += 1
                elif s == 1:
                    W[k_offset * n + 1, t] += 1
                    W[(k_offset + 1) * n + 1, t] += -1
                elif s == 2:
                    W[(k_offset + 1) * n + 0, t] += 1
                    W[(k_offset + 1) * n + 1, t] += 1
                elif s == 3:
                    W[k_offset * n + 0, t] += 1
                    W[k_offset * n + 1, t] += 1
                elif s == 4:
                    W[k_offset * n + 0, t] += -1
                    W[(k_offset + 1) * n + 0, t] += 1
                elif s == 5:
                    W[(k_offset + 1) * n + 1, t] += 1
                elif s == 6:
                    W[k_offset * n + 0, t] += 1
                
                t += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
