import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We construct rank 44 decomposition.
    # Split m=3 into j=0,1 (Strassen part) and j=2 (Trivial part).
    # Split p=8 into 4 blocks of k=0,1; 2,3; 4,5; 6,7.
    
    # Part 1: Strassen for 2x2x2 blocks (4 blocks * 7 terms = 28 terms)
    # Part 2: Trivial for 2x1x8 (16 terms)
    
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..1
    # C indices: k in 0..1, i in 0..1
    
    # Strassen multipliers for A (2x2)
    # P1: (a11+a22)
    # P2: (a21+a22)
    # P3: a11
    # P4: a22
    # P5: (a11+a12) -> Wait, standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Note: Indices in 0-based for code.
    # A[0,0]=a11, A[0,1]=a12, A[1,0]=a21, A[1,1]=a22.
    
    strassen_A = [
        [1, 1, 0, 0, 0, 0, 0], # M1: a00 + a11? No. M1 uses A11+A22.
        # Let's be precise.
        # M1: A[0,0] + A[1,1]
        # M2: A[1,0] + A[1,1]
        # M3: A[0,0]
        # M4: A[1,1]
        # M5: A[0,0] + A[0,1]
        # M6: A[1,0] - A[0,0]
        # M7: A[0,1] - A[1,1]
        [1, 1, 0, 0, 0, 0, 0], # M1: A00 + A11
        [0, 0, 1, 1, 0, 0, 0], # M2: A10 + A11
        [1, 0, 0, 0, 0, 0, 0], # M3: A00
        [0, 0, 0, 1, 0, 0, 0], # M4: A11
        [1, 1, 0, 0, 0, 0, 0], # M5: A00 + A01
        [1, -1, 0, 0, 0, 0, 0], # M6: A10 - A00
        [0, 1, 0, -1, 0, 0, 0]  # M7: A01 - A11
    ]
    
    strassen_B = [
        # M1: B00 + B11
        # M2: B00
        # M3: B01 - B11
        # M4: B10 - B00
        # M5: B11
        # M6: B00 + B01
        # M7: B10 + B11
        [1, 1, 0, 0, 0, 0, 0], # M1: B00 + B11
        [1, 0, 0, 0, 0, 0, 0], # M2: B00
        [0, -1, 1, 0, 0, 0, 0], # M3: B01 - B11
        [-1, 1, 0, 0, 0, 0, 0], # M4: B10 - B00
        [0, 0, 0, 1, 0, 0, 0], # M5: B11
        [1, 0, 1, 0, 0, 0, 0], # M6: B00 + B01
        [0, 1, 0, 1, 0, 0, 0]  # M7: B10 + B11
    ]
    
    strassen_C = [
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Vectors for C (index k*n+i)
        # C00 (k=0,i=0): idx 0
        # C01 (k=0,i=1): idx 1
        # C10 (k=1,i=0): idx 2
        # C11 (k=1,i=1): idx 3
        [1, 0, 0, 1, -1, 1, 0], # M1: C00, M4: C00, M5: -C00, M7: C00
        [0, 1, 0, 0, 1, 0, 0],   # M3: C01, M5: C01
        [0, 0, 1, 1, 0, 0, 0],   # M2: C10, M4: C10
        [1, 0, 0, 1, -1, 0, 1]   # M1: C11, M2: -C11, M3: C11, M6: C11
    ]
    # Note: The above strassen_C is transposed or misaligned.
    # Let's define W factors for each term r in 0..6.
    # W_r is a vector of size p*n = 4.
    # For term r, W_r[k*n + i] is the coefficient.
    
    W_strassen = [
        [1, 0, 0, 1], # Term 0 (M1): C00 + C11? No.
        # M1 contributes to C00 and C11.
        # C00 = M1 + ...
        # C11 = M1 + ...
        # So W_0 = [1, 0, 0, 1] (C00=1, C01=0, C10=0, C11=1)
        
        # M2: C10 + ... C11 - ...
        # C10 = M2 + ...
        # C11 = ... - M2 + ...
        # W_1 = [0, 0, 1, -1]
        
        # M3: C01 + ... C11 + ...
        # C01 = M3 + ...
        # C11 = ... + M3 + ...
        # W_2 = [0, 1, 0, 1]
        
        # M4: C00 + ... C10 + ...
        # C00 = ... + M4 ...
        # C10 = ... + M4 ...
        # W_3 = [1, 0, 1, 0]
        
        # M5: C00 - ... C01 + ...
        # C00 = ... - M5 ...
        # C01 = ... + M5 ...
        # W_4 = [-1, 1, 0, 0]
        
        # M6: C11 + ...
        # C11 = ... + M6 ...
        # W_5 = [0, 0, 0, 1]
        
        # M7: C00 + ...
        # C00 = ... + M7 ...
        # W_6 = [1, 0, 0, 0]
    ]
    
    # Verify Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My W vectors:
    # W_0 (M1): [1, 0, 0, 1] -> C00+1, C11+1. Correct.
    # W_1 (M2): [0, 0, 1, -1] -> C10+1, C11-1. Correct.
    # W_2 (M3): [0, 1, 0, 1] -> C01+1, C11+1. Correct.
    # W_3 (M4): [1, 0, 1, 0] -> C00+1, C10+1. Correct.
    # W_4 (M5): [-1, 1, 0, 0] -> C00-1, C01+1. Correct.
    # W_5 (M6): [0, 0, 0, 1] -> C11+1. Correct.
    # W_6 (M7): [1, 0, 0, 0] -> C00+1. Correct.
    
    # Now populate U, V, W for the 4 blocks.
    # Block t (0..3) uses k offsets 2t, 2t+1.
    for t in range(4):
        k_offset = t * 2
        for r in range(7):
            # U factor
            # j in 0,1. i in 0,1.
            # U[i*m + j]
            # strassen_A[r] corresponds to linear combo of A[0,0], A[0,1], A[1,0], A[1,1]
            # A[0,0] -> idx 0*3+0 = 0
            # A[0,1] -> idx 0*3+1 = 1
            # A[1,0] -> idx 1*3+0 = 3
            # A[1,1] -> idx 1*3+1 = 4
            
            # strassen_A[r] is [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
            c00, c01, c10, c11 = strassen_A[r]
            if c00 != 0: U[0][col] = c00
            if c01 != 0: U[1][col] = c01
            if c10 != 0: U[3][col] = c10
            if c11 != 0: U[4][col] = c11
            
            # V factor
            # j in 0,1. k in 0,1 (relative to block).
            # V[j*p + k]
            # j=0, k=2t -> idx 0*8 + 2t
            # j=0, k=2t+1 -> idx 0*8 + 2t+1
            # j=1, k=2t -> idx 1*8 + 2t
            # j=1, k=2t+1 -> idx 1*8 + 2t+1
            
            # strassen_B[r] is [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
            d00, d01, d10, d11 = strassen_B[r]
            if d00 != 0: V[0*8 + k_offset][col] = d00
            if d01 != 0: V[0*8 + k_offset + 1][col] = d01
            if d10 != 0: V[1*8 + k_offset][col] = d10
            if d11 != 0: V[1*8 + k_offset + 1][col] = d11
            
            # W factor
            # k in 0,1 (relative). i in 0,1.
            # W[k*n + i]
            # k=0, i=0 -> idx 0
            # k=0, i=1 -> idx 1
            # k=1, i=0 -> idx 2
            # k=1, i=1 -> idx 3
            
            # But we need to offset by k_offset.
            # The W vector in strassen_C was defined for local k=0,1.
            # w_vec = W_strassen[r]
            # Local idx 0 (k=0,i=0) -> Global k=2t, i=0 -> idx (2t)*2 + 0
            # Local idx 1 (k=0,i=1) -> Global k=2t, i=1 -> idx (2t)*2 + 1
            # Local idx 2 (k=1,i=0) -> Global k=2t+1, i=0 -> idx (2t+1)*2 + 0
            # Local idx 3 (k=1,i=1) -> Global k=2t+1, i=1 -> idx (2t+1)*2 + 1
            
            w_vec = W_strassen[r]
            for local_idx, val in enumerate(w_vec):
                if val != 0:
                    # local_idx: 0->k0i0, 1->k0i1, 2->k1i0, 3->k1i1
                    if local_idx == 0:
                        g_idx = (k_offset) * 2 + 0
                    elif local_idx == 1:
                        g_idx = (k_offset) * 2 + 1
                    elif local_idx == 2:
                        g_idx = (k_offset + 1) * 2 + 0
                    elif local_idx == 3:
                        g_idx = (k_offset + 1) * 2 + 1
                    W[g_idx][col] = val
            
            col += 1
    
    # Part 2: Trivial 2x1x8
    # j=2.
    # U[2*3 + 2] = U[8] = 1 for all 16 terms? No, U is specific to the term.
    # For term r in 0..15:
    #   i in 0,1. k in 0..7.
    #   U[i*3 + 2] = 1
    #   V[2*8 + k] = 1
    #   W[k*2 + i] = 1
    
    for i in range(2):
        for k in range(8):
            U[i*3 + 2][col] = 1
            V[2*8 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
