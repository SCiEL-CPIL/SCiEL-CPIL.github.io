import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Build 2x2x2 Strassen decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    # We will build the 4x4x4 decomposition by composing the 2x2x2 one.
    # For 4x4x4, A is 4x4, B is 4x4, C is 4x4.
    # We split A, B, C into 2x2 blocks of 2x2 matrices.
    # There are 7 Strassen terms M0..M6.
    # Each Mk is a 2x2 matrix.
    # Each Mk is computed by 2x2x2 Strassen.
    # So we have 7 * 7 = 49 scalar multiplications.
    
    # Let's define the linear combinations for A and B for the 7 Strassen steps.
    # A blocks: A00, A01, A10, A11 (indices 0..3 for 2x2 blocks)
    # B blocks: B00, B01, B10, B11
    
    # Step 0: (A00 + A11) * (B00 + B11)
    # Step 1: (A10 + A11) * B00
    # Step 2: A00 * (B01 - B11)
    # Step 3: A11 * (B10 - B00)
    # Step 4: (A00 + A01) * B11
    # Step 5: (A10 - A00) * (B00 + B01)
    # Step 6: (A01 - A11) * (B10 + B11)
    
    # We will create U, V, W for 4x4x4.
    # U size: 16 x 49
    # V size: 16 x 49
    # W size: 16 x 49
    
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    # Helper to add a 2x2x2 Strassen decomposition into the main matrices
    # offset_u, offset_v, offset_w are the starting indices in the 4x4x4 matrices
    # coeff_A: 4x4 matrix of coefficients for A blocks (flattened to 4x4)
    # coeff_B: 4x4 matrix of coefficients for B blocks
    # coeff_C: 4x4 matrix of coefficients for C blocks (output)
    
    # Actually, it's easier to just iterate.
    
    # Let's define the 2x2x2 Strassen U, V, W once.
    # U222, V222, W222 are 4x7.
    U222 = [[0]*7 for _ in range(4)]
    V222 = [[0]*7 for _ in range(4)]
    W222 = [[0]*7 for _ in range(4)]
    
    # Term 0: (a00+a11)(b00+b11) -> c00 +=, c01 +=, c10 +=, c11 +=
    # This is not quite right. The 2x2x2 Strassen computes C = A*B.
    # The 7 terms are specific linear combinations.
    # Let's just use the standard 2x2x2 Strassen matrices.
    
    # Standard 2x2x2 Strassen:
    # U:
    # t0: a00+a11, a01, a10+a11, a10-a00, a00+a01, a11, a01-a11
    # t1: ...
    # This is getting messy to hardcode manually.
    
    # Let's use a known set of 2x2x2 Strassen matrices.
    # U (4x7):
    # 1 0 1 0 1 0 0
    # 0 1 0 0 0 1 1
    # 1 1 0 0 0 0 0
    # 0 0 1 1 0 -1 0
    # Wait, the rows of U correspond to a00, a01, a10, a11.
    # Col 0: 1,0,1,0 -> a00+a11? No, a00 is row 0, a11 is row 3.
    # Let's check indices: i=0,j=0 -> a00. i=0,j=1 -> a01. i=1,j=0 -> a10. i=1,j=1 -> a11.
    
    # U222:
    # col 0: a00+a11 -> [1, 0, 0, 1]
    # col 1: a10+a11 -> [0, 0, 1, 1]
    # col 2: a00 -> [1, 0, 0, 0]
    # col 3: a11 -> [0, 0, 0, 1]
    # col 4: a00+a01 -> [1, 1, 0, 0]
    # col 5: a10-a00 -> [-1, 0, 1, 0]
    # col 6: a01-a11 -> [0, 1, 0, -1]
    
    # V222:
    # col 0: b00+b11 -> [1, 0, 0, 1]
    # col 1: b00 -> [1, 0, 0, 0]
    # col 2: b01-b11 -> [0, 1, 0, -1]
    # col 3: b10-b00 -> [-1, 0, 1, 0]
    # col 4: b11 -> [0, 0, 0, 1]
    # col 5: b00+b01 -> [1, 1, 0, 0]
    # col 6: b10+b11 -> [0, 0, 1, 1]
    
    # W222:
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # W rows: c00, c01, c10, c11.
    # W222:
    # row 0 (c00): [1, 0, 0, 1, -1, 0, 1]
    # row 1 (c01): [0, 0, 1, 0, 1, 0, 0]
    # row 2 (c10): [0, 1, 0, 1, 0, 0, 0]
    # row 3 (c11): [1, -1, 1, 0, 0, 1, 0]
    
    # Let's verify this set.
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # This looks correct.
    
    U222 = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    V222 = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    W222 = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    # Now build 4x4x4.
    # We have 7 Strassen steps.
    # For each step s (0..6), we have a 2x2x2 multiplication.
    # The inputs are linear combinations of the 4x4 blocks.
    # We need to map the 2x2 block indices to the 4x4 matrix indices.
    
    # A blocks: A00, A01, A10, A11.
    # A00 is A[0:2, 0:2]
    # A01 is A[0:2, 2:4]
    # A10 is A[2:4, 0:2]
    # A11 is A[2:4, 2:4]
    
    # For each Strassen step s, we have coeff_A (4x4) and coeff_B (4x4).
    # These are sparse.
    
    # Step 0: A00+A11, B00+B11
    # Step 1: A10+A11, B00
    # Step 2: A00, B01-B11
    # Step 3: A11, B10-B00
    # Step 4: A00+A01, B11
    # Step 5: A10-A00, B00+B01
    # Step 6: A01-A11, B10+B11
    
    # We will iterate over s in 0..6.
    # For each s, we iterate over t in 0..6 (inner Strassen).
    # Total 49 terms.
    
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    # Define the block coefficients for each Strassen step.
    # coeff_A[s][block] where block is 0..3 (00, 01, 10, 11)
    # coeff_B[s][block]
    
    coeff_A = [
        [1, 0, 0, 1], # 0
        [0, 0, 1, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 0, 0, 1], # 3
        [1, 1, 0, 0], # 4
        [-1, 0, 1, 0],# 5
        [0, 1, 0, -1] # 6
    ]
    coeff_B = [
        [1, 0, 0, 1], # 0
        [1, 0, 0, 0], # 1
        [0, 1, 0, -1],# 2
        [0, 0, 1, -1],# 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [0, 0, 1, 1]  # 6
    ]
    
    term_idx = 0
    for s in range(7):
        for t in range(7):
            # For this term, the U part is determined by U222 and coeff_A.
            # U222[i, t] is the coefficient for the i-th element of the 2x2 matrix in step s.
            # The i-th element of the 2x2 matrix corresponds to a specific block.
            # i=0 -> block 00, i=1 -> block 01, i=2 -> block 10, i=3 -> block 11.
            
            for i in range(4):
                # i is the row index in U222 (0..3)
                # This corresponds to block i of A.
                # The coefficient is U222[i][t] * coeff_A[s][i].
                val = U222[i][t] * coeff_A[s][i]
                if val != 0:
                    # Map block i to 4x4 matrix rows/cols.
                    # Block 00: rows 0-1, cols 0-1
                    # Block 01: rows 0-1, cols 2-3
                    # Block 10: rows 2-3, cols 0-1
                    # Block 11: rows 2-3, cols 2-3
                    
                    r_start = (i // 2) * 2
                    c_start = (i % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            row = r_start + dr
                            col = c_start + dc
                            U_444[row * 4 + col][term_idx] = val
            
            for j in range(4):
                # j is the row index in V222 (0..3)
                # This corresponds to block j of B.
                val = V222[j][t] * coeff_B[s][j]
                if val != 0:
                    r_start = (j // 2) * 2
                    c_start = (j % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            row = r_start + dr
                            col = c_start + dc
                            V_444[row * 4 + col][term_idx] = val
            
            for k in range(4):
                # k is the row index in W222 (0..3)
                # This corresponds to block k of C.
                val = W222[k][t]
                if val != 0:
                    r_start = (k // 2) * 2
                    c_start = (k % 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            row = r_start + dr
                            col = c_start + dc
                            W_444[row * 4 + col][term_idx] = val
            
            term_idx += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # 2. Build 4x4x1 decomposition (Schoolbook)
    # Rank 16.
    U_441 = [[0]*16 for _ in range(16)]
    V_441 = [[0]*16 for _ in range(4)]
    W_441 = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j][col] = 1
            V_441[j][col] = 1
            W_441[i][col] = 1
    
    # 3. Combine
    # U: 16 x 49 + 16 x 16 -> 16 x 65
    # V: 16 x 49 + 4 x 16 -> 20 x 65
    # W: 16 x 49 + 4 x 16 -> 20 x 65
    
    rank = 65
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    for i in range(16):
        for t in range(49):
            U[i][t] = U_444[i][t]
        for t in range(16):
            U[i][49+t] = U_441[i][t]
            
    for i in range(16):
        for t in range(49):
            V[i][t] = V_444[i][t]
    for i in range(4):
        for t in range(16):
            V[16+i][49+t] = V_441[i][t]
            
    for i in range(16):
        for t in range(49):
            W[i][t] = W_444[i][t]
    for i in range(4):
        for t in range(16):
            W[16+i][49+t] = W_441[i][t]
            
    return {"U": U, "V": V, "W": W}
