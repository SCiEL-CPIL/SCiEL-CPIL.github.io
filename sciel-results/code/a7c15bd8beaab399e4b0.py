import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 3*4*8 = 96
    # We attempt to reduce rank by using Strassen's algorithm for 2x2 blocks
    # where possible, and schoolbook for the remaining terms.
    
    # Strategy:
    # C (3x8) = A (3x4) * B (4x8)
    # Partition A into rows: A0 (1x4), A12 (2x4)
    # Partition B into columns: B01 (4x2), B23 (4x2), B45 (4x2), B67 (4x2)
    # Or partition B into rows for multiplication with A blocks.
    
    # Let's use Strassen for the 2x2 subproblems involving A[1:3, :] and B[:, 0:2], etc.
    # Actually, it's easier to partition the matrices into 2x2 blocks where dimensions allow.
    
    # A is 3x4. We can view it as:
    # A = [A_top; A_bottom] where A_top is 1x4, A_bottom is 2x4.
    # B is 4x8.
    # C = A * B = [A_top * B; A_bottom * B]
    # C_top is 1x8. C_bottom is 2x8.
    
    # For C_bottom (2x8), we can use Strassen for each 2x2 block of B?
    # B is 4x8. Partition B into 4 blocks of 4x2? Or 2x4 blocks?
    # Strassen requires 2x2 matrices.
    # A_bottom is 2x4. Partition A_bottom into A_bottom_left (2x2) and A_bottom_right (2x2).
    # B is 4x8. Partition B into B_top (2x8) and B_bottom (2x8).
    # Then A_bottom * B = A_bottom_left * B_top + A_bottom_right * B_bottom.
    # This doesn't directly give 2x2 multiplications unless we also partition B.
    
    # Let's try a different partitioning.
    # Partition A into 3 blocks of rows? No, Strassen is for 2x2.
    
    # Let's stick to the previous successful approach from the "inspiration" which achieved 88.
    # That approach used Strassen for 2x2 blocks of A[1:3, :] * B[:, :] where B was partitioned into 4x2 blocks?
    # No, the previous code did:
    # Strassen for A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    # Strassen for A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    # ...
    # Strassen for A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    # ...
    
    # This covers the bottom 2 rows of C (rows 1 and 2).
    # Each Strassen call handles a 2x2 * 2x2 multiplication.
    # A[1:3, 0:2] is 2x2. B[0:2, 0:2] is 2x2. Result is C[1:3, 0:2] (2x2).
    # We need to cover all 2x2 blocks of C[1:3, 0:8].
    # C[1:3, 0:8] is 2x8. It can be divided into four 2x2 blocks horizontally:
    # C[1:3, 0:2], C[1:3, 2:4], C[1:3, 4:6], C[1:3, 6:8].
    # Each of these is the product of a 2x2 block of A and a 2x2 block of B?
    # C[1:3, 0:2] = A[1:3, 0:4] * B[0:4, 0:2]
    # This is a 2x4 * 4x2 multiplication. Not directly 2x2.
    # However, A[1:3, 0:4] = [A11, A12] where A11=A[1:3,0:2], A12=A[1:3,2:4].
    # B[0:4, 0:2] = [B1; B2] where B1=B[0:2,0:2], B2=B[2:4,0:2].
    # So C[1:3, 0:2] = A11*B1 + A12*B2.
    # This is the sum of two 2x2 multiplications.
    # If we use Strassen for A11*B1 and A12*B2 separately, we get 7+7=14 terms for this 2x2 block of C.
    # But we can't just add the terms because they contribute to the same output.
    # Wait, the tensor decomposition is linear. If C = X + Y, and we have decompositions for X and Y,
    # we can concatenate the terms.
    # So for C[1:3, 0:2], we need terms for A11*B1 and A12*B2.
    # A11*B1 is 2x2*2x2 -> rank 7.
    # A12*B2 is 2x2*2x2 -> rank 7.
    # Total 14 terms for this 2x2 block of C.
    
    # How many such 2x2 blocks of C[1:3, 0:8] are there?
    # C[1:3, 0:8] is 2x8. There are 4 such blocks (cols 0-1, 2-3, 4-5, 6-7).
    # For each block, we have 2 multiplications (left part of A * top part of B, right part of A * bottom part of B).
    # So 4 blocks * 2 multiplications * 7 terms = 56 terms.
    
    # Now for C[0, 0:8] (top row).
    # C[0, k] = sum_j A[0, j] B[j, k].
    # This is a 1x4 * 4x8 multiplication.
    # We can use schoolbook for this.
    # Number of terms = 4 * 8 = 32.
    
    # Total rank = 56 + 32 = 88.
    
    # Let's implement this.
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def strassen_2x2(x_offset, y_offset, z_offset):
        # x_offset(r, c) gives index in U for A[r, c]
        # y_offset(r, c) gives index in V for B[r, c]
        # z_offset(r, c) gives index in W for C[r, c]
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = {x_offset(0,0): 1, x_offset(1,1): 1}
        v1 = {y_offset(0,0): 1, y_offset(1,1): 1}
        w1 = {z_offset(0,0): 1, z_offset(0,1): 1, z_offset(1,0): 1, z_offset(1,1): 1}
        add_term(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        u2 = {x_offset(1,0): 1, x_offset(1,1): 1}
        v2 = {y_offset(0,0): 1}
        w2 = {z_offset(1,0): 1, z_offset(1,1): 1}
        add_term(u2, v2, w2)
        
        # M3 = A00(B10 - B11)
        u3 = {x_offset(0,0): 1}
        v3 = {y_offset(1,0): 1, y_offset(1,1): -1}
        w3 = {z_offset(0,0): 1, z_offset(0,1): 1}
        add_term(u3, v3, w3)
        
        # M4 = A11(B00 - B01)
        u4 = {x_offset(1,1): 1}
        v4 = {y_offset(0,0): 1, y_offset(0,1): -1}
        w4 = {z_offset(1,0): 1, z_offset(1,1): 1}
        add_term(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        u5 = {x_offset(0,0): 1, x_offset(0,1): 1}
        v5 = {y_offset(1,1): 1}
        w5 = {z_offset(0,1): 1}
        add_term(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = {x_offset(1,0): 1, x_offset(0,0): -1}
        v6 = {y_offset(0,0): 1, y_offset(0,1): 1}
        w6 = {z_offset(1,0): 1}
        add_term(u6, v6, w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = {x_offset(0,1): 1, x_offset(1,1): -1}
        v7 = {y_offset(1,0): 1, y_offset(1,1): 1}
        w7 = {z_offset(0,1): 1}
        add_term(u7, v7, w7)
        
    # Process C[1:3, 0:8]
    # C[1:3, 0:2] = A[1:3, 0:2] * B[0:2, 0:2] + A[1:3, 2:4] * B[2:4, 0:2]
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (0+c),  # A[1:3, 0:2]
        y_offset=lambda r,c: (0+r)*8 + (0+c),  # B[0:2, 0:2]
        z_offset=lambda r,c: (0+c)*3 + (1+r)   # C[1:3, 0:2]
    )
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (2+c),  # A[1:3, 2:4]
        y_offset=lambda r,c: (2+r)*8 + (0+c),  # B[2:4, 0:2]
        z_offset=lambda r,c: (0+c)*3 + (1+r)   # C[1:3, 0:2]
    )
    
    # C[1:3, 2:4] = A[1:3, 0:2] * B[0:2, 2:4] + A[1:3, 2:4] * B[2:4, 2:4]
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (0+c),  # A[1:3, 0:2]
        y_offset=lambda r,c: (0+r)*8 + (2+c),  # B[0:2, 2:4]
        z_offset=lambda r,c: (2+c)*3 + (1+r)   # C[1:3, 2:4]
    )
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (2+c),  # A[1:3, 2:4]
        y_offset=lambda r,c: (2+r)*8 + (2+c),  # B[2:4, 2:4]
        z_offset=lambda r,c: (2+c)*3 + (1+r)   # C[1:3, 2:4]
    )
    
    # C[1:3, 4:6] = A[1:3, 0:2] * B[0:2, 4:6] + A[1:3, 2:4] * B[2:4, 4:6]
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (0+c),  # A[1:3, 0:2]
        y_offset=lambda r,c: (0+r)*8 + (4+c),  # B[0:2, 4:6]
        z_offset=lambda r,c: (4+c)*3 + (1+r)   # C[1:3, 4:6]
    )
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (2+c),  # A[1:3, 2:4]
        y_offset=lambda r,c: (2+r)*8 + (4+c),  # B[2:4, 4:6]
        z_offset=lambda r,c: (4+c)*3 + (1+r)   # C[1:3, 4:6]
    )
    
    # C[1:3, 6:8] = A[1:3, 0:2] * B[0:2, 6:8] + A[1:3, 2:4] * B[2:4, 6:8]
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (0+c),  # A[1:3, 0:2]
        y_offset=lambda r,c: (0+r)*8 + (6+c),  # B[0:2, 6:8]
        z_offset=lambda r,c: (6+c)*3 + (1+r)   # C[1:3, 6:8]
    )
    strassen_2x2(
        x_offset=lambda r,c: (1+r)*4 + (2+c),  # A[1:3, 2:4]
        y_offset=lambda r,c: (2+r)*8 + (6+c),  # B[2:4, 6:8]
        z_offset=lambda r,c: (6+c)*3 + (1+r)   # C[1:3, 6:8]
    )
    
    # Process C[0, 0:8] using schoolbook
    for j in range(4):
        for k in range(8):
            u = {0*4 + j: 1}  # A[0, j]
            v = {j*8 + k: 1}  # B[j, k]
            w = {k*3 + 0: 1}  # C[0, k]
            add_term(u, v, w)
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
    
    return {"U": U, "V": V, "W": W}
