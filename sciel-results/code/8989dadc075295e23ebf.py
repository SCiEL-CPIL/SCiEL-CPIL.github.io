import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen forms for 2x2 multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    
    # U (A factors) coefficients for each P
    # Indexing a: 0=a00, 1=a01, 2=a10, 3=a11
    U_forms = [
        [1, 0, 0, 1], # P1: a00 + a11
        [0, 0, 1, 1], # P2: a10 + a11
        [1, 0, 0, 0], # P3: a00
        [0, 0, 0, 1], # P4: a11
        [1, 1, 0, 0], # P5: a00 + a01
        [-1, 0, 1, 0],# P6: a10 - a00
        [0, 1, 0, -1] # P7: a01 - a11
    ]
    
    # V (B factors) coefficients for each P
    # Indexing b: 0=b00, 1=b01, 2=b10, 3=b11
    V_forms = [
        [1, 0, 0, 1], # P1: b00 + b11
        [1, 0, 0, 0], # P2: b00
        [0, 1, 0, -1],# P3: b01 - b11
        [-1, 0, 1, 0],# P4: b10 - b00
        [0, 0, 0, 1], # P5: b11
        [1, 1, 0, 0], # P6: b00 + b01
        [0, 0, 1, 1]  # P7: b10 + b11
    ]
    
    # W (C reconstruction) coefficients for each P
    # C = [[c00, c01], [c10, c11]]
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    # Indexing c: 0=c00, 1=c01, 2=c10, 3=c11
    # Note: The task defines W[k*n+i, r].
    # n=2.
    # c00 -> k=0, i=0 -> idx 0
    # c01 -> k=1, i=0 -> idx 2 (Wait, W[k*n+i]. k is col of C? No.)
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # C is p x n. p=8, n=2.
    # C[k, i].
    # In a 2x2 block of C (which is part of the 8x2 C):
    # The block corresponds to rows k0, k0+1 and cols i0, i0+1.
    # But here n=2, so i is always 0,1.
    # The "2x2" Strassen block produces a 2x2 result.
    # This 2x2 result maps to 2 rows of C (k, k+1) and 2 cols of C (i, i+1)?
    # No, C is 8x2.
    # The Strassen multiplication is 2x2 * 2x2 -> 2x2.
    # The "2x2" B block is 2 rows of B, 2 cols of B.
    # Let's trace the indices carefully.
    
    # A is 2x4.
    # B is 4x8.
    # C is 8x2.
    
    # We split A into two 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3).
    # We split B into four 2x4 blocks: B_TL (rows 0-1, cols 0-3), B_TR (rows 0-1, cols 4-7), B_BL (rows 2-3, cols 0-3), B_BR (rows 2-3, cols 4-7).
    # Wait, B is 4x8.
    # Rows of B correspond to cols of A.
    # C[k, i] = sum_j A[i, j] B[j, k].
    
    # Let's use the 8 Strassen instances approach.
    # Instance (a_block, b_row_block, b_col_block)
    # a_block: 0 (A_L, cols 0-1), 1 (A_R, cols 2-3)
    # b_row_block: 0 (B rows 0-1), 1 (B rows 2-3)
    # b_col_block: 0 (B cols 0-1), 1 (B cols 2-3), 2 (B cols 4-5), 3 (B cols 6-7)
    
    # Total 2 * 2 * 4 = 16 instances?
    # No, Strassen is for 2x2 * 2x2.
    # A_L is 2x2.
    # B block must be 2x2.
    # So B is split into 2 (rows) * 4 (cols) = 8 blocks of 2x2?
    # B has 4 rows, 8 cols.
    # 2x2 blocks: 2 row blocks * 4 col blocks = 8 blocks.
    # A has 2 row blocks? No, A is 2x4.
    # A is split into 2 col blocks of 2x2.
    # So we have 2 A blocks and 8 B blocks.
    # Each pair (A_block, B_block) where the B_block's row index matches the A_block's col index?
    # A_L (cols 0-1) multiplies B rows 0-1.
    # A_R (cols 2-3) multiplies B rows 2-3.
    # So:
    # A_L * B[0:2, 0:2] -> contributes to C[0:2, :]
    # A_L * B[0:2, 2:4] -> contributes to C[2:4, :]
    # A_L * B[0:2, 4:6] -> contributes to C[4:6, :]
    # A_L * B[0:2, 6:8] -> contributes to C[6:8, :]
    # A_R * B[2:4, 0:2] -> contributes to C[0:2, :]
    # A_R * B[2:4, 2:4] -> contributes to C[2:4, :]
    # A_R * B[2:4, 4:6] -> contributes to C[4:6, :]
    # A_R * B[2:4, 6:8] -> contributes to C[6:8, :]
    
    # Total 8 Strassen instances.
    # Each instance has rank 7.
    # Total rank 56.
    
    num_instances = 8
    rank = num_instances * 7
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    r = 0
    for a_block in range(2):
        # a_block 0: A cols 0-1. a_row_offset 0, a_col_offset 0
        # a_block 1: A cols 2-3. a_row_offset 0, a_col_offset 2
        a_col_off = a_block * 2
        
        # B row block must match a_block
        b_row_off = a_block * 2
        
        for b_col_block in range(4):
            # b_col_block 0: B cols 0-1
            # b_col_block 1: B cols 2-3
            # b_col_block 2: B cols 4-5
            # b_col_block 3: B cols 6-7
            b_col_off = b_col_block * 2
            
            # C row offset
            c_row_off = b_col_off
            
            for s in range(7):
                # Fill U
                for i in range(2):
                    for j in range(2):
                        u_idx = i * m + (a_col_off + j)
                        U[u_idx][r] = U_forms[s][i * 2 + j]
                
                # Fill V
                for j in range(2):
                    for k in range(2):
                        v_idx = (b_row_off + j) * p + (b_col_off + k)
                        V[v_idx][r] = V_forms[s][j * 2 + k]
                
                # Fill W
                # The 2x2 result C_sub is added to C[c_row_off:c_row_off+2, 0:2]
                # W index: k*n + i
                # k is global row of C (0..7)
                # i is global col of C (0..1)
                
                # C_sub[0,0] -> global C[c_row_off, 0]
                # C_sub[0,1] -> global C[c_row_off, 1]
                # C_sub[1,0] -> global C[c_row_off+1, 0]
                # C_sub[1,1] -> global C[c_row_off+1, 1]
                
                # Coefficients from W_forms (derived from c_reconstruction)
                # P1: c00(1), c11(1)
                # P2: c10(1), c11(-1)
                # P3: c01(1), c11(1)
                # P4: c00(1), c10(1)
                # P5: c00(-1), c01(1)
                # P6: c11(1)
                # P7: c00(1)
                
                # Let's build W_coeffs for each s
                if s == 0: # P1
                    W[c_row_off * n + 0][r] += 1
                    W[(c_row_off + 1) * n + 1][r] += 1
                elif s == 1: # P2
                    W[c_row_off * n + 1][r] += 1 # Wait, c10 is row 1 col 0?
                    # C_sub indices:
                    # 0: (0,0)
                    # 1: (0,1)
                    # 2: (1,0)
                    # 3: (1,1)
                    # c10 is (1,0) -> idx 2
                    # c11 is (1,1) -> idx 3
                    # P2: c10(1), c11(-1)
                    W[(c_row_off + 1) * n + 0][r] += 1
                    W[(c_row_off + 1) * n + 1][r] -= 1
                elif s == 2: # P3
                    # c01(1), c11(1)
                    W[c_row_off * n + 1][r] += 1
                    W[(c_row_off + 1) * n + 1][r] += 1
                elif s == 3: # P4
                    # c00(1), c10(1)
                    W[c_row_off * n + 0][r] += 1
                    W[(c_row_off + 1) * n + 0][r] += 1
                elif s == 4: # P5
                    # c00(-1), c01(1)
                    W[c_row_off * n + 0][r] -= 1
                    W[c_row_off * n + 1][r] += 1
                elif s == 5: # P6
                    # c11(1)
                    W[(c_row_off + 1) * n + 1][r] += 1
                elif s == 6: # P7
                    # c00(1)
                    W[c_row_off * n + 0][r] += 1
                
                r += 1
                
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
