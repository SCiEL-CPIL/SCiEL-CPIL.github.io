import time

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 28
    
    # Precompute the Strassen patterns for 2x2 multiplication
    # Each pattern is (u_dict, v_dict, w_dict)
    # Indices are 0..3 for 2x2 matrix
    strassen_patterns = [
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c01 -= M1, c10 -= M1, c11 += M1 ? No.
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Let's verify indices for W (output 2x2)
        # W[k*n+i, r] where k is row of C? No, W is (p*n, rank).
        # C is n x p = 2 x 4.
        # In the block decomposition, each block C_block is 2x2.
        # So W for a block is 2x2.
        
        # Pattern 1: M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1 = (a00+a11)(b00+b11)
        # Wait, the W coefficients for M1 in Strassen are:
        # c00 += M1, c11 += M1? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
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
        
        # So for M1: U={0:1, 3:1}, V={0:1, 3:1}, W={0:1, 3:1} (c00, c11)
        # For M2: U={2:1, 3:1}, V={0:1}, W={2:1, 3:-1} (c10, c11)
        # For M3: U={0:1}, V={1:1, 3:-1}, W={1:1, 3:1} (c01, c11) -> Wait, C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 (idx 1) and C11 (idx 3).
        # For M4: U={3:1}, V={2:1, 0:-1}, W={0:1, 2:1} (c00, c10)
        # For M5: U={0:1, 1:1}, V={3:1}, W={0:-1, 1:1} (c00, c01)
        # For M6: U={2:1, 0:-1}, V={0:1, 1:1}, W={3:1} (c11)
        # For M7: U={1:1, 3:-1}, V={2:1, 3:1}, W={0:1} (c00)
        
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Blocks:
    # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3).
    # B is 4x4. Split into B00, B01, B10, B11 (2x2 blocks).
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    
    # We need to compute 4 products:
    # P1 = A0 B00
    # P2 = A1 B10
    # P3 = A0 B01
    # P4 = A1 B11
    
    # For each product, we use 7 Strassen multiplications.
    # Total 28.
    
    blocks = [
        # (A_offset, B_row_offset, B_col_offset, C_offset)
        # A0 is cols 0-1 of A. B00 is rows 0-1, cols 0-1 of B. C0 is cols 0-1 of C.
        (0, 0, 0, 0),   # A0 B00 -> C0 part 1
        (2, 2, 0, 0),   # A1 B10 -> C0 part 2. A1 is cols 2-3. B10 is rows 2-3, cols 0-1.
        (0, 0, 2, 2),   # A0 B01 -> C1 part 1. B01 is rows 0-1, cols 2-3. C1 is cols 2-3.
        (2, 2, 2, 2),   # A1 B11 -> C1 part 2.
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            for local_idx, val in u_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = local_r * m + (a_off + local_c)
                U[g_idx][col_idx] = val
            
            for local_idx, val in v_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = (b_row_off + local_r) * p + (b_col_off + local_c)
                V[g_idx][col_idx] = val
            
            for local_idx, val in w_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_i = local_r
                g_k = c_off + local_c
                g_idx = g_k * n + g_i
                W[g_idx][col_idx] = val
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
