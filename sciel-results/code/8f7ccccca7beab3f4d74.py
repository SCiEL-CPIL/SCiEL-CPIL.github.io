"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 6
    
    # We construct a rank-77 decomposition:
    # 1) A rank-49 decomposition for the 4x4x4 part (first 4 columns of B)
    #    using recursive Strassen (7x7 structure).
    # 2) A rank-28 decomposition for the 4x4x2 part (last 2 columns of B)
    #    using 4 separate 2x2x2 Strassen decompositions.
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    # Define the 2x2x2 Strassen decomposition steps.
    # Each step is a tuple (alpha, beta, gamma) where each is a dict mapping
    # (row, col) to a coefficient.
    # The 7 Strassen products for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00      -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)      -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)      -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11      -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    #
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_222 = [
        # (alpha, beta, gamma)
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Part 1: 4x4x4 part (cols 0-48 of rank)
    # We use the recursive Strassen approach:
    # At the top level, we have 7 products, each of which is a product of 
    # linear combinations of 2x2 blocks. Each such product is itself a 
    # 2x2x2 multiplication that we decompose with Strassen.
    
    # The 7 top-level Strassen products for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a product of (sum of A blocks) * (sum of B blocks).
    # We need to expand each Mi into individual 2x2x2 multiplications.
    
    # For M1 = (A00+A11)(B00+B11):
    # This expands to A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But we can't just do 4 separate multiplications - that would be rank 28 for this product.
    # Instead, we use the structure: (A00+A11) and (B00+B11) are each 2x2 matrices.
    # So M1 is a single 2x2x2 multiplication: C_block = (A00+A11)*(B00+B11)
    # The result is a 2x2 matrix that contributes to multiple C blocks.
    
    # Actually, the correct recursive Strassen approach is:
    # For each of the 7 top-level products, we have a "virtual" 2x2x2 multiplication.
    # The A side is a linear combination of A blocks, the B side is a linear combination 
    # of B blocks, and the C side is a linear combination of C blocks.
    # We decompose each virtual 2x2x2 multiplication using the 2x2x2 Strassen decomposition.
    
    # So for each top-level product s (0..6), we have:
    #   A_virtual = sum_i alpha_s[i] * A_block_i  (a 2x2 matrix)
    #   B_virtual = sum_j beta_s[j] * B_block_j  (a 2x2 matrix)
    #   C_virtual = sum_k gamma_s[k] * C_block_k  (a 2x2 matrix)
    #   A_virtual * B_virtual = C_virtual
    #
    # We decompose A_virtual * B_virtual -> C_virtual using the 7 Strassen steps.
    # Each Strassen step t gives:
    #   A_sub = sum_i alpha_t[i] * (A_virtual)_i
    #   B_sub = sum_j beta_t[j] * (B_virtual)_j
    #   C_sub = sum_k gamma_t[k] * (C_virtual)_k
    #   A_sub * B_sub = C_sub
    #
    # Substituting:
    #   A_sub = sum_i alpha_t[i] * sum_i' alpha_s[i'] * A_block_i' = sum_i' (sum_i alpha_t[i] * alpha_s[i'] * delta_{i,i'}) ... 
    #   This is getting complicated. Let me use the approach from the inspiration.
    
    # The key insight from the inspiration: 
    # For each top-level product s and each Strassen step t, we have a column.
    # The U, V, W entries for that column are:
    #   U[r][col] = sum over (block_i, sub_i) of alpha_s[block_i] * alpha_t[sub_i]
    #   where r is the global index corresponding to (block_i, sub_i)
    # Similarly for V and W.
    
    # Let's implement this.
    
    # Top-level Strassen products for 4x4x4:
    # Each is (alpha_dict, beta_dict, gamma_dict) where keys are block indices (0..3)
    # Block 0: (0,0), Block 1: (0,1), Block 2: (1,0), Block 3: (1,1)
    top_level_strassen = [
        # M1 = (A00+A11)(B00+B11)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2 = (A10+A11)B00
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # M3 = A00(B01-B11)
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # M4 = A11(B10-B00)
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # M5 = (A00+A01)B11
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6 = (A10-A00)(B00+B01)
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7 = (A01-A11)(B10+B11)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # Block offsets: block index -> (row_offset, col_offset) in the 4x4 matrix
    block_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    
    # For each top-level product s and Strassen step t:
    for s in range(7):
        alpha_top, beta_top, gamma_top = top_level_strassen[s]
        for t in range(7):
            alpha_sub, beta_sub, gamma_sub = strassen_222[t]
            col = s * 7 + t
            
            # Process U (A side)
            # A_sub = sum_{block_i} alpha_top[block_i] * sum_{sub_i} alpha_sub[sub_i] * A_block_i[sub_i]
            # For each block_i in alpha_top and sub_i in alpha_sub:
            for block_i, coeff_top in alpha_top.items():
                for sub_i, coeff_sub in alpha_sub.items():
                    # sub_i is (row, col) within the 2x2 block
                    si, sj = sub_i
                    # Global position in A:
                    # Block block_i is at (block_offsets[block_i][0] + si, block_offsets[block_i][1] + sj)
                    br, bc = block_offsets[block_i]
                    global_row = br + si
                    global_col = bc + sj
                    u_idx = global_row * m + global_col
                    U[u_idx][col] += coeff_top * coeff_sub
            
            # Process V (B side)
            # B is 4x4 (first 4 columns of the full 4x6 B)
            for block_j, coeff_top in beta_top.items():
                for sub_j, coeff_sub in beta_sub.items():
                    sj_row, sj_col = sub_j
                    br, bc = block_offsets[block_j]
                    global_row = br + sj_row
                    global_col = bc + sj_col
                    # V index: row * p + col, but col is in the first 4 columns of B
                    v_idx = global_row * p + global_col
                    V[v_idx][col] += coeff_top * coeff_sub
            
            # Process W (C side)
            # C is 4x4 (first 4 columns of the full 4x6 C)
            for block_k, coeff_top in gamma_top.items():
                for sub_k, coeff_sub in gamma_sub.items():
                    sk_row, sk_col = sub_k
                    br, bc = block_offsets[block_k]
                    global_row = br + sk_row
                    global_col = bc + sk_col
                    # W index: col * n + row
                    w_idx = global_col * n + global_row
                    W[w_idx][col] += coeff_top * coeff_sub
    
    # Part 2: 4x4x2 part (cols 49-76 of rank)
    # A is 4x4, B2 is 4x2 (cols 4-5 of B), C2 is 4x2 (cols 4-5 of C)
    # We use 4 separate 2x2x2 Strassen decompositions:
    # M1: A00 * B0 -> C0 (A00 is rows 0-1, cols 0-1; B0 is rows 0-1, cols 4-5; C0 is rows 0-1, cols 4-5)
    # M2: A01 * B1 -> C0 (A01 is rows 0-1, cols 2-3; B1 is rows 2-3, cols 4-5)
    # M3: A10 * B0 -> C1 (A10 is rows 2-3, cols 0-1; C1 is rows 2-3, cols 4-5)
    # M4: A11 * B1 -> C1 (A11 is rows 2-3, cols 2-3)
    
    # Each is a 2x2x2 multiplication, rank 7. Total rank 28.
    
    mults = [
        # (a_block, b_block, c_block)
        # a_block: (row_offset, col_offset) in A
        # b_block: (row_offset, col_offset) in B (global)
        # c_block: (row_offset, col_offset) in C (global)
        ((0, 0), (0, 4), (0, 4)),   # A00 * B0 -> C0
        ((0, 2), (2, 4), (0, 4)),   # A01 * B1 -> C0
        ((2, 0), (0, 4), (2, 4)),   # A10 * B0 -> C1
        ((2, 2), (2, 4), (2, 4)),   # A11 * B1 -> C1
    ]
    
    col_start_2 = 49
    
    for m_idx, (a_block, b_block, c_block) in enumerate(mults):
        for t in range(7):
            alpha_sub, beta_sub, gamma_sub = strassen_222[t]
            col = col_start_2 + m_idx * 7 + t
            
            # Process U (A side)
            for (si, sj), coeff in alpha_sub.items():
                global_row = a_block[0] + si
                global_col = a_block[1] + sj
                u_idx = global_row * m + global_col
                U[u_idx][col] += coeff
            
            # Process V (B side)
            for (si, sj), coeff in beta_sub.items():
                global_row = b_block[0] + si
                global_col = b_block[1] + sj
                v_idx = global_row * p + global_col
                V[v_idx][col] += coeff
            
            # Process W (C side)
            for (si, sj), coeff in gamma_sub.items():
                global_row = c_block[0] + si
                global_col = c_block[1] + sj
                w_idx = global_col * n + global_row
                W[w_idx][col] += coeff
    
    # Verify with random matrices
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    np.random.seed(42)
    A_test = np.random.randint(0, 3, (n, m))
    B_test = np.random.randint(0, 3, (m, p))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
            C_computed[i, k] = val
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
