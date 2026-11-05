import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 2 * Rank(4x4x4) = 2 * 49 = 98
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # Indices:
    # A: 2x2, B: 2x2, C: 2x2
    # U_A: 4x7, V_B: 4x7, W_C: 4x7
    # We define the 7 products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's build the 4x4x4 decomposition first (rank 49)
    # Then map it to the 4x4x8 case.
    # The 4x4x8 case is two independent 4x4x4 cases.
    # Case 1: A (4x4) * B_left (4x4) -> C_left (4x4)
    # Case 2: A (4x4) * B_right (4x4) -> C_right (4x4)
    
    # We need to construct the U, V, W for 4x4x4 using tensor product of Strassen.
    # Strassen U (2x2 to 7):
    # col 0: A00, A11 -> 1, 1, 0, 0 (for A00, A01, A10, A11)
    # Wait, standard Strassen:
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    # Let's represent A indices as (i,j) in 0..3
    # Block (bi, bj) where bi, bj in 0..1
    # A[bi*2+i, bj*2+j]
    
    # We will generate the 49 columns for the first 4x4x4 block.
    # And another 49 columns for the second 4x4x4 block.
    
    # Helper to add a Strassen tensor product contribution
    # For a specific block of A (bi, bj) and block of B (bk, bl)
    # The product contributes to C block (bi, bl)
    
    # Let's just hardcode the 7 Strassen factors and their combinations.
    # U_strassen (4x7):
    # Rows are A00, A01, A10, A11
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [1, 0, 1, 0, 0, 1, 1],  # B01
        [0, 0, -1, 1, 1, 0, -1],# B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Now construct 4x4x4 (rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # We iterate over 2x2 blocks of A, B, C.
    # For each combination of A-block (bi, bj), B-block (bk, bl), 
    # the Strassen product gives C-block (bi, bl).
    # Wait, the tensor product structure is:
    # C_{bi, bl} = sum_{bk} (A_{bi, bk} * B_{bk, bl})
    # Using Strassen for each 2x2 block multiplication:
    # The "rank" for 4x4x4 is 7*7=49.
    # The 49 terms are indexed by (s1, s2) where s1, s2 in 0..6.
    # s1 corresponds to the Strassen factor for the "row" block interaction?
    # Actually, the tensor product of two rank-7 tensors has rank 49.
    # The indices are (u, v) where u,v are Strassen indices.
    
    # Let's trace the indices.
    # A[bi*2+i, bk*2+k]
    # B[bk*2+k, bl*2+l]
    # C[bi*2+i, bl*2+l]
    
    # The decomposition is:
    # C = sum_{u,v} U_{u,v} V_{u,v} W_{u,v}
    # Where U, V, W are 16x49 matrices.
    
    # For a fixed pair (u, v):
    # U part comes from A.
    # V part comes from B.
    # W part comes from C.
    
    # The Strassen decomposition for 2x2 is:
    # C_2x2 = sum_s U_s[i, s] V_s[j, s] W_s[k, s] (indices simplified)
    
    # For 4x4:
    # C[bi*2+i, bl*2+l] = sum_{bk} sum_{u, v} 
    #    (U_s[u] applied to A[bi*2+i, bk*2+k]?) 
    # No, the tensor product is:
    # T_4x4x4 = T_2x2x2 \otimes T_2x2x2
    # This means:
    # C[bi*2+i, bl*2+l] = sum_{bk} sum_{u,v} 
    #    (U_s[bi, u] * A[bi*2+i, bk*2+k] ?)
    
    # Let's be precise.
    # T_{ijk} = sum_{u,v} U_{iu, ku} V_{kv, lv} W_{lv, ju} ... this is getting confusing.
    
    # Let's use the explicit block structure.
    # A is composed of 4 blocks A_{bi, bk}
    # B is composed of 4 blocks B_{bk, bl}
    # C is composed of 4 blocks C_{bi, bl}
    
    # C_{bi, bl} = sum_{bk} A_{bi, bk} B_{bk, bl}
    
    # We apply Strassen to the multiplication of 2x2 blocks.
    # For a fixed bi, bl, the sum is over bk.
    # But Strassen replaces the sum of 4 products (for 2x2) with 7 products.
    # Here, for each (bi, bl), we are summing over bk=0,1.
    # This is not a 2x2 matrix multiplication of blocks, it's a sum of two 2x2 matrix multiplications.
    # Wait, 4x4 = 2x2 blocks of 2x2.
    # C_{bi, bl} = A_{bi, 0} B_{0, bl} + A_{bi, 1} B_{1, bl}
    # This is a sum of two 2x2 matrix products.
    # We can't just apply Strassen to the sum directly as one 2x2x2 tensor.
    # However, the tensor product of two 2x2x2 tensors gives the 4x4x4 tensor.
    # T_4 = T_2 \otimes T_2.
    # T_2 has indices (i, j, k) for 2x2.
    # T_4 has indices (I, J, K) for 4x4, where I=(i1, i2), J=(j1, j2), K=(k1, k2).
    # T_4[I, J, K] = T_2[i1, j1, k1] * T_2[i2, j2, k2]
    
    # So, C[I, K] = sum_J A[I, J] B[J, K]
    # = sum_{j1, j2} A[i1, i2, j1, j2] B[j1, j2, k1, k2]
    # = sum_{j1, j2} (sum_{u,v} U_{i1, u} V_{j1, u} W_{k1, u} ... no)
    
    # Correct Tensor Product:
    # C[i1, i2, k1, k2] = sum_{j1, j2} A[i1, i2, j1, j2] B[j1, j2, k1, k2]
    # Using T_2 decomposition:
    # A[i1, j1] is not a scalar, it's part of a 2x2 block? No.
    # The indices are:
    # A: (i1, j1) and (i2, j2) are separate dimensions.
    # A[i1, i2, j1, j2] is just A[row, col] where row=i1*2+i2, col=j1*2+j2.
    
    # The decomposition is:
    # C[row, col] = sum_{s1, s2} U[row, s1, s2] V[col, s1, s2] W[out, s1, s2]
    # Actually, the standard construction is:
    # For each pair of Strassen indices (u, v) from 0..6:
    # The term is (U_s[u] \otimes U_s[v]) * (V_s[u] \otimes V_s[v]) * (W_s[u] \otimes W_s[v])
    # This produces a 4x4x4 term.
    
    # Let's construct U, V, W for one 4x4x4 block (rank 49).
    U_44 = [[0]*49 for _ in range(16)]
    V_44 = [[0]*49 for _ in range(16)]
    W_44 = [[0]*49 for _ in range(16)]
    
    for u in range(7):
        for v in range(7):
            col = u*7 + v
            # U part: A[row, col]
            # row is (i1, i2), col is (j1, j2)
            # U_s[u] gives coefficients for i1 (row block) and j1 (col block)
            # U_s[v] gives coefficients for i2 (row inner) and j2 (col inner)
            # Wait, U_s is 4x7. Rows are 0..3 corresponding to A00, A01, A10, A11.
            # So U_s[u][r] is coeff for A[r] where r is 0..3.
            # r=0 -> i1=0, j1=0
            # r=1 -> i1=0, j1=1
            # r=2 -> i1=1, j1=0
            # r=3 -> i1=1, j1=1
            
            # So for a fixed u, the contribution to A[i1, j1] is U_s[u][i1*2+j1].
            # Similarly for v, contribution to A[i2, j2] is U_s[v][i2*2+j2].
            
            # The total U for A[row, col] where row=i1*2+i2, col=j1*2+j2
            # is U_s[u][i1*2+j1] * U_s[v][i2*2+j2].
            
            for i1 in range(2):
                for i2 in range(2):
                    row = i1*2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            col_a = j1*2 + j2
                            U_44[row][col] += U_s[i1*2+j1][u] * U_s[i2*2+j2][v]
                            
                            # V part: B[col, out]
                            # col = (j1, j2), out = (k1, k2)
                            # V_s[u] gives coeff for B[j1, k1]
                            # V_s[v] gives coeff for B[j2, k2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    col_b = j1*2 + j2 # same as col_a
                                    out = k1*2 + k2
                                    V_44[col_b][col] += V_s[j1*2+k1][u] * V_s[j2*2+k2][v]
                                    
                                    # W part: C[row, out]
                                    # row = (i1, i2), out = (k1, k2)
                                    # W_s[u] gives coeff for C[i1, k1]
                                    # W_s[v] gives coeff for C[i2, k2]
                                    W_44[out][col] += W_s[i1*2+k1][u] * W_s[i2*2+k2][v]
                                    # Note: W index in W_44 is p*n = 4*4=16.
                                    # The problem statement says W[k*n+i, r].
                                    # So W index is k*4 + i.
                                    # My W_44 index above is 'out' which is k*2 + ... wait.
                                    # C[row, out]. row=i, out=k.
                                    # W index = k*n + i = k*4 + i.
                                    # My loop: out = k1*2+k2. This is the column index of C.
                                    # row = i1*2+i2. This is the row index of C.
                                    # So W index should be out*4 + row.
                                    # Let's re-verify the W indexing.
                                    # Task: W[k*n+i, r].
                                    # k is column of C (0..3), i is row of C (0..3).
                                    # So W index = k*4 + i.
                                    # In my loop:
                                    # k is 'out' (0..3)
                                    # i is 'row' (0..3)
                                    # So W index = out*4 + row.
                                    
                                    # I need to fix the W_44 assignment.
                                    # I was using W_44[out][col]. That's wrong.
                                    # It should be W_44[out*4 + row][col].
                                    
    # Let's redo the W part carefully.
    # I'll rewrite the loop to be safe.
    
    U_44 = [[0]*49 for _ in range(16)]
    V_44 = [[0]*49 for _ in range(16)]
    W_44 = [[0]*49 for _ in range(16)]
    
    for u in range(7):
        for v in range(7):
            col = u*7 + v
            for i1 in range(2):
                for i2 in range(2):
                    row_c = i1*2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            col_a = j1*2 + j2
                            # U
                            U_44[col_a][col] += U_s[i1*2+j1][u] * U_s[i2*2+j2][v]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    col_b = j1*2 + j2
                                    row_c_col = k1*2 + k2 # This is the column index of C
                                    # V
                                    V_44[col_b][col] += V_s[j1*2+k1][u] * V_s[j2*2+k2][v]
                                    # W
                                    # W index = k*n + i = row_c_col * 4 + row_c
                                    W_44[row_c_col * 4 + row_c][col] += W_s[i1*2+k1][u] * W_s[i2*2+k2][v]
                                    
    # Now we have U_44, V_44, W_44 for one 4x4x4 multiplication.
    # We need two such multiplications for 4x4x8.
    # A is 4x4.
    # B is 4x8. Split into B_left (4x4) and B_right (4x4).
    # C is 4x8. Split into C_left (4x4) and C_right (4x4).
    
    # U is same for both (depends only on A).
    # V is split: V_left for B_left, V_right for B_right.
    # W is split: W_left for C_left, W_right for C_right.
    
    # Total rank = 49 + 49 = 98.
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)] # m*p = 4*8 = 32
    W = [[0]*98 for _ in range(32)] # p*n = 8*4 = 32
    
    # Fill U
    for r in range(16):
        for c in range(49):
            U[r][c] = U_44[r][c]
            U[r][c+49] = U_44[r][c]
            
    # Fill V
    # B_left columns 0..3 correspond to V_44 rows 0..15?
    # V_44 is 16x49. Rows are j*4+k? No, V_44 rows are col_a (0..15).
    # In 4x4x4, B is 4x4. Row index of V is col of A = row of B.
    # So V_44[r, c] corresponds to B[r, col_b] where r is 0..15?
    # Wait, V_44 was constructed with rows 0..15.
    # In the 4x4x4 context, A is 4x4, B is 4x4.
    # U rows: A rows (0..15).
    # V rows: B rows (0..15).
    # W rows: C indices (0..15).
    
    # For 4x4x8:
    # A is 4x4. U rows 0..15.
    # B is 4x8. V rows 0..31.
    #   B_left is cols 0..3 of B. Rows 0..3 of B_left are rows 0..3 of B.
    #   So V_left uses V_44 rows 0..15 mapped to V rows 0..15.
    #   B_right is cols 4..7 of B. Rows 0..3 of B_right are rows 0..3 of B.
    #   So V_right uses V_44 rows 0..15 mapped to V rows 16..31?
    #   Wait, B is 4x8. Rows are 0..3. Cols are 0..7.
    #   V index is j*p + k. j is row of B (0..3), k is col of B (0..7).
    #   So V row index = j*8 + k.
    #   For B_left (k=0..3): V row = j*8 + k.
    #   For B_right (k=4..7): V row = j*8 + k.
    
    # My V_44 rows are indexed 0..15. How were they mapped?
    # In the loop: col_b = j1*2 + j2. This is 0..3.
    # But V_44 has 16 rows.
    # Ah, V_44 row index was col_b (0..3) in the inner loop?
    # No, `V_44[col_b][col]`. col_b is 0..3.
    # So V_44 only uses rows 0..3?
    # Let's check the V_44 construction loop.
    # `for k1 in range(2): for k2 in range(2): col_b = j1*2 + j2`
    # col_b depends only on j1, j2. It is constant for the inner k loops.
    # So V_44[row][col] is accumulated for row=col_b (0..3).
    # So V_44 rows 4..15 are zero.
    # This is correct because B in 4x4x4 is 4x4, so rows 0..3.
    
    # So for 4x4x8:
    # V_left: B rows 0..3, cols 0..3.
    # V row index = j*8 + k.
    # For j in 0..3, k in 0..3:
    #   V[j*8 + k][c] = V_44[j][c]
    
    # V_right: B rows 0..3, cols 4..7.
    # V row index = j*8 + k where k in 4..7.
    # For j in 0..3, k in 4..7:
    #   Let k' = k - 4.
    #   V[j*8 + k][c+49] = V_44[j][c]
    
    for j in range(4):
        for k in range(4):
            v_row_left = j*8 + k
            v_row_right = j*8 + (k+4)
            for c in range(49):
                V[v_row_left][c] = V_44[j][c]
                V[v_row_right][c+49] = V_44[j][c]
                
    # Fill W
    # W index = k*n + i. k is col of C (0..7), i is row of C (0..3).
    # C_left: cols 0..3.
    # C_right: cols 4..7.
    
    # W_44 rows are indexed `row_c_col * 4 + row_c`.
    # row_c_col is 0..3 (col of C in 4x4).
    # row_c is 0..3 (row of C in 4x4).
    # So W_44 index = col_c_4x4 * 4 + row_c.
    
    # For C_left (cols 0..3 of 4x8):
    # k in 0..3.
    # W index = k*4 + i.
    # This matches W_44 index directly.
    # So W[k*4 + i][c] = W_44[k*4 + i][c]
    
    # For C_right (cols 4..7 of 4x8):
    # k in 4..7. Let k' = k - 4.
    # W index = k*4 + i.
    # W_44 index for the corresponding 4x4 block is k'*4 + i.
    # So W[k*4 + i][c+49] = W_44[k'*4 + i][c]
    
    for k in range(4):
        for i in range(4):
            w_row_left = k*4 + i
            w_row_right = (k+4)*4 + i
            for c in range(49):
                W[w_row_left][c] = W_44[w_row_left][c]
                W[w_row_right][c+49] = W_44[w_row_left][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
