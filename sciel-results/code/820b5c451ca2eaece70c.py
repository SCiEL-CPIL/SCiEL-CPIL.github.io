"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Standard Strassen for 2x2x2
    # C = A * B, A,B,C are 2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    #           B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    #           C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    U = [[0]*7 for _ in range(4)]  # 2*2=4 rows
    V = [[0]*7 for _ in range(4)]  # 2*2=4 rows
    W = [[0]*7 for _ in range(4)]  # 2*2=4 rows
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: A side, indexed by [i*2+j]
    # V: B side, indexed by [j*2+k]
    # W: C side, indexed by [k*2+i]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    
    # M3: A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1; V[3][2] = -1  # B01, -B11
    
    # M4: A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1; V[0][3] = -1  # B10, -B00
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    
    # W: C side
    # C00 = M1+M4-M5+M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3+M5
    W[1][2] = 1; W[1][4] = 1
    # C10 = M2+M4
    W[2][1] = 1; W[2][3] = 1
    # C11 = M1-M2+M3+M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _embed_strassen_2x2x2(U, V, W, row_offset_a, col_offset_a, row_offset_b, col_offset_b, row_offset_c, col_offset_c, U_full, V_full, W_full, start_col):
    """Embed a 2x2x2 Strassen decomposition into larger matrices."""
    n = 2
    for c in range(7):
        for i in range(n):
            for j in range(n):
                idx_a = (row_offset_a + i) * 4 + (col_offset_a + j)  # A is 4x4, so stride is 4
                U_full[idx_a][start_col + c] += U[i * n + j][c]
        for j in range(n):
            for k in range(n):
                idx_b = (row_offset_b + j) * 2 + (col_offset_b + k)  # B is 4x6, so stride is 6... wait
                V_full[idx_b][start_col + c] += V[j * n + k][c]
        for k in range(n):
            for i in range(n):
                idx_c = (row_offset_c + k) * 4 + (row_offset_c2 + i) if False else (row_offset_c + k) * 4 + (col_offset_c + i)
                # W is indexed by [k*n+i] where n=4 for the full problem
                # In the full 4x4x6 problem, W has p*n = 6*4 = 24 rows
                # W[k*4 + i] for k in 0..5, i in 0..3
                W_full[idx_c][start_col + c] += W[k * n + i][c]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = 4, 4, 6
    # We'll use rank-49 for 4x4x4 and rank-28 for 4x4x2
    # Total rank = 49 + 28 = 77
    
    # First, let's build the 4x4x4 decomposition using recursive Strassen
    # 4x4x4 can be decomposed using 9 instances of 2x2x2 (rank 7 each) = rank 63
    # But with sharing, we can get rank 49
    
    # Actually, the known rank-49 decomposition for 4x4x4 uses a specific structure.
    # Let me use the approach: 4x4x4 = combine 9 2x2x2 multiplications with sharing.
    
    # For 4x4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[A00*B00+A01*B10, A00*B01+A01*B11],
    #      [A10*B00+A11*B10, A10*B01+A11*B11]]
    
    # Using Strassen-style for the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2x2 multiplication, rank 7, total 7*7 = 49
    
    # Let's build this explicitly
    U_444 = [[0]*49 for _ in range(16)]  # 4*4=16
    V_444 = [[0]*49 for _ in range(16)]  # 4*4=16
    W_444 = [[0]*49 for _ in range(16)]  # 4*4=16
    
    # Get the 2x2x2 Strassen decomposition
    Us, Vs, Ws = _strassen_2x2x2()
    
    # For each of the 7 block-level Strassen products, embed the 2x2x2 decomposition
    # Block offsets: A00 at (0,0), A01 at (0,2), A10 at (2,0), A11 at (2,2)
    # Same for B and C
    
    block_offsets = [(0,0), (0,2), (2,0), (2,2)]
    
    # Define the 7 Strassen products at block level
    # Each product is (linear_combination_of_A_blocks) * (linear_combination_of_B_blocks)
    strassen_products = [
        # (A_coeffs, B_coeffs) where each is dict from block_idx to coeff
        # M1 = (A00+A11)(B00+B11)
        ({0:1, 3:1}, {0:1, 3:1}),
        # M2 = (A10+A11)B00
        ({2:1, 3:1}, {0:1}),
        # M3 = A00(B01-B11)
        ({0:1}, {1:1, 3:-1}),
        # M4 = A11(B10-B00)
        ({3:1}, {2:1, 0:-1}),
        # M5 = (A00+A01)B11
        ({0:1, 1:1}, {3:1}),
        # M6 = (A10-A00)(B00+B01)
        ({2:1, 0:-1}, {0:1, 1:1}),
        # M7 = (A01-A11)(B10+B11)
        ({1:1, 3:-1}, {2:1, 3:1}),
    ]
    
    for prod_idx, (a_coeffs, b_coeffs) in enumerate(strassen_products):
        base_col = prod_idx * 7
        # For each term in A combination and B combination, embed the 2x2x2
        # But we need to be careful: the product is (sum_i c_i * A_block_i) * (sum_j d_j * B_block_j)
        # This expands to sum_{i,j} c_i * d_j * A_block_i * B_block_j
        # Each A_block_i * B_block_j is a 2x2x2 multiplication
        
        for i_idx, a_coeff in a_coeffs.items():
            i_row, i_col = block_offsets[i_idx]
            for j_idx, b_coeff in b_coeffs.items():
                j_row, j_col = block_offsets[j_idx]
                # Embed 2x2x2 at A offset (i_row, i_col), B offset (j_row, j_col), C offset... 
                # But the C offset depends on which C block this contributes to
                # This gets complicated. Let me use a different approach.
        
        # Actually, let me just directly construct the rank-49 decomposition
        # by using the known structure.
    
    # Let me use a simpler approach: directly implement the known rank-49 decomposition
    # by building it from the 2x2x2 Strassen decomposition with proper embedding.
    
    # For 4x4x4, the rank-49 decomposition:
    # We have 7 groups of 7, each group corresponds to one Strassen product at the block level.
    # Within each group, we have the 2x2x2 Strassen decomposition.
    
    # The key insight: for product M_l = (sum_i a_{l,i} A_i)(sum_j b_{l,j} B_j),
    # where A_i and B_j are 2x2 blocks, this equals sum_{i,j} a_{l,i} b_{l,j} A_i B_j.
    # Each A_i B_j contributes to specific C blocks.
    
    # Let me just build it term by term.
    
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    # For each of the 7 block-level products, and for each pair of blocks (i,j) in the product,
    # embed the 2x2x2 Strassen decomposition.
    
    for prod_idx, (a_coeffs, b_coeffs) in enumerate(strassen_products):
        for i_idx, a_coeff in a_coeffs.items():
            i_row, i_col = block_offsets[i_idx]
            for j_idx, b_coeff in b_coeffs.items():
                j_row, j_col = block_offsets[j_idx]
                # The product A_i * B_j contributes to C block at position... 
                # In the full 4x4x4, C = A*B, so C[k,l] = sum_j A[k,j]*B[j,l]
                # Block A_i is at rows i_row..i_row+1, cols i_col..i_col+1
                # Block B_j is at rows j_row..j_row+1, cols j_col..j_col+1
                # The product A_i * B_j is only non-zero if i_col == j_row (col of A matches row of B)
                # Wait, that's not right. In matrix multiplication C = A*B:
                # C[p,q] = sum_r A[p,r] * B[r,q]
                # So block A at (i_row, i_col) multiplies with block B at (i_col, j_col) to give C block at (i_row, j_col)
                
                # So for block A_i at (i_row, i_col) and block B_j at (j_row, j_col):
                # This product is valid only if i_col == j_row
                # And it contributes to C block at (i_row, j_col)
                
                if i_col != j_row:
                    continue
                    
                c_row, c_col = i_row, j_col
                c_idx = block_offsets.index((c_row, c_col)) if (c_row, c_col) in block_offsets else None
                if c_idx is None:
                    continue
                
                # Now embed the 2x2x2 Strassen decomposition for A_i * B_j -> C_block
                # with coefficients a_coeff * b_coeff
                for s in range(7):
                    col = prod_idx * 7 + s
                    # U: A side
                    for pi in range(2):
                        for pj in range(2):
                            a_idx = (i_row + pi) * 4 + (i_col + pj)
                            U_444[a_idx][col] += a_coeff * Us[pi * 2 + pj][s]
                    # V: B side
                    for pj in range(2):
                        for pk in range(2):
                            b_idx = (j_row + pj) * 4 + (j_col + pk)
                            V_444[b_idx][col] += b_coeff * Vs[pj * 2 + pk][s]
                    # W: C side
                    for pk in range(2):
                        for pi in range(2):
                            c_idx_pos = (c_row + pk) * 4 + (c_col + pi)
                            W_444[c_idx_pos][col] += Ws[pk * 2 + pi][s]
    
    # Now for the 4x4x2 part (columns 4,5 of B)
    # We need to compute A(4x4) * B(4x2) -> C(4x2)
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]] where each B block is 2x2
    # C = [[A00*B00+A01*B10, A00*B01+A01*B11],
    #      [A10*B00+A11*B10, A10*B01+A11*B11]]
    
    # For 4x4x2, we can use a similar Strassen-style decomposition.
    # The rank of 4x4x2 is known to be 24 (using 3 groups of 2x2x2 with sharing)
    # Actually, let me think: 4x4x2 has 4*4*2 = 32 entries.
    # We can decompose it as 4 instances of 2x2x2 (rank 7 each) = rank 28, 
    # but with sharing we might get 24.
    
    # Simple approach: 4 separate 2x2x2 multiplications, rank 28
    # This gives total rank 49 + 28 = 77
    
    U_442 = [[0]*28 for _ in range(16)]  # A is 4x4
    V_442 = [[0]*28 for _ in range(8)]   # B is 4x2
    W_442 = [[0]*28 for _ in range(8)]   # C is 4x2, indexed by [k*4+i]
    
    # 4 block multiplications:
    # (A00*B00 -> C00), (A01*B10 -> C00), (A10*B00 -> C10), (A11*B10 -> C10)
    # (A00*B01 -> C01), (A01*B11 -> C01), (A10*B01 -> C11), (A11*B11 -> C11)
    # That's 8 multiplications of 2x2x2, rank 8*7 = 56. Too high.
    
    # Better: use Strassen-style for the block level.
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # C01 = A00*B01 + A01*B11
    # C11 = A10*B01 + A11*B11
    
    # We can use:
    # M1 = (A00+A11)(B00+B10) -> contributes to C00, C10, C11, C01 (with signs)
    # Actually, let me just use 4 separate 2x2x2 decompositions for the 4 C blocks,
    # but that's rank 28. Or use sharing.
    
    # Let me use the simple approach: for each of the 4 C blocks, use a 2x2x2 decomposition
    # C00 = A00*B00 + A01*B10: this is sum of two 2x2x2 products, rank 14
    # Similarly for others. Total rank = 4 * 14 = 56. Still high.
    
    # Better: use the fact that C00 and C10 share B00, B10.
    # Let me just use rank 28 (4 separate 2x2x2 for the 4 "diagonal" contributions)
    # No, let me think differently.
    
    # For 4x4x2, a known decomposition has rank 24.
    # Let me use: split into two 2x4x2 problems (top and bottom halves of A).
    # 2x4x2: C(2x2) = A(2x4) * B(4x2)
    # This can be done with rank 12 (using 2x2x2 Strassen twice with sharing)
    # Total: 2 * 12 = 24
    
    # 2x4x2: A is 2x4, B is 4x2, C is 2x2
    # Split A into A_left(2x2), A_right(2x2), B into B_top(2x2), B_bot(2x2)
    # C = A_left * B_top + A_right * B_bot
    # Use Strassen-style: 
    # M1 = (A_left + A_right)(B_top + B_bot)
    # M2 = A_left * B_top
    # M3 = A_right * B_bot
    # C = M1 - M2 - M3
    # That's 3 2x2x2 multiplications, rank 21. Not better.
    
    # Actually C = A_left*B_top + A_right*B_bot, just 2 products, rank 14.
    # Total for 4x4x2: 2 * 14 = 28.
    
    # Let me just use rank 28 for the 4x4x2 part. Total = 49 + 28 = 77.
    
    # Build 4x4x2 with rank 28:
    # 4 separate 2x2x2 multiplications:
    # A00*B00 -> C00, A01*B10 -> C00, A10*B00 -> C10, A11*B10 -> C10
    # A00*B01 -> C01, A01*B11 -> C01, A10*B01 -> C11, A11*B11 -> C11
    # That's 8 products, rank 56. No good.
    
    # Let me use 2 products per C block:
    # C00 = A00*B00 + A01*B10: use 2 separate 2x2x2, rank 14
    # C10 = A10*B00 + A11*B10: rank 14
    # C01 = A00*B01 + A01*B11: rank 14
    # C11 = A10*B01 + A11*B11: rank 14
    # Total: 56. Still too high.
    
    # I need sharing. Let me use the Strassen approach at the block level for 4x4x2.
    
    # For 4x4x2, use:
    # M1 = (A00+A11)(B00+B10) 
    # M2 = (A00+A11)(B01+B11)
    # M3 = (A10-A00)(B00+B10)
    # M4 = (A10-A00)(B01+B11)
    # M5 = A00*B10
    # M6 = A00*B11
    # M7 = A10*B00
    # M8 = A10*B01
    # ... this is getting complicated.
    
    # Let me just use the schoolbook for the 4x4x2 part: rank 32.
    # Total: 49 + 32 = 81.
    
    # Or use a better decomposition. Let me use rank 24.
    
    # Actually, let me just implement it carefully.
    
    # For 4x4x2, I'll use the following approach:
    # Treat it as two independent 2x4x2 problems (rows 0-1 and 2-3 of A).
    # Each 2x4x2: C(2x2) = A(2x4) * B(4x2)
    # For 2x4x2, use rank 12 decomposition.
    
    # 2x4x2 rank 12:
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x2
    # C = A1*B1 + A2*B2
    # Use: M1 = (A1+A2)(B1+B2), M2 = A1*B1, M3 = A2*B2
    # C = M1 - M2 - M3
    # Rank: 3*7 = 21. Not 12.
    
    # Hmm, 2x4x2 should have lower rank. Let me think...
    # 2x4x2: 2*4*2 = 16 entries. Lower bound is max(2*4, 4*2, 2*2) = 8.
    # Rank 12 might be achievable.
    
    # Let me just use the schoolbook for 4x4x2: rank 32.
    # Total: 49 + 32 = 81.
    
    # Let me go with total rank 81 for now and see if I can improve.
    
    # Actually, let me reconsider. Let me just use the full schoolbook for everything:
    # Rank 96. That's the baseline.
    
    # With rank-49 for 4x4x4 and schoolbook 32 for 4x4x2: rank 81.
    
    # Let me implement this.
    
    # Build full matrices
    total_rank = 49 + 32
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]  # m*p = 4*6 = 24
    W = [[0]*total_rank for _ in range(24)]  # p*n = 6*4 = 24
    
    # Copy 4x4x4 part (first 49 columns)
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
            V[i][c] = V_444[i][c]
            W[i][c] = W_444[i][c]
    
    # Build 4x4x2 part (columns 49-80)
    # Schoolbook for 4x4x2: C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
    # B columns 4,5 correspond to k=0,1 in the 4x4x2 subproblem
    # V index: j*6 + k for j in 0..3, k in 4,5 -> j*6+4, j*6+5
    # W index: k*4 + i for k in 0,1 (local) -> global k = 4,5 -> 4*4+i, 5*4+i
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k_local in range(2):
                k_global = 4 + k_local
                col = col_offset + i*4 + j*2 + k_local
                U[i*4 + j][col] = 1
                V[j*6 + k_global][col] = 1
                W[k_global*4 + i][col] = 1
    
    # Verify
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Quick verification with random matrices
    A_test = np.random.randint(0, 3, (4, 4))
    B_test = np.random.randint(0, 3, (4, 6))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # C[i,k] = sum_c U[i*4+j... wait, the indexing is different
    # U is indexed by [i*m+j] = [i*4+j], V by [j*p+k] = [j*6+k], W by [k*n+i] = [k*4+i]
    # C[i,k] = sum_j sum_c U[i*4+j, c] * V[j*6+k, c] * W[k*4+i, c]
    
    C_computed = np.zeros((4, 6), dtype=float)
    for i in range(4):
        for k in range(6):
            val = 0
            for j in range(4):
                for c in range(total_rank):
                    val += U_arr[i*4+j, c] * V_arr[j*6+k, c] * W_arr[k*4+i, c]
            C_computed[i, k] = val
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank = 96
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(24)]
        W = [[0]*rank for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
