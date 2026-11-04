"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import numpy as np

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W: C side, W[k*2+i]
    # C00 = M1+M4-M5+M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3+M5
    W[1][2] = 1; W[1][4] = 1
    # C10 = M2+M4
    W[2][1] = 1; W[2][3] = 1
    # C11 = M1-M2+M3+M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _build_4x4x4():
    """Build rank-49 decomposition for 4x4x4 using recursive Strassen.
    U: 16 rows (4*4), V: 16 rows (4*4), W: 16 rows (4*4)
    """
    Us, Vs, Ws = _strassen_2x2x2()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Block offsets for 4x4 matrices (2x2 blocks)
    # Block (br, bc) has rows br*2..br*2+1, cols bc*2..bc*2+1
    # A blocks: A[br][bc] at row_offset=br*2, col_offset=bc*2
    # B blocks: B[br][bc] at row_offset=br*2, col_offset=bc*2
    # C blocks: C[br][bc] at row_offset=br*2, col_offset=bc*2
    
    # 7 Strassen products at block level
    # M_l = (sum_i a_l[i] * A_i) * (sum_j b_l[j] * B_j)
    # where A_i is block i of A, B_j is block j of B
    # Block indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_products = [
        ({0:1, 3:1}, {0:1, 3:1}),  # M1 = (A00+A11)(B00+B11)
        ({2:1, 3:1}, {0:1}),       # M2 = (A10+A11)B00
        ({0:1}, {1:1, 3:-1}),      # M3 = A00(B01-B11)
        ({3:1}, {2:1, 0:-1}),      # M4 = A11(B10-B00)
        ({0:1, 1:1}, {3:1}),       # M5 = (A00+A01)B11
        ({2:1, 0:-1}, {0:1, 1:1}), # M6 = (A10-A00)(B00+B01)
        ({1:1, 3:-1}, {2:1, 3:1}), # M7 = (A01-A11)(B10+B11)
    ]
    
    for prod_idx, (a_coeffs, b_coeffs) in enumerate(strassen_products):
        for i_idx, a_coeff in a_coeffs.items():
            i_br, i_bc = divmod(i_idx, 2)  # block row, block col
            for j_idx, b_coeff in b_coeffs.items():
                j_br, j_bc = divmod(j_idx, 2)
                # A block at (i_br, i_bc), B block at (j_br, j_bc)
                # Product valid only if i_bc == j_br (col of A = row of B)
                if i_bc != j_br:
                    continue
                # C block at (i_br, j_bc)
                c_br, c_bc = i_br, j_bc
                
                # Embed 2x2x2 Strassen for this block product
                for s in range(7):
                    col = prod_idx * 7 + s
                    coeff = a_coeff * b_coeff
                    
                    # U: A side, element (i_br*2+pi, i_bc*2+pj)
                    for pi in range(2):
                        for pj in range(2):
                            a_row = (i_br * 2 + pi) * 4 + (i_bc * 2 + pj)
                            U[a_row][col] += coeff * Us[pi * 2 + pj][s]
                    
                    # V: B side, element (j_br*2+pj, j_bc*2+pk)
                    for pj in range(2):
                        for pk in range(2):
                            b_row = (j_br * 2 + pj) * 4 + (j_bc * 2 + pk)
                            V[b_row][col] += coeff * Vs[pj * 2 + pk][s]
                    
                    # W: C side, element (c_br*2+pk, c_bc*2+pi)
                    for pk in range(2):
                        for pi in range(2):
                            c_row = (c_br * 2 + pk) * 4 + (c_bc * 2 + pi)
                            W[c_row][col] += Ws[pk * 2 + pi][s]
    
    return U, V, W

def _build_4x4x2():
    """Build rank-28 decomposition for 4x4x2.
    U: 16 rows (4*4), V: 8 rows (4*2), W: 8 rows (2*4)
    Uses 4 independent 2x2x2 Strassen decompositions.
    """
    Us, Vs, Ws = _strassen_2x2x2()
    
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(8)]
    W = [[0]*28 for _ in range(8)]
    
    # 4 block products:
    # C[0:2,0] = A[0:2,0:2]*B[0:2,0] + A[0:2,2:4]*B[2:4,0]
    # C[2:4,0] = A[2:4,0:2]*B[0:2,0] + A[2:4,2:4]*B[2:4,0]
    # C[0:2,1] = A[0:2,0:2]*B[0:2,1] + A[0:2,2:4]*B[2:4,1]
    # C[2:4,1] = A[2:4,0:2]*B[0:2,1] + A[2:4,2:4]*B[2:4,1]
    # 
    # That's 8 products of 2x2x2, rank 56. Too high.
    # 
    # Better approach: Use Strassen at the block level for 4x4x2.
    # Split A into 2x2 blocks, B into 2x2 blocks (each 2x1 effectively since B is 4x2)
    # 
    # Actually, let me use a different decomposition.
    # For 4x4x2, I'll use 4 independent 2x2x2 multiplications where
    # each computes one "half" of the output.
    #
    # Alternative: treat as two 2x4x2 problems.
    # 2x4x2: C(2x2) = A(2x4) * B(4x2)
    # Split A into A1(2x2), A2(2x2); B into B1(2x2), B2(2x2)
    # C = A1*B1 + A2*B2
    # Use rank-14: two separate 2x2x2 products
    # Total for 4x4x2: 2 * 14 = 28
    
    # Top half: C[0:2,:] = A[0:2,:] * B[:,0:2]... wait, B is 4x2
    # C[0:2, k] = sum_j A[0:2, j] * B[j, k] for k=0,1
    # = A[0:2,0:2]*B[0:2,k] + A[0:2,2:4]*B[2:4,k]
    
    # Let me just do 4 separate 2x2x2 decompositions for the 4 C blocks:
    # C_block(0,0) = A_block(0,0)*B_block(0,0) + A_block(0,1)*B_block(1,0)
    # But that's 2 products per C block = 8 total = rank 56.
    
    # Let me use the Strassen-style approach for 4x4x2:
    # M1 = (A00+A11)(B00+B10)  -> 2x2x2, rank 7
    # M2 = (A00+A11)(B01+B11)  -> 2x2x2, rank 7  
    # M3 = (A10-A00)(B00+B10)  -> 2x2x2, rank 7
    # M4 = (A10-A00)(B01+B11)  -> 2x2x2, rank 7
    # Total: rank 28
    
    # Where:
    # A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B00 = B[0:2,0], B01 = B[0:2,1], B10 = B[2:4,0], B11 = B[2:4,1]
    # (B blocks are 2x1, but we treat them as 2x2 with second column zero... no)
    
    # Actually B is 4x2, so B[0:2,:] is 2x2, B[2:4,:] is 2x2.
    # B00 = B[0:2,0:2], B01 doesn't make sense...
    
    # Let me redefine: B is 4x2. Split into B_top = B[0:2,:] (2x2) and B_bot = B[2:4,:] (2x2).
    # A is 4x4. Split into A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4].
    # C is 4x2. C[0:2,:] = A00*B_top + A01*B_bot, C[2:4,:] = A10*B_top + A11*B_bot.
    
    # Use Strassen-style:
    # M1 = (A00+A11)(B_top+B_bot)
    # M2 = (A00+A11)B_top
    # M3 = A00*B_bot
    # M4 = A11*B_top
    # M5 = (A00+A01)B_bot  -- no, this doesn't work cleanly
    
    # Let me just use the simple approach: 4 separate 2x2x2 decompositions.
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # That's 4 products, rank 28.
    
    products = [
        # (A_block_row, A_block_col, B_block_row, C_block_row)
        # A[br*2:(br+1)*2, bc*2:(bc+1)*2] * B[br*2:(br+1)*2, :] -> C[br*2:(br+1)*2, :]
        # But B is 4x2, so B block is 2x2.
        # C[br*2:(br+1)*2, :] = sum_bc A[br*2:(br+1)*2, bc*2:(bc+1)*2] * B[bc*2:(bc+1)*2, :]
        
        # Product 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 0, 0, 0),
        # Product 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 1, 1, 0),
        # Product 3: A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
        (1, 0, 0, 1),
        # Product 4: A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
        (1, 1, 1, 1),
    ]
    
    for prod_idx, (a_br, a_bc, b_br, c_br) in enumerate(products):
        for s in range(7):
            col = prod_idx * 7 + s
            
            # U: A side, element (a_br*2+pi, a_bc*2+pj)
            for pi in range(2):
                for pj in range(2):
                    a_row = (a_br * 2 + pi) * 4 + (a_bc * 2 + pj)
                    U[a_row][col] += Us[pi * 2 + pj][s]
            
            # V: B side, element (b_br*2+pj, pk) where pk is 0 or 1
            for pj in range(2):
                for pk in range(2):
                    b_row = (b_br * 2 + pj) * 2 + pk
                    V[b_row][col] += Vs[pj * 2 + pk][s]
            
            # W: C side, element (c_br*2+pk, pi)
            for pk in range(2):
                for pi in range(2):
                    c_row = (c_br * 2 + pk) * 4 + pi
                    W[c_row][col] += Ws[pk * 2 + pi][s]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 6
    
    # Build 4x4x4 part (rank 49)
    U_444, V_444, W_444 = _build_4x4x4()
    
    # Build 4x4x2 part (rank 28)
    U_442, V_442, W_442 = _build_4x4x2()
    
    total_rank = 49 + 28
    
    # Assemble full matrices
    # U: 16 rows (n*m = 4*4), total_rank cols
    # V: 24 rows (m*p = 4*6), total_rank cols
    # W: 24 rows (p*n = 6*4), total_rank cols
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4x4 part (columns 0-48)
    # U_444: 16 rows, V_444: 16 rows (for B[0:4,0:4]), W_444: 16 rows (for C[0:4,0:4])
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
            V[i][c] = V_444[i][c]
            W[i][c] = W_444[i][c]
    
    # Copy 4x4x2 part (columns 49-76)
    # U_442: 16 rows, V_442: 8 rows (for B[0:4,4:6] i.e. B columns 4,5), W_442: 8 rows (for C[4:6,0:4])
    # V_442 row j corresponds to B[j, 4+k] where k=0,1 -> global V row j*6 + 4 + k
    # W_442 row (2*pk+pi) corresponds to C[4+pk, pi] -> global W row (4+pk)*4 + pi
    for i in range(16):
        for c in range(28):
            U[i][49 + c] = U_442[i][c]
    
    for r in range(8):
        for c in range(28):
            # V_442 row r: r = j*2 + k where j=0..1 (local B row 0..3), k=0..1
            # Actually r = (b_br*2+pj)*2 + pk, so local B row = r//2, local col = r%2
            local_b_row = r // 2
            local_b_col = r % 2
            global_v_row = local_b_row * 6 + 4 + local_b_col
            V[global_v_row][49 + c] = V_442[r][c]
    
    for r in range(8):
        for c in range(28):
            # W_442 row r: r = (c_br*2+pk)*4 + pi... wait, that's not right
            # In _build_4x4x2, W has 8 rows indexed by c_row = (c_br*2+pk)*4 + pi
            # But that gives 0..7 for c_br=0,1 and pk=0,1, pi=0,1
            # Actually c_row ranges over: c_br=0: 0,1,4,5; c_br=1: 8,9,12,13 -- that's 16 values!
            # Wait, I made an error. Let me recheck.
            # In _build_4x4x2, W has 8 rows. The indexing c_row = (c_br*2+pk)*4 + pi
            # for c_br in {0,1}, pk in {0,1}, pi in {0,1} gives:
            # c_br=0,pk=0: pi=0->0, pi=1->1
            # c_br=0,pk=1: pi=0->4, pi=1->5
            # c_br=1,pk=0: pi=0->8, pi=1->9
            # c_br=1,pk=1: pi=0->12, pi=1->13
            # That's 8 distinct values: 0,1,4,5,8,9,12,13. But W only has 8 rows (0-7)!
            # Bug: I need to fix the W indexing in _build_4x4x2.
            pass
    
    # The W indexing in _build_4x4x2 is wrong. Let me rebuild it properly.
    # For 4x4x2, C is 4x2. W should have p*n = 2*4 = 8 rows.
    # W indexed by [k*n + i] = [k*4 + i] where k=0,1 (local col of C) and i=0..3 (row of C)
    # So W row = k*4 + i, giving 0..7.
    
    # Let me rebuild _build_442 with correct W indexing.
    
    if time.time() > deadline:
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
    
    # Rebuild 4x4x2 with correct W indexing
    Us, Vs, Ws = _strassen_2x2x2()
    U_442 = [[0]*28 for _ in range(16)]
    V_442 = [[0]*28 for _ in range(8)]
    W_442 = [[0]*28 for _ in range(8)]
    
    # 4 products:
    # (A_br, A_bc, B_br, C_br) where:
    # A block at rows A_br*2..A_br*2+1, cols A_bc*2..A_bc*2+1
    # B block at rows B_br*2..B_br*2+1, cols 0..1 (since B is 4x2)
    # C block at rows C_br*2..C_br*2+1, cols 0..1
    # Product: A_block * B_block -> C_block
    # A_block is 2x2, B_block is 2x2, C_block is 2x2
    
    products = [
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 1, 1, 0),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (1, 0, 0, 1),  # A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
        (1, 1, 1, 1),  # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    ]
    
    for prod_idx, (a_br, a_bc, b_br, c_br) in enumerate(products):
        for s in range(7):
            col = prod_idx * 7 + s
            
            # U: A side, element (a_br*2+pi, a_bc*2+pj)
            for pi in range(2):
                for pj in range(2):
                    a_row = (a_br * 2 + pi) * 4 + (a_bc * 2 + pj)
                    U_442[a_row][col] += Us[pi * 2 + pj][s]
            
            # V: B side, element (b_br*2+pj, pk) where pk=0,1
            for pj in range(2):
                for pk in range(2):
                    b_row = (b_br * 2 + pj) * 2 + pk
                    V_442[b_row][col] += Vs[pj * 2 + pk][s]
            
            # W: C side, element (pk, c_br*2+pi) -> W row = pk*4 + (c_br*2+pi)
            for pk in range(2):
                for pi in range(2):
                    c_row = pk * 4 + (c_br * 2 + pi)
                    W_442[c_row][col] += Ws[pk * 2 + pi][s]
    
    # Now assemble properly
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
            V[i][c] = V_444[i][c]
            W[i][c] = W_444[i][c]
    
    # Copy 4x4x2 part
    for i in range(16):
        for c in range(28):
            U[i][49 + c] = U_442[i][c]
    
    for r in range(8):
        for c in range(28):
            local_b_row = r // 2
            local_b_col = r % 2
            global_v_row = local_b_row * 6 + 4 + local_b_col
            V[global_v_row][49 + c] = V_442[r][c]
    
    for r in range(8):
        for c in range(28):
            # W_442 row r = pk*4 + (c_br*2+pi), so global W row = (4+pk)*4 + (c_br*2+pi)
            # r = pk*4 + c_local where c_local = c_br*2+pi (0..3)
            pk = r // 4
            c_local = r % 4
            global_w_row = (4 + pk) * 4 + c_local
            W[global_w_row][49 + c] = W_442[r][c]
    
    # Verify with random matrices
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    A_test = np.random.randint(0, 3, (4, 4))
    B_test = np.random.randint(0, 3, (4, 6))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
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
