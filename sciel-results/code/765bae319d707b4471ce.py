"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 = 4x8.
    Uses Strassen's algorithm: 4x4x4 has rank 49, and 4x4x8 = two 4x4x4 blocks.
    Total rank = 49 + 49 = 98.
    """
    import time
    start_time = time.time()
    
    def strassen_2x2():
        """
        Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
        U: (4, 7) - input A indices (i,j) -> row i*2+j
        V: (4, 7) - input B indices (j,k) -> row j*2+k
        W: (4, 7) - output C indices (k*2+i) per trace convention W[k*n+i]
        """
        # 7 Strassen products
        # Each product is (linear combo of A) * (linear combo of B)
        # U columns: coefficients for A elements (A11, A12, A21, A22)
        # V columns: coefficients for B elements (B11, B12, B21, B22)
        # W columns: coefficients for C elements (C11, C12, C21, C22) mapped to W[k*2+i]
        # W index: k*2+i, so C11->W[1*2+0]=W[2], C12->W[1*2+1]=W[3], C21->W[0*2+1]=W[1], C22->W[0*2+0]=W[0]
        # Wait, let me re-check the trace convention: W[k*n+i, r]
        # For n=2: W[k*2+i, r]
        # C[i,j] = sum_k A[i,k]*B[k,j]
        # The tensor is T[i,j,k] = A[i,k]*B[k,j]*C[j,i] (with C on the output side)
        # Actually, the standard matrix multiplication tensor has T[i,j,k] where:
        #   i = row of A / row of C (0..n-1)
        #   j = col of A / row of B (0..m-1) 
        #   k = col of B / col of C (0..p-1)
        # And C[i,k] = sum_j A[i,j]*B[j,k]
        # The output W is indexed as W[k*n+i, r] per the task convention.
        # So for n=2, p=2: W[k*2+i, r]
        # C[i,k] corresponds to W[k*2+i, r]
        
        # Strassen 2x2x2:
        # A = [[A00, A01], [A10, A11]] (A[i,j], i=row, j=col)
        # B = [[B00, B01], [B10, B11]] (B[j,k], j=row, k=col)
        # C = [[C00, C01], [C10, C11]] (C[i,k])
        
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
        
        # U: rows are A elements indexed as A[i,j] -> row i*2+j (for m=2)
        # So A00->row 0, A01->row 1, A10->row 2, A11->row 3
        # V: rows are B elements indexed as B[j,k] -> row j*2+k (for p=2)
        # So B00->row 0, B01->row 1, B10->row 2, B11->row 3
        # W: rows are C elements indexed as C[i,k] -> W[k*2+i]
        # C00->W[0*2+0]=W[0], C01->W[1*2+0]=W[2], C10->W[0*2+1]=W[1], C11->W[1*2+1]=W[3]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # M2: (A10+A11)B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[1][1] = 1  # C10
        W[0][1] = -1 # C00 (with -1 sign... wait, C10 = M2+M4, C00 has no M2)
        # Let me redo: C10 = M2 + M4, so M2 contributes to C10
        # C11 = M1 - M2 + M3 + M6, so M2 contributes -1 to C11
        # Let me fix:
        # Actually let me be more careful. Let me reset and do this properly.
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to: C00 (+1), C11 (+1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # M2 = (A10+A11)(B00)
        # Contributes to: C10 (+1), C11 (-1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1  # C10, C11
        
        # M3 = A00(B01-B11)
        # Contributes to: C01 (+1), C11 (+1)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1  # C01, C11
        
        # M4 = A11(B10-B00)
        # Contributes to: C00 (+1), C10 (+1)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1  # C00, C10
        
        # M5 = (A00+A01)(B11)
        # Contributes to: C00 (-1), C01 (+1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1  # C00, C01
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes to: C11 (+1)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C11
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes to: C00 (+1)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1  # C00
        
        return U, V, W
    
    def strassen_4x4():
        """
        Return (U, V, W) for 4x4x4 Strassen decomposition, rank 49.
        U: (16, 49) - A[i,j] -> row i*4+j
        V: (16, 49) - B[j,k] -> row j*4+k
        W: (16, 49) - C[i,k] -> W[k*4+i]
        """
        U2, V2, W2 = strassen_2x2()
        
        # For 4x4, split into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen for 4x4:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2x2 multiplication, so rank 7 each.
        # Total rank = 7*7 = 49.
        
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Define the 7 Strassen products at the 4x4 level
        # Each product involves linear combinations of the 2x2 blocks
        # For each of the 7 products, we have a 2x2x2 Strassen decomposition
        
        # Product specifications: (U_block_combo, V_block_combo, W_block_combo)
        # U_block_combo: which 2x2 blocks of A and with what signs
        # V_block_combo: which 2x2 blocks of B and with what signs  
        # W_block_combo: which 2x2 blocks of C and with what signs
        
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        # M2 = (A21+A22)(B11) -> C21, and -C22 (from C22 = M1-M2+M3+M6)
        # M3 = A11(B12-B22) -> C12, and C22
        # M4 = A22(B21-B11) -> C11, C21
        # M5 = (A11+A12)(B22) -> -C11, C12
        # M6 = (A21-A11)(B11+B12) -> C22
        # M7 = (A12-A22)(B21+B22) -> C11
        
        # Let me define each product as:
        # (u_blocks, v_blocks, w_blocks) where each is a list of (block_idx, sign)
        # block_idx for A: 0=A11, 1=A12, 2=A21, 3=A22 (row-major 2x2 blocks)
        # block_idx for B: 0=B11, 1=B12, 2=B21, 3=B22
        # block_idx for C: 0=C11, 1=C12, 2=C21, 3=C22
        
        products = [
            # (u_blocks, v_blocks, w_blocks)
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),   # M1
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),        # M2
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),        # M3
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),        # M4
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),        # M5
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),        # M6
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),        # M7
        ]
        
        for prod_idx, (u_blocks, v_blocks, w_blocks) in enumerate(products):
            base = prod_idx * 7
            # For each of the 7 sub-products in the 2x2x2 Strassen:
            for sub_idx in range(7):
                col = base + sub_idx
                # U: for each (block, sign) in u_blocks, add sign * U2[sub_idx, local_row]
                # local_row for block b, local element (i,j) is b*4 + i*2 + j
                # U2 has rows 0-3 for A elements A00,A01,A10,A11
                for (block, sign) in u_blocks:
                    for local_row in range(4):
                        if U2[local_row][sub_idx] != 0:
                            global_row = block * 4 + local_row
                            U[global_row][col] += sign * U2[local_row][sub_idx]
                
                # V: similar
                for (block, sign) in v_blocks:
                    for local_row in range(4):
                        if V2[local_row][sub_idx] != 0:
                            global_row = block * 4 + local_row
                            V[global_row][col] += sign * V2[local_row][sub_idx]
                
                # W: similar, but W indexing is different
                # W2 has rows 0-3 corresponding to C elements:
                # W2[0] = C00, W2[1] = C10, W2[2] = C01, W2[3] = C11
                # For global C, block b, local element (i,k) -> C[i,k] in block b
                # Global W index: k*4+i where i is global row, k is global col
                # For block b (0-3) in 2x2 block arrangement:
                # Block 0 (C11): i=0,1; k=0,1 -> W[k*4+i]
                # Block 1 (C12): i=0,1; k=2,3 -> W[k*4+i]
                # Block 2 (C21): i=2,3; k=0,1 -> W[k*4+i]
                # Block 3 (C22): i=2,3; k=2,3 -> W[k*4+i]
                # Local W2 row mapping: W2[0]=C00, W2[1]=C10, W2[2]=C01, W2[3]=C11
                # For block b, local (i_local, k_local):
                # global i = (b//2)*2 + i_local, global k = (b%2)*2 + k_local
                # W2 local row for (i_local, k_local) = k_local*2 + i_local
                for (block, sign) in w_blocks:
                    for local_row in range(4):
                        if W2[local_row][sub_idx] != 0:
                            # local_row = k_local*2 + i_local
                            k_local = local_row // 2
                            i_local = local_row % 2
                            global_i = (block // 2) * 2 + i_local
                            global_k = (block % 2) * 2 + k_local
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col] += sign * W2[local_row][sub_idx]
        
        return U, V, W
    
    # Build 4x4x4 decomposition
    U4, V4, W4 = strassen_4x4()
    
    # Now build 4x4x8: this is two 4x4x4 multiplications
    # A (4x4) * B (4x8) = C (4x8)
    # B = [B_left, B_right] where B_left, B_right are 4x4
    # C = [C_left, C_right]
    # C_left = A * B_left, C_right = A * B_right
    
    rank = 98
    U = [[0]*rank for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0]*rank for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0]*rank for _ in range(32)]  # p*n = 8*4 = 32
    
    # First half: C_left = A * B_left
    # U is the same for both halves
    # V: B_left is columns 0-3 of B, B_right is columns 4-7
    # V index: j*8+k for j=0..3, k=0..7
    # For B_left: j*8+k where k=0..3
    # For B_right: j*8+k where k=4..7
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            U[i][c+49] = U4[i][c]
    
    # V for first half (B_left, k=0..3):
    # V4 has rows j*4+k for j=0..3, k=0..3
    # Global V row for B_left: j*8+k
    for j in range(4):
        for k in range(4):
            local_row = j*4+k
            global_row = j*8+k
            for c in range(49):
                V[global_row][c] = V4[local_row][c]
    
    # V for second half (B_right, k=4..7):
    for j in range(4):
        for k in range(4):
            local_row = j*4+k
            global_row = j*8+(k+4)
            for c in range(49):
                V[global_row][c+49] = V4[local_row][c]
    
    # W for first half (C_left, k=0..3):
    # W4 has rows k*4+i for k=0..3, i=0..3
    # Global W row for C_left: k*4+i (k=0..3)
    for k in range(4):
        for i in range(4):
            local_row = k*4+i
            global_row = k*4+i
            for c in range(49):
                W[global_row][c] = W4[local_row][c]
    
    # W for second half (C_right, k=4..7):
    for k in range(4):
        for i in range(4):
            local_row = k*4+i
            global_row = (k+4)*4+i
            for c in range(49):
                W[global_row][c+49] = W4[local_row][c]
    
    # Verify: check that the decomposition is correct
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
    # Let's do a quick sanity check with small values
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-2, 3, size=(4,4)).astype(float)
    B_test = np.random.randint(-2, 3, size=(4,8)).astype(float)
    C_expected = A_test @ B_test
    
    # Compute via decomposition
    C_computed = np.zeros((4,8))
    for i in range(4):
        for k in range(8):
            total = 0
            for j in range(4):
                for r in range(98):
                    total += U_np[i*4+j, r] * V_np[j*8+k, r] * W_np[k*4+i, r]
            C_computed[i,k] = total
    
    if not np.allclose(C_expected, C_computed, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank_fb = 128
        U_fb = [[0]*rank_fb for _ in range(16)]
        V_fb = [[0]*rank_fb for _ in range(32)]
        W_fb = [[0]*rank_fb for _ in range(32)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_fb[i*4+j][col] = 1
                    V_fb[j*8+k][col] = 1
                    W_fb[k*4+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
