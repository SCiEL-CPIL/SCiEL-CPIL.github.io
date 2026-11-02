"""Matrix multiplication 2x4x6 with Strassen-based rank reduction."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return the 7 rank-1 terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where u is 4-dim (row-major 2x2), v is 4-dim, w is 4-dim.
    C = A*B where A is 2x2, B is 2x2, C is 2x2.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen:
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    C[0,0] = p1 + p4 - p5 + p7
    C[0,1] = p3 + p5
    C[1,0] = p2 + p4
    C[1,1] = p1 - p2 + p3 + p6
    """
    # u vectors: extract from A (row-major: a,b,c,d)
    # v vectors: extract from B (row-major: e,f,g,h)
    # w vectors: place into C (row-major: c00,c01,c10,c11)
    
    terms = []
    # p1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
    u1 = [1, 0, 0, 1]  # a+d
    v1 = [1, 0, 0, 1]  # e+h
    w1 = [1, 0, 0, 1]  # C[0,0] + C[1,1]
    terms.append((u1, v1, w1))
    
    # p2 = (c+d)e -> contributes to C[1,0] (positive) and C[1,1] (negative)
    u2 = [0, 0, 1, 1]  # c+d
    v2 = [1, 0, 0, 0]  # e
    w2 = [0, 0, 1, -1]  # C[1,0] - C[1,1]
    terms.append((u2, v2, w2))
    
    # p3 = a(f-h) -> contributes to C[0,1] (positive) and C[1,1] (positive)
    u3 = [1, 0, 0, 0]  # a
    v3 = [0, 1, 0, -1]  # f-h
    w3 = [0, 1, 0, 1]  # C[0,1] + C[1,1]
    terms.append((u3, v3, w3))
    
    # p4 = d(g-e) -> contributes to C[0,0] (positive) and C[1,0] (positive)
    u4 = [0, 0, 0, 1]  # d
    v4 = [0, 0, 1, -1]  # g-e
    w4 = [1, 0, 1, 0]  # C[0,0] + C[1,0]
    terms.append((u4, v4, w4))
    
    # p5 = (a+b)h -> contributes to C[0,0] (negative) and C[0,1] (positive)
    u5 = [1, 1, 0, 0]  # a+b
    v5 = [0, 0, 0, 1]  # h
    w5 = [-1, 1, 0, 0]  # -C[0,0] + C[0,1]
    terms.append((u5, v5, w5))
    
    # p6 = (c-a)(e+f) -> contributes to C[1,1] (positive)
    u6 = [-1, 0, 1, 0]  # c-a
    v6 = [1, 1, 0, 0]  # e+f
    w6 = [0, 0, 0, 1]  # C[1,1]
    terms.append((u6, v6, w6))
    
    # p7 = (b-d)(g+h) -> contributes to C[0,0] (positive)
    u7 = [0, 1, 0, -1]  # b-d
    v7 = [0, 0, 1, 1]  # g+h
    w7 = [1, 0, 0, 0]  # C[0,0]
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 x 4x6 matrix multiplication.
    Uses Strassen-based block decomposition to reduce rank from 48 to 42.
    
    Split the 2x4 matrix A into blocks and 4x6 matrix B into blocks,
    then use Strassen's algorithm on 2x2 blocks where applicable.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Strategy: Split A (2x4) into two 2x2 blocks: A = [A_left | A_right]
    # Split B (4x6) into two 4x3 blocks: B = [B_top; B_bottom]
    # Then C = A*B = A_left * B_top + A_right * B_bottom
    # Each term is (2x2) * (2x3) which has rank 6 (schoolbook)
    # Total rank = 6 + 6 = 12? No, that's not right.
    
    # Better approach: Split into 2x2 blocks for Strassen
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into [A1 | A2] where A1, A2 are 2x2
    # Split B into [B1; B2] where B1, B2 are 2x3
    # C = A1*B1 + A2*B2
    # Each (2x2)*(2x3) has schoolbook rank 6
    # But we can use a Strassen-like trick if we extend B to 2x2
    
    # Alternative: Use the fact that (2x2)*(2x2) has rank 7 (Strassen)
    # Pad B to be 2x2 by adding a zero column? No, that changes the problem.
    
    # Let me think differently. The schoolbook rank is n*m*p = 2*4*6 = 48.
    # We can use block Strassen: split the 4-dimension into 2+2.
    # A is 2x4 = [A1 | A2], each 2x2
    # B is 4x6 = [B1; B2], each 2x6
    # C = A1*B1 + A2*B2
    # Each (2x2)*(2x6) has schoolbook rank 12
    # Total = 24. Not better than 48? Wait, 12+12=24 < 48. Yes!
    
    # Can we do better with Strassen on each (2x2)*(2x6)?
    # (2x2)*(2x6): treat B as two 2x3 blocks? Or use Strassen on the 2x2 part.
    # Actually, for (2x2)*(2x6), we can think of it as 6 separate (2x2)*(2x1) multiplications.
    # Each (2x2)*(2x1) has schoolbook rank 4.
    # But we can batch: the 2x2 multiplication part is shared.
    # If we use Strassen on the 2x2 structure, we get 7 terms instead of 8 for the 2x2 part,
    # but each term still needs to multiply with the 2x6.
    
    # Let me just use the block decomposition: rank = 2 * (2*2*6) = 24? No.
    # (2x2)*(2x6) schoolbook: 2*2*6 = 24 terms. Two of those: 48. Same as before.
    
    # I need a better strategy. Let me use Strassen properly.
    # Split the m=4 dimension into two halves of size 2.
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # Now, A1*B1 is (2x2)*(2x6). We can write this as:
    #   Let B1 = [b11 b12 ... b16; b21 b22 ... b26]
    #   A1*B1 = [a11 a12; a21 a22] * B1
    #   = [a11*row1(B1) + a12*row2(B1); a21*row1(B1) + a22*row2(B1)]
    # Schoolbook: 4*6 = 24 terms for A1*B1, same for A2*B2, total 48.
    
    # To use Strassen, I need both matrices to be 2x2. So let me split p=6 too.
    # Split p=6 into 3+3.
    # B1 is 2x6 = [B1a | B1b] where B1a, B1b are 2x3. Still not 2x2.
    
    # Let me try: n=2, m=4, p=6.
    # Split m into 2+2, and p into 3+3.
    # A = [A1 | A2], A1,A2 are 2x2
    # B = [B1; B2], B1,B2 are 2x6
    # B1 = [B11 | B12], B11,B12 are 2x3
    # This is getting complicated. Let me just go with the simple block split.
    
    # Simple block: split m=4 into 2+2.
    # Rank = 2 * (2*2*6) = 24. Wait, that's not right either.
    # (2x2)*(2x6) has 2*2*6 = 24 scalar multiplications in schoolbook.
    # So two of them: 48. Same as original.
    
    # I think I need to actually implement Strassen for a rectangular case.
    # Let me try: treat the 2x4 * 4x6 as follows.
    # Pad A to 2x6 by adding zero columns? No.
    
    # Alternative known result: The rank of (2,4,6) tensor.
    # Let me just use a direct Strassen-like decomposition.
    
    # Here's a cleaner approach:
    # Split A (2x4) into two 2x2 blocks: A = [A_L | A_R]
    # Split B (4x6) into two 2x6 blocks: B = [B_T; B_B]
    # C = A_L * B_T + A_R * B_B
    # For A_L * B_T: (2x2)*(2x6). 
    # I can use the following trick: 
    #   A_L = [[a,b],[c,d]], B_T = [[e,f,g,h,i,j],[k,l,m,n,o,p]]
    #   A_L * B_T = [[ae+bk, af+bl, ag+bm, ah+bn, ai+bo, aj+bp],
    #                [ce+dk, cf+dl, cg+dm, ch+dn, ci+do, cj+dp]]
    # Schoolbook: 24 terms.
    # Can I do it in fewer? The 2x2 multiplication structure suggests 7 "macro" terms,
    # but each macro term involves a linear combination of the B entries.
    
    # Let me just go with rank 24 using a different split.
    # Actually, wait. Let me reconsider.
    
    # The schoolbook is 48. If I split m=4 into 2+2, I get two independent (2x2)*(2x6) 
    # multiplications, each of rank 12 in schoolbook (2*2*6=24? No, 2*2*6=24).
    # 24 + 24 = 48. No improvement.
    
    # The only way to improve is to use Strassen on a 2x2 * 2x2 subproblem.
    # So I need to also split p. Let me split p=6 into 3+3.
    # Then I have four (2x2)*(2x3) subproblems. Each has schoolbook rank 12.
    # 4*12 = 48. Still no improvement.
    
    # Split p=6 into 2+2+2.
    # Then I have six (2x2)*(2x2) subproblems. Each can be done in rank 7 (Strassen).
    # 6*7 = 42. That's an improvement!
    
    # Let's go with this:
    # A (2x4) = [A1 | A2], A1,A2 are 2x2
    # B (4x6) = [B1; B2], B1,B2 are 2x6
    # B1 = [B11 | B12 | B13], each 2x2
    # B2 = [B21 | B22 | B23], each 2x2
    # C = A1*B1 + A2*B2
    # C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    # Each term is (2x2)*(2x2), done in rank 7 via Strassen.
    # Total rank = 6 * 7 = 42.
    
    strassen_terms = _strassen_2x2_decomposition()
    
    # Build U, V, W matrices
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    
    n, m, p = 2, 4, 6
    rank = 6 * 7  # 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Indexing:
    # A is 2x4, row-major: A[i*m + j] for i in 0..1, j in 0..3
    #   A[0] = a00, A[1] = a01, A[2] = a02, A[3] = a03
    #   A[4] = a10, A[5] = a11, A[6] = a12, A[7] = a13
    # B is 4x6, row-major: B[j*p + k] for j in 0..3, k in 0..5
    #   B[0..5] = row 0 of B (j=0): b00,b01,b02,b03,b04,b05
    #   B[6..11] = row 1 of B (j=1): b10,b11,b12,b13,b14,b15
    #   B[12..17] = row 2 of B (j=2): b20,b21,b22,b23,b24,b25
    #   B[18..23] = row 3 of B (j=3): b30,b31,b32,b33,b34,b35
    # C is 2x6, output W[k*n + i] for k in 0..5, i in 0..1
    #   W[0] = c00, W[1] = c01, W[2] = c02, W[3] = c03, W[4] = c04, W[5] = c05
    #   W[6] = c10, W[7] = c11, W[8] = c12, W[9] = c13, W[10] = c14, W[11] = c15
    
    # Block structure:
    # A1 = A[:, 0:2] = [[a00, a01], [a10, a11]] -> indices in A: 0,1,4,5
    # A2 = A[:, 2:4] = [[a02, a03], [a12, a13]] -> indices in A: 2,3,6,7
    # B1 = B[0:2, :] = rows 0,1 of B -> indices in B: 0-5, 6-11
    #   B11 = B[0:2, 0:2] -> B indices: 0,1,6,7
    #   B12 = B[0:2, 2:4] -> B indices: 2,3,8,9
    #   B13 = B[0:2, 4:6] -> B indices: 4,5,10,11
    # B2 = B[2:4, :] = rows 2,3 of B -> indices in B: 12-17, 18-23
    #   B21 = B[2:4, 0:2] -> B indices: 12,13,18,19
    #   B22 = B[2:4, 2:4] -> B indices: 14,15,20,21
    #   B23 = B[2:4, 4:6] -> B indices: 16,17,22,23
    
    # For each of the 6 blocks (A1 or A2) * (B11, B12, B13, B21, B22, B23):
    #   Use 7 Strassen terms.
    
    # Map: block_index -> (A_block, B_block)
    # A_block: 0 for A1, 1 for A2
    # B_block: 0 for B11, 1 for B12, 2 for B13, 3 for B21, 4 for B22, 5 for B23
    
    # A1 indices in U: [0, 1, 4, 5] corresponding to [a00, a01, a10, a11]
    # A2 indices in U: [2, 3, 6, 7] corresponding to [a02, a03, a12, a13]
    
    A1_indices = [0, 1, 4, 5]  # a00, a01, a10, a11
    A2_indices = [2, 3, 6, 7]  # a02, a03, a12, a13
    
    # B block indices in V (24 entries):
    # B11: B indices 0,1,6,7 -> [b00, b01, b10, b11]
    # B12: B indices 2,3,8,9 -> [b02, b03, b12, b13]
    # B13: B indices 4,5,10,11 -> [b04, b05, b14, b15]
    # B21: B indices 12,13,18,19 -> [b20, b21, b30, b31]
    # B22: B indices 14,15,20,21 -> [b22, b23, b32, b33]
    # B23: B indices 16,17,22,23 -> [b24, b25, b34, b35]
    
    B_block_indices = [
        [0, 1, 6, 7],   # B11: b00, b01, b10, b11
        [2, 3, 8, 9],   # B12: b02, b03, b12, b13
        [4, 5, 10, 11], # B13: b04, b05, b14, b15
        [12, 13, 18, 19], # B21: b20, b21, b30, b31
        [14, 15, 20, 21], # B22: b22, b23, b32, b33
        [16, 17, 22, 23], # B23: b24, b25, b34, b35
    ]
    
    # C output indices in W (12 entries): W[k*n + i] where k=col of C (0..5), i=row of C (0..1)
    # C is 2x6. 
    # C[0,0]=c00 -> W[0], C[0,1]=c01 -> W[1], ..., C[0,5]=c05 -> W[5]
    # C[1,0]=c10 -> W[6], C[1,1]=c11 -> W[7], ..., C[1,5]=c15 -> W[11]
    
    # For block (A_x, B_y) where x in {0,1}, y in {0,1,2,3,4,5}:
    #   The result is a 2x2 matrix C_block.
    #   If y=0 (B11): C_block contributes to C[0:2, 0:2] -> W indices [0,1,6,7]
    #   If y=1 (B12): C_block contributes to C[0:2, 2:4] -> W indices [2,3,8,9]
    #   If y=2 (B13): C_block contributes to C[0:2, 4:6] -> W indices [4,5,10,11]
    #   If y=3 (B21): C_block contributes to C[0:2, 0:2] -> W indices [0,1,6,7]
    #   If y=4 (B22): C_block contributes to C[0:2, 2:4] -> W indices [2,3,8,9]
    #   If y=5 (B23): C_block contributes to C[0:2, 4:6] -> W indices [4,5,10,11]
    
    C_block_indices = [
        [0, 1, 6, 7],   # cols 0-1
        [2, 3, 8, 9],   # cols 2-3
        [4, 5, 10, 11], # cols 4-5
        [0, 1, 6, 7],   # cols 0-1
        [2, 3, 8, 9],   # cols 2-3
        [4, 5, 10, 11], # cols 4-5
    ]
    
    # Now fill in U, V, W
    col = 0
    for x in range(2):  # A block: 0=A1, 1=A2
        A_indices = A1_indices if x == 0 else A2_indices
        for y in range(6):  # B block
            B_indices = B_block_indices[y]
            C_indices = C_block_indices[y]
            
            for (u_vec, v_vec, w_vec) in strassen_terms:
                # u_vec has 4 entries: [a00, a01, a10, a11] or [a02, a03, a12, a13]
                for idx_in_block, u_val in enumerate(u_vec):
                    if u_val != 0:
                        U[A_indices[idx_in_block]][col] = u_val
                
                # v_vec has 4 entries: [b00, b01, b10, b11] or similar
                for idx_in_block, v_val in enumerate(v_vec):
                    if v_val != 0:
                        V[B_indices[idx_in_block]][col] = v_val
                
                # w_vec has 4 entries: [c00, c01, c10, c11]
                for idx_in_block, w_val in enumerate(w_vec):
                    if w_val != 0:
                        W[C_indices[idx_in_block]][col] = w_val
                
                col += 1
    
    # Verify: rank should be 42
    assert col == 42, f"Expected rank 42, got {col}"
    
    # Verify correctness with random test
    if time.time() < deadline:
        try:
            np.random.seed(42)
            A_test = np.random.randn(2, 4)
            B_test = np.random.randn(4, 6)
            C_expected = A_test @ B_test
            
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Reconstruct C from decomposition
            # C[i,k] = sum_c U[i*m + ... ] ... 
            # Actually: C[i,k] = sum_c (sum_j A[i,j]*V[j*p+k, c]) * (sum_j A[i,j]*U[i*m+j, c])? No.
            # The tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]?
            # No, the decomposition is: the multiplication tensor T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
            # where T_{i,j,k} = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
            # Actually, the matrix multiplication tensor has T_{i,j,k} = 1 for all valid (i,j,k) in the 
            # "multiplication" sense. Let me just verify by computing C from the factors.
            
            # C[i,k] should equal sum_c (sum_j A[i,j] * V[j*p+k, c]) * (something)?
            # Let me think: The rank-r decomposition means:
            # C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k] = sum_c sum_j U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # Wait, that's not quite right. The standard form is:
            # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]?
            # No. Let me re-derive.
            
            # The tensor for matrix multiplication: T is a 3-tensor with T[i,j,k] = 1 
            # (it's the identity-like tensor for the bilinear form).
            # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
            # (indices adjusted for our convention)
            
            # In our convention: U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
            # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
            # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (the multiplication tensor)
            
            # To compute C = A*B:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k] / (U[i*m+j,c]*V[j*p+k,c])
            # This is getting circular. Let me just do the direct computation.
            
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])
            # No, that's not right either.
            
            # The correct interpretation: 
            # Let X_c = sum_j A[i,j] * U[i*m+j, c] for each i, c
            # Let Y_c = sum_j B[j,k] * V[j*p+k, c] for each k, c
            # Then C[i,k] = sum_c X_c * Y_c * W[k*n+i, c]
            
            # Let me verify this.
            C_computed = np.zeros((2, 6))
            for c in range(42):
                for i in range(2):
                    X_c = 0.0
                    for j in range(4):
                        X_c += A_test[i, j] * U_np[i * 4 + j, c]
                    for k in range(6):
                        Y_c = 0.0
                        for j in range(4):
                            Y_c += B_test[j, k] * V_np[j * 6 + k, c]
                        C_computed[i, k] += X_c * Y_c * W_np[k * 2 + i, c]
            
            error = np.max(np.abs(C_computed - C_expected))
            if error > 1e-10:
                # Fallback to schoolbook if Strassen is wrong
                rank_school = n * m * p
                U_school = [[0] * rank_school for _ in range(n * m)]
                V_school = [[0] * rank_school for _ in range(m * p)]
                W_school = [[0] * rank_school for _ in range(p * n)]
                col_s = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i * m + j][col_s] = 1
                            V_school[j * p + k][col_s] = 1
                            W_school[k * n + i][col_s] = 1
                            col_s += 1
                return {"U": U_school, "V": V_school, "W": W_school}
        except Exception:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col_s] = 1
                        V_school[j * p + k][col_s] = 1
                        W_school[k * n + i][col_s] = 1
                        col_s += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

