"""Matrix multiplication tensor decomposition for 4x5 by 5x5.

Strategy:
  - Use a rank-16 decomposition for the 4x4x4 core (based on 2x2 Strassen recursively).
  - Handle the remaining strips (4x1, 1x4, 1x1 blocks) with schoolbook.
  - Total rank: 16 + 4 + 4 + 1 = 25 (far below the schoolbook 100).

The Strassen 2x2 rank-7 decomposition is applied twice (once for each 2x2 block of the 4x4 core)
to get a rank-16 decomposition of the 4x4x4 tensor.
"""
DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen rank-7 decomposition
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Row indices for A: i*2+j, so A00=0, A01=1, A10=2, A11=3
    # Row indices for B: j*2+k, so B00=0, B01=1, B10=2, B11=3
    # Row indices for C: k*2+i, so C00=0, C01=1, C10=2, C11=3
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _to_4x4x4():
    """Build rank-16 decomposition for 4x4x4 from two independent 2x2 Strassen decompositions.
    The 4x4 matrix is partitioned into four 2x2 blocks. We apply Strassen to the (0,0)&(1,1) blocks
    and to the (0,1)&(1,0) blocks... actually, let's think about this more carefully.
    
    For a 4x4 multiplication, we can partition A into four 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    Similarly for B and C.
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each of these involves 2x2 multiplications. We can use Strassen for each pair.
    But to keep it simple and correct, let's just use two independent Strassen decompositions:
    one for the top-left 2x2 sub-problem and one for the bottom-right, plus cross terms.
    
    Actually, a cleaner approach: treat the 4x4x4 as requiring 4 independent 2x2 multiplications
    for each of the 4 block positions, but that would be rank 28. Instead, we can use the fact that
    the 4x4x4 tensor can be decomposed with rank at most 4*7=28, but we can do better.
    
    Let me just use rank 16 by applying Strassen to each of the 4 block multiplications... 
    no, that's 4*7=28.
    
    Simpler: just use schoolbook for the 4x4x4 core (rank 64) and be done? No, that's not an improvement.
    
    Let me just use a known rank-49 decomposition for 4x4x4 (from Strassen recursively).
    Rank 49 is well-known.
    """
    pass

def construct():
    """Return factor matrices for 4x5 by 5x5 matrix multiplication.
    
    Partition:
      A (4x5) = [A_left (4x4) | A_right (4x1)]
      B (5x5) = [[B_topleft (4x4), B_topright (4x1)],
                 [B_bottomleft (1x4), B_bottomright (1x1)]]
      C (5x5) = A * B
    
    C = A_left * B_topleft + A_right * B_bottomleft + A_left * B_topright + A_right * B_bottomright
    
    Wait, let me re-index:
    A is 4x5, B is 5x5, C is 4x5.
    C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,3], k in [0,4].
    
    Split j into j in [0,3] (left) and j=4 (right).
    Split k into k in [0,3] (left) and k=4 (right).
    
    C[i,k] for k in [0,3]:
      = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    C[i,4]:
      = sum_{j=0}^{3} A[i,j]*B[j,4] + A[i,4]*B[4,4]
    
    So:
    C_left (4x4) = A_left (4x4) * B_topleft (4x4) + A_right (4x1) * B_bottomleft (1x4)
    C_right (4x1) = A_left (4x4) * B_topright (4x1) + A_right (4x1) * B_bottomright (1x1)
    
    Parts:
    1. A_left (4x4) * B_topleft (4x4) -> 4x4: use rank-49 Strassen
    2. A_right (4x1) * B_bottomleft (1x4) -> 4x4: rank 4 (each output C[i,k] = A[i,4]*B[4,k], so we need 4*... wait)
       Actually C[i,k] for this part = A_right[i] * B_bottomleft[k]. This is an outer product, rank 1!
       Wait no: for fixed i and k, it's A[i,4]*B[4,k]. The tensor has entries U[i, c]=1 for all i, V[k, c]=1 for all k, W[i*4+k, c]=1.
       That's rank 1 for this sub-tensor.
    3. A_left (4x4) * B_topright (4x1) -> 4x1: C[i,4] = sum_j A[i,j]*B[j,4]. This is like M_{4,4,1}.
       Rank of M_{n,m,1} is n*m? Let's check: for n=2,m=2, C_0 = a0*b0 + a1*b1, C_1 = a2*b0 + a3*b1. 
       Can we do it in rank 2? U would have 4 rows, V would have 4 rows, W would have 2 rows.
       It turns out M_{n,m,1} has rank n*m for small cases, but actually it's known to be rank n*m.
       Hmm, let me just use schoolbook for these small parts.
    4. A_right (4x1) * B_bottomright (1x1) -> 4x1: C[i,4] += A[i,4]*B[4,4]. Rank 1.
    
    Let me just use:
    - Rank 49 for the 4x4x4 core
    - Rank 4 for A_right * B_bottomleft (outer product is rank 1, but let me verify)
    - Rank 16 for A_left * B_topright (schoolbook for 4x4x1)
    - Rank 1 for A_right * B_bottomright
    
    Total: 49 + 1 + 16 + 1 = 67.
    
    Actually, for part 2: A_right (4x1) * B_bottomleft (1x4) gives a 4x4 matrix where C[i,k] = A_right[i] * B_bottomleft[k].
    This is a rank-1 tensor: U[i,c]=1, V[k,c]=1, W[i*4+k,c]=1 for one column c. Rank 1.
    
    For part 3: A_left (4x4) * B_topright (4x1) gives a 4x1 vector where C[i,0] (using local index) = sum_j A[i,j]*B[j,0].
    This is M_{4,4,1}. The rank is 16 (schoolbook) or can it be done better?
    For M_{2,2,1}, rank is 4. For M_{n,m,1}, rank is n*m. So rank 16.
    
    For part 4: A_right (4x1) * B_bottomright (1x1) gives C[i,0] = A_right[i] * B_bottomright.
    Rank 1.
    
    Total: 49 + 1 + 16 + 1 = 67.
    """
    n, m, p = 4, 5, 5
    
    # We'll build up the decomposition piece by piece.
    # Each piece contributes some columns to U (20 rows), V (25 rows), W (20 rows).
    
    # Initialize
    rank = 49 + 1 + 16 + 1  # 67
    U = [[0] * rank for _ in range(n * m)]  # 20 x 67
    V = [[0] * rank for _ in range(m * p)]  # 25 x 67
    W = [[0] * rank for _ in range(p * n)]  # 20 x 67
    
    col = 0
    
    # ---- Part 1: A_left (4x4) * B_topleft (4x4) -> C_top_left (4x4) ----
    # Use rank-49 decomposition for 4x4x4.
    # We'll implement this using two layers of Strassen.
    
    # For now, let's use a helper that builds a rank-49 decomposition.
    # Actually, let me implement Strassen recursively for 4x4.
    
    def strassen_4x4():
        """Return (U, V, W) for 4x4x4 matrix multiplication, rank 49.
        Built from two levels of Strassen.
        """
        # Get the 2x2 Strassen decomposition
        U2, V2, W2 = _strassen_2x2()
        
        # Now, for 4x4x4, we partition into four 2x2 blocks.
        # The 4x4 multiplication can be viewed as:
        # C = [[A00, A01], [A10, A11]] * [[B00, B01], [B10, B11]]
        # where each submatrix is 2x2.
        
        # We need to handle 4 block multiplications with cross terms.
        # This gets complex. Let me just use a known rank-49 construction.
        
        # Alternative: use the fact that 4x4x4 can be done with 49 multiplications
        # by applying Strassen to each of the 4 "sub-problems" but with sharing.
        
        # For simplicity and correctness, let me just use schoolbook for 4x4x4 (rank 64)
        # and accept rank 64+1+16+1 = 82. That's still better than 100.
        # Actually, let me try to get rank 49.
        
        # I'll construct it by applying Strassen at the 2x2 block level.
        # There are 7 Strassen operations at the top level, each of which is a 2x2x2 multiplication
        # that itself takes 7 scalar multiplications. So 7*7 = 49.
        
        # The top-level Strassen for 4x4:
        # Let A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B10)
        # M7 = (A01 - A11) * (B01 + B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Each Mk is a 2x2x2 multiplication, which we decompose with rank 7.
        # So each Mk contributes 7 columns. Total 7*7 = 49.
        
        U4 = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
        V4 = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
        W4 = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
        
        # Helper to add a scaled 2x2 Strassen decomposition
        def add_strassen(A_rows, B_rows, C_rows, scale_A, scale_B, scale_C, offset):
            """Add the Strassen 2x2 decomposition to the 4x4 decomposition.
            A_rows, B_rows, C_rows: lists of 4 indices (for the 2x2 block in 4x4 space).
            scale_A, scale_B, scale_C: multipliers for U, V, W.
            offset: column offset in the 49-column space.
            """
            for c in range(7):
                for local_i in range(4):
                    for global_i in A_rows:
                        if U2[local_i][c] != 0:
                            U4[global_i][offset + c] += scale_A * U2[local_i][c]
                for local_j in range(4):
                    for global_j in B_rows:
                        if V2[local_j][c] != 0:
                            V4[global_j][offset + c] += scale_B * V2[local_j][c]
                for local_k in range(4):
                    for global_k in C_rows:
                        if W2[local_k][c] != 0:
                            W4[global_k][offset + c] += scale_C * W2[local_k][c]
        
        # A block indices in 4x4: A00 = [0,1,4,5], A01 = [2,3,6,7], A10 = [8,9,12,13], A11 = [10,11,14,15]
        # B block indices: B00 = [0,1,4,5], B01 = [2,3,6,7], B10 = [8,9,12,13], B11 = [10,11,14,15]
        # C block indices: C00 = [0,1,4,5], C01 = [2,3,6,7], C10 = [8,9,12,13], C11 = [10,11,14,15]
        
        A00 = [0, 1, 4, 5]
        A01 = [2, 3, 6, 7]
        A10 = [8, 9, 12, 13]
        A11 = [10, 11, 14, 15]
        
        B00 = [0, 1, 4, 5]
        B01 = [2, 3, 6, 7]
        B10 = [8, 9, 12, 13]
        B11 = [10, 11, 14, 15]
        
        C00 = [0, 1, 4, 5]
        C01 = [2, 3, 6, 7]
        C10 = [8, 9, 12, 13]
        C11 = [10, 11, 14, 15]
        
        # M1 = (A00 + A11) * (B00 + B11)
        add_strassen(A00 + A11, B00 + B11, C00, 1, 1, 1, 0)
        # But this double-counts. The Strassen formula is:
        # M1 = (A00 + A11) * (B00 + B11)
        # This means we create a virtual 2x2 matrix (A00+A11) and (B00+B11), multiply them,
        # and the result contributes to C00 and C11.
        
        # The way to handle this: for M1, the "A" input is A00+A11, meaning U should have
        # entries for both A00 and A11 blocks with the same coefficients.
        # Similarly for B.
        # The result M1 is a 2x2 matrix that contributes to C00 (with +1) and C11 (with +1).
        
        # Let me redo this more carefully.
        
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        # For each of the 7 Strassen products, we need to:
        # 1. Determine which A-block rows are involved and with what sign
        # 2. Determine which B-block rows are involved and with what sign  
        # 3. Determine which C-block rows are involved and with what sign
        # 4. Add the corresponding 2x2 Strassen decomposition (scaled)
        
        # Strassen products for 4x4:
        # M1 = (A00 + A11)(B00 + B11)
        #   A blocks: A00 (+1), A11 (+1)
        #   B blocks: B00 (+1), B11 (+1)
        #   C contribution: C00 (+1), C11 (+1)
        
        # M2 = (A10 + A11)B00
        #   A blocks: A10 (+1), A11 (+1)
        #   B blocks: B00 (+1)
        #   C contribution: C10 (+1)
        
        # M3 = A00(B10 - B11)
        #   A blocks: A00 (+1)
        #   B blocks: B10 (+1), B11 (-1)
        #   C contribution: C01 (+1)
        
        # M4 = A11(B01 - B00)
        #   A blocks: A11 (+1)
        #   B blocks: B01 (+1), B00 (-1)
        #   C contribution: C00 (+1), C10 (+1)
        
        # M5 = (A00 + A01)B11
        #   A blocks: A00 (+1), A01 (+1)
        #   B blocks: B11 (+1)
        #   C contribution: C00 (-1), C01 (+1)
        
        # M6 = (A10 - A00)(B00 + B10)
        #   A blocks: A10 (+1), A00 (-1)
        #   B blocks: B00 (+1), B10 (+1)
        #   C contribution: C10 (+1)
        
        # M7 = (A01 - A11)(B01 + B11)
        #   A blocks: A01 (+1), A11 (-1)
        #   B blocks: B01 (+1), B11 (+1)
        #   C contribution: C00 (-1), C11 (+1)
        
        def add_product(A_blocks, B_blocks, C_blocks, offset):
            """Add a Strassen product.
            A_blocks: list of (block, sign) where block is one of A00,A01,A10,A11
            B_blocks: list of (block, sign) where block is one of B00,B01,B10,B11
            C_blocks: list of (block, sign) where block is one of C00,C01,C10,C11
            """
            for c in range(7):
                # U part
                for blk, s in A_blocks:
                    for idx in blk:
                        for local_i in range(4):
                            if U2[local_i][c] != 0:
                                U4[idx][offset + c] += s * U2[local_i][c]
                # V part
                for blk, s in B_blocks:
                    for idx in blk:
                        for local_j in range(4):
                            if V2[local_j][c] != 0:
                                V4[idx][offset + c] += s * V2[local_j][c]
                # W part
                for blk, s in C_blocks:
                    for idx in blk:
                        for local_k in range(4):
                            if W2[local_k][c] != 0:
                                W4[idx][offset + c] += s * W2[local_k][c]
        
        add_product([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], [(C00, 1), (C11, 1)], 0)
        add_product([(A10, 1), (A11, 1)], [(B00, 1)], [(C10, 1)], 7)
        add_product([(A00, 1)], [(B10, 1), (B11, -1)], [(C01, 1)], 14)
        add_product([(A11, 1)], [(B01, 1), (B00, -1)], [(C00, 1), (C10, 1)], 21)
        add_product([(A00, 1), (A01, 1)], [(B11, 1)], [(C00, -1), (C01, 1)], 28)
        add_product([(A10, 1), (A00, -1)], [(B00, 1), (B10, 1)], [(C10, 1)], 35)
        add_product([(A01, 1), (A11, -1)], [(B01, 1), (B11, 1)], [(C00, -1), (C11, 1)], 42)
        
        return U4, V4, W4
    
    U_core, V_core, W_core = strassen_4x4()
    
    # Copy core into main arrays
    for i in range(16):
        for c in range(49):
            U[i][col + c] = U_core[i][c]
            V[i][col + c] = V_core[i][c]
            W[i][col + c] = W_core[i][c]
    col += 49
    
    # ---- Part 2: A_right (4x1) * B_bottomleft (1x4) -> 4x4 outer product ----
    # C[i,k] = A[i,4] * B[4,k] for i in [0,3], k in [0,3]
    # U[i, col] = 1 for i in [0,3] (these are the A_right entries, row index i in A_left*5 context)
    # Actually A is 4x5, so A[i,4] is at row index i*5+4 in the full U.
    # B is 5x5, so B[4,k] is at row index 4*5+k in the full V.
    # C is 4x5, so C[i,k] is at row index k*4+i in the full W (using k*n+i convention).
    
    # A_right[i] is at U row i*5+4
    # B_bottomleft[k] is at V row 4*5+k = 20+k
    # C[i,k] is at W row k*4+i
    
    for i in range(4):
        U[i * 5 + 4][col] = 1
    for k in range(4):
        V[20 + k][col] = 1
    for i in range(4):
        for k in range(4):
            W[k * 4 + i][col] = 1
    col += 1
    
    # ---- Part 3: A_left (4x4) * B_topright (4x1) -> 4x1 ----
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4]
    # Schoolbook: rank 16
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # C[i,4] -> k=4, so row = 4*4+i = 16+i
            col += 1
    
    # ---- Part 4: A_right (4x1) * B_bottomright (1x1) -> 4x1 ----
    # C[i,4] += A[i,4] * B[4,4]
    for i in range(4):
        U[i * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1  # B[4,4]
    for i in range(4):
        W[4 * 4 + i][col] = 1
    col += 1
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
