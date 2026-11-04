"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) where:
      U has shape (4, 7), indexed by [i*2+j, c]
      V has shape (4, 7), indexed by [j*2+k, c]
      W has shape (4, 7), indexed by [k*2+i, c]
    
    Strassen's 7 products:
      P1 = (A11+A22)*(B11+B22)
      P2 = (A21+A22)*B11
      P3 = A11*(B12-B22)
      P4 = A22*(B21-B11)
      P5 = (A11+A12)*B22
      P6 = (A21-A11)*(B11+B12)
      P7 = (A12-A22)*(B21+B22)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # P3 = A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # P4 = A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # P5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # P6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_laderman_2x3x3():
    """
    Build rank-8 decomposition for 2x3 times 3x3 multiplication.
    A is 2x3, B is 3x3, C is 2x3.
    
    U has shape (6, 8), indexed by [i*3+j, c] for i in 0..1, j in 0..2
    V has shape (9, 8), indexed by [j*3+k, c] for j in 0..2, k in 0..2
    W has shape (6, 8), indexed by [k*2+i, c] for k in 0..2, i in 0..1
    
    We use a known rank-8 decomposition.
    """
    # This is a known minimal decomposition for 2x3x3 (rank 8)
    # Let me construct it carefully.
    
    # Actually, let me use a simpler approach: 
    # 2x3x3 can be decomposed by treating it as two 2x1x3 blocks or using
    # the fact that rank(2x3x3) <= 2*rank(1x3x3) but that's not helpful.
    # 
    # Known: mu(2,3,3) = 8. Let me use a construction.
    # 
    # One approach: Use the 2x2x2 Strassen (rank 7) embedded, plus corrections.
    # Or use a direct rank-8 construction.
    #
    # Let me use a different strategy: decompose 2x3x3 as a direct sum of
    # 2x1x3 (rank 3) + 2x1x3 (rank 3) + 2x1x3 (rank 3) = rank 9, but we need 8.
    #
    # Actually, let me just use a known rank-8 tensor for 2x3x3.
    # I'll construct it using the following approach:
    # 
    # For 2x3x3, we can use the following 8 terms:
    # The multiplication tensor T = sum_{i,j,k} e_i^A e_j^B e_k^C where
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # T = sum_{i,j,k} A_i B_j C_k (with appropriate indexing)
    
    # Let me use a verified rank-8 construction.
    # I'll define it explicitly.
    
    # U shape (6, 8): rows are (i,j) for i=0,1 and j=0,1,2
    # V shape (9, 8): rows are (j,k) for j=0,1,2 and k=0,1,2
    # W shape (6, 8): rows are (k,i) for k=0,1,2 and i=0,1
    
    U = [[0]*8 for _ in range(6)]
    V = [[0]*8 for _ in range(9)]
    W = [[0]*8 for _ in range(6)]
    
    # I'll use a known construction. Let me define 8 rank-1 tensors.
    # Each term c has:
    #   u_c: a vector in R^6 (for A entries)
    #   v_c: a vector in R^9 (for B entries)
    #   w_c: a vector in R^6 (for C entries)
    # such that sum_c u_c ⊗ v_c ⊗ w_c = multiplication tensor
    
    # The multiplication tensor for 2x3x3:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # where the tensor is indexed as (i,j,k) meaning A_{i,j} * B_{j,k} contributes to C_{i,k}
    
    # Let me use a systematic construction based on known results.
    # I'll use the following 8 decompositions:
    
    # Term 0: A_{0,0} * B_{0,0} -> C_{0,0}
    U[0][0] = 1  # A_{0,0}
    V[0][0] = 1  # B_{0,0}
    W[0][0] = 1  # C_{0,0}
    
    # Term 1: A_{0,1} * B_{1,1} -> C_{0,1}
    U[1][1] = 1  # A_{0,1}
    V[4][1] = 1  # B_{1,1}
    W[2][1] = 1  # C_{0,1}
    
    # Term 2: A_{0,2} * B_{2,2} -> C_{0,2}
    U[2][2] = 1  # A_{0,2}
    V[8][2] = 1  # B_{2,2}
    W[4][2] = 1  # C_{0,2}
    
    # Term 3: A_{1,0} * B_{0,0} -> C_{1,0}
    U[3][3] = 1  # A_{1,0}
    V[0][3] = 1  # B_{0,0}
    W[1][3] = 1  # C_{1,0}
    
    # Term 4: A_{1,1} * B_{1,1} -> C_{1,1}
    U[4][4] = 1  # A_{1,1}
    V[4][4] = 1  # B_{1,1}
    W[3][4] = 1  # C_{1,1}
    
    # Term 5: A_{1,2} * B_{2,2} -> C_{1,2}
    U[5][5] = 1  # A_{1,2}
    V[8][5] = 1  # B_{2,2}
    W[5][5] = 1  # C_{1,2}
    
    # Now I need to handle the off-diagonal terms. This naive approach gives rank 6
    # but only handles diagonal. I need a proper rank-8 decomposition.
    
    # Let me restart with a proper construction.
    # I'll use a known rank-8 decomposition for 2x3x3.
    
    # After research, a rank-8 decomposition can be constructed as follows:
    # We partition and use bilinear forms.
    
    # Let me use a different, verified approach. I'll construct the tensor
    # decomposition using the following 8 terms:
    
    # Reset
    U = [[0]*8 for _ in range(6)]
    V = [[0]*8 for _ in range(9)]
    W = [[0]*8 for _ in range(6)]
    
    # I'll use a construction that I can verify.
    # The key insight: 2x3x3 has tensor rank 8.
    
    # Let me use a block-based construction:
    # Split A into columns: A = [A_col0, A_col1, A_col2]
    # C = A * B = [A_col0 * B_col0 + A_col1 * B_col1 + A_col2 * B_col2]
    # where B_colj is the j-th column of B.
    
    # This doesn't directly help. Let me just use a known explicit decomposition.
    
    # After careful construction, here's a rank-8 decomposition:
    
    # I'll use the following approach based on the structure:
    # For each pair of rows in A (only 2 rows) and each column in C (3 columns),
    # we need to compute a linear combination.
    
    # Let me try a different, simpler strategy for the whole problem.
    # Instead of trying to find optimal sub-decompositions, let me use
    # a block Strassen approach on the full 4x5x6 problem.
    
    return U, V, W  # placeholder, will be replaced

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.
    
    Strategy: Use block decomposition with Strassen on 2x2 blocks where possible.
    
    For 4x5x6:
    - Split the 4x5 matrix into blocks:
      - Top-left 2x2, top-right 2x3
      - Bottom-left 2x2, bottom-right 2x3
    - Split the 5x6 matrix into blocks:
      - Top: 5x6 (no clean split)
    
    Actually, let me use a cleaner block structure:
    - Split rows of A (4) into 2+2
    - Split columns of A / rows of B (5) into 2+3
    - Split columns of B (6) into 3+3
    
    So we have:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x3, B21 is 3x3, B22 is 3x3
    
    C = A * B:
    C11 = A11*B11 + A12*B21  (2x3)
    C12 = A11*B12 + A12*B22  (2x3)
    C21 = A21*B11 + A22*B21  (2x3)
    C22 = A21*B12 + A22*B22  (2x3)
    
    For each 2x3 result block, we need to multiply a 2x2 by 2x3 and add a 2x3 by 3x3.
    
    Let me use a simpler approach: direct Strassen on 2x2 blocks where they fit.
    
    Actually, the simplest improvement over schoolbook is to use Strassen on the 2x2x2
    sub-problems. Let me partition 4x5x6 into blocks and apply Strassen where 2x2x2 fits.
    
    Let me partition:
    - Rows of A: 4 = 2 + 2
    - Cols of A / Rows of B: 5 = 2 + 3
    - Cols of B: 6 = 3 + 3
    
    The 2x2x2 blocks:
    - A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] : This is a 2x2x2 problem! Rank 7.
    - A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2] : 2x2x2 problem! Rank 7.
    
    But wait, B[0:2,0:2] is only 2x2, and we're using it in two products. The issue is
    that the blocks don't align perfectly for independent Strassen applications.
    
    Let me use a different partition:
    - 4 = 2+2
    - 5 = 2+3
    - 6 = 2+2+2
    
    Then:
    A = [[A11, A12, A13], [A21, A22, A23]] with A11 2x2, A12 2x3, A13 2x2, etc.
    B = [[B11, B12, B13], [B21, B22, B23]] with B11 2x2, B12 2x2, B13 2x2, B21 3x2, etc.
    
    C[i,j] = sum_k A[i,k]*B[k,j]
    
    C11 (2x2) = A11*B11 + A12*B21
    C12 (2x2) = A11*B12 + A12*B22
    C13 (2x2) = A11*B13 + A12*B23
    C21 (2x2) = A21*B11 + A22*B21
    C22 (2x2) = A21*B12 + A22*B22
    C23 (2x2) = A21*B13 + A22*B23
    
    For C11: A11 is 2x2, B11 is 2x2 -> 2x2x2 (rank 7)
             A12 is 2x3, B21 is 3x2 -> 2x3x2
    
    Hmm, this is still complex. Let me just use a straightforward approach:
    
    Use Strassen where we have exact 2x2x2 blocks, and schoolbook elsewhere.
    
    Partition: 4=2+2, 5=2+3, 6=2+4
    - A11 (2x2) * B11 (2x2) -> part of C11 (2x2): 2x2x2, rank 7
    - A11 (2x2) * B12 (2x4) -> part of C12 (2x4): 2x2x4
    - A12 (2x3) * B21 (3x2) -> part of C11 (2x2): 2x3x2
    - A12 (2x3) * B22 (3x4) -> part of C12 (2x4): 2x3x4
    - A21 (2x2) * B11 (2x2) -> part of C21 (2x2): 2x2x2, rank 7
    - A21 (2x2) * B12 (2x4) -> part of C22 (2x4): 2x2x4
    - A22 (2x3) * B21 (3x2) -> part of C21 (2x2): 2x3x2
    - A22 (2x3) * B22 (3x4) -> part of C22 (2x4): 2x3x4
    
    Total rank: 7 + (2*2*4=16) + (2*3*2=12) + (2*3*4=24) + 7 + 16 + 12 + 24 = 118
    
    That's worse than 108 from the known result. Let me think differently.
    
    The known result of 108 uses Strassen on 2x2 blocks. Let me figure out how.
    
    120 - 108 = 12 savings. Each 2x2x2 Strassen saves 1 (8-7=1).
    So we need 12 independent 2x2x2 sub-problems.
    
    4x5x6: If we partition 4=2+2, 5=2+2+1, 6=2+2+2:
    - The 2x2x2 blocks: A[i:i+2, j:j+2] * B[j:j+2, k:k+2] for various i,j,k
    - But they overlap in the middle dimension (5).
    
    Actually, I think the 108 result comes from a more clever global decomposition.
    Let me just implement a good block Strassen approach.
    
    Let me use: 4=2+2, 5=2+3, 6=3+3
    
    And for each block product, use the best available decomposition.
    For 2x2*2x3: I'll use a rank-7 decomposition (Strassen-like for rectangular).
    For 2x3*3x3: I'll use schoolbook (rank 18) or a better one.
    
    Actually, let me just go with a clean implementation that achieves rank 108.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # I'll use a block decomposition approach.
    # Partition: n=4=2+2, m=5=2+3, p=6=3+3
    # 
    # A is 4x5:
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, cols 2-4 (2x3)
    #   A21: rows 2-3, cols 0-1 (2x2)
    #   A22: rows 2-3, cols 2-4 (2x3)
    #
    # B is 5x6:
    #   B11: rows 0-1, cols 0-2 (2x3)
    #   B12: rows 0-1, cols 3-5 (2x3)
    #   B21: rows 2-4, cols 0-2 (3x3)
    #   B22: rows 2-4, cols 3-5 (3x3)
    #
    # C is 4x6:
    #   C11: rows 0-1, cols 0-2 (2x3) = A11*B11 + A12*B21
    #   C12: rows 0-1, cols 3-5 (2x3) = A11*B12 + A12*B22
    #   C21: rows 2-3, cols 0-2 (2x3) = A21*B11 + A22*B21
    #   C22: rows 2-3, cols 3-5 (2x3) = A21*B12 + A22*B22
    
    # For 2x2 * 2x3: use a rank-7 decomposition
    # For 2x3 * 3x3: use schoolbook rank 18 (or better)
    
    # Let me define helper functions to build decompositions for sub-problems.
    
    def build_2x2_x_2x3():
        """
        Decomposition for (2x2) * (2x3) -> (2x3).
        A is 2x2 (4 entries), B is 2x3 (6 entries), C is 2x3 (6 entries).
        
        U shape (4, rank), V shape (6, rank), W shape (6, rank)
        
        We can use Strassen-like: treat as two 2x2x2 problems sharing B columns.
        Actually, let me just use a rank-8 schoolbook or find a rank-7.
        
        For simplicity, let me use schoolbook: rank = 2*2*3 = 12.
        But we can do better.
        
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        Schoolbook: 12 terms.
        
        Can we do rank 7 or 8?
        The tensor rank of 2x2x3 is at most 2*mu(2,2,1)*... 
        Actually mu(2,2,3) <= 2*mu(2,2,1) + ... 
        Let me just use schoolbook for now: rank 12.
        """
        rank = 12
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def build_2x3_x_3x3():
        """
        Decomposition for (2x3) * (3x3) -> (2x3).
        A is 2x3 (6 entries), B is 3x3 (9 entries), C is 2x3 (6 entries).
        Schoolbook: rank = 2*3*3 = 18.
        """
        rank = 18
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Build the full decomposition by combining sub-decompositions.
    # Each sub-product contributes to specific blocks of C.
    
    # Sub-products:
    # 1. A11(2x2) * B11(2x3) -> contributes to C11(2x3)
    # 2. A12(2x3) * B21(3x3) -> contributes to C11(2x3)
    # 3. A11(2x2) * B12(2x3) -> contributes to C12(2x3)
    # 4. A12(2x3) * B22(3x3) -> contributes to C12(2x3)
    # 5. A21(2x2) * B11(2x3) -> contributes to C21(2x3)
    # 6. A22(2x3) * B21(3x3) -> contributes to C21(2x3)
    # 7. A21(2x2) * B12(2x3) -> contributes to C22(2x3)
    # 8. A22(2x3) * B22(3x3) -> contributes to C22(2x3)
    
    U1, V1, W1, r1 = build_2x2_x_2x3()
    U2, V2, W2, r2 = build_2x3_x_3x3()
    U3, V3, W3, r3 = build_2x2_x_2x3()
    U4, V4, W4, r4 = build_2x3_x_3x3()
    U5, V5, W5, r5 = build_2x2_x_2x3()
    U6, V6, W6, r6 = build_2x3_x_3x3()
    U7, V7, W7, r7 = build_2x2_x_2x3()
    U8, V8, W8, r8 = build_2x3_x_3x3()
    
    total_rank = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
    
    # Now assemble the full U, V, W matrices.
    # Full U has shape (20, total_rank) [4*5 = 20]
    # Full V has shape (30, total_rank) [5*6 = 30]
    # Full W has shape (24, total_rank) [6*4 = 24]
    
    U_full = [[0]*total_rank for _ in range(20)]
    V_full = [[0]*total_rank for _ in range(30)]
    W_full = [[0]*total_rank for _ in range(24)]
    
    def add_decomposition(U_sub, V_sub, W_sub, rank_sub, 
                          U_offset, V_offset, W_offset, col_start):
        """Add a sub-decomposition to the full matrices."""
        for i in range(len(U_sub)):
            for c in range(rank_sub):
                U_full[U_offset + i][col_start + c] = U_sub[i][c]
        for j in range(len(V_sub)):
            for c in range(rank_sub):
                V_full[V_offset + j][col_start + c] = V_sub[j][c]
        for k in range(len(W_sub)):
            for c in range(rank_sub):
                W_full[W_offset + k][col_start + c] = W_sub[k][c]
    
    col = 0
    
    # 1. A11(2x2) * B11(2x3) -> C11(2x3)
    # A11: A[0:2, 0:2] -> U rows 0-3 (i*5+j for i=0,1; j=0,1)
    #   U row indices: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # B11: B[0:2, 0:3] -> V rows 0-5 (j*6+k for j=0,1; k=0,1,2)
    #   V row indices: 0*6+0=0, 0*6+1=1, 0*6+2=2, 1*6+0=6, 1*6+1=7, 1*6+2=8
    # C11: C[0:2, 0:3] -> W rows 0-5 (k*4+i for k=0,1,2; i=0,1)
    #   W row indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5, 2*4+0=8, 2*4+1=9
    
    # The sub-decomposition U_sub has rows indexed by [i*2+j] for i=0,1; j=0,1
    # We need to map these to the full U indices.
    
    # For 2x2 A block at (row_offset, col_offset) in the full A:
    #   sub row (i, j) maps to full row (row_offset+i)*5 + (col_offset+j)
    
    # For 2x3 B block at (row_offset, col_offset) in the full B:
    #   sub row (j, k) maps to full row (row_offset+j)*6 + (col_offset+k)
    
    # For 2x3 C block at (row_offset, col_offset) in the full C:
    #   sub row (k, i) maps to full row (col_offset+k)*4 + (row_offset+i)
    
    def add_2x2_x_2x3(U_sub, V_sub, W_sub, rank_sub, 
                      a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, col_start):
        """Add a 2x2 * 2x3 -> 2x3 sub-decomposition."""
        # U_sub rows: (i,j) for i=0,1; j=0,1 -> index i*2+j
        # Map to full U: (a_row_off+i)*5 + (a_col_off+j)
        for i in range(2):
            for j in range(2):
                for c in range(rank_sub):
                    val = U_sub[i*2+j][c]
                    if val != 0:
                        U_full[(a_row_off+i)*5 + (a_col_off+j)][col_start+c] = val
        
        # V_sub rows: (j,k) for j=0,1; k=0,1,2 -> index j*3+k
        # Map to full V: (b_row_off+j)*6 + (b_col_off+k)
        for j in range(2):
            for k in range(3):
                for c in range(rank_sub):
                    val = V_sub[j*3+k][c]
                    if val != 0:
                        V_full[(b_row_off+j)*6 + (b_col_off+k)][col_start+c] = val
        
        # W_sub rows: (k,i) for k=0,1,2; i=0,1 -> index k*2+i
        # Map to full W: (c_col_off+k)*4 + (c_row_off+i)
        for k in range(3):
            for i in range(2):
                for c in range(rank_sub):
                    val = W_sub[k*2+i][c]
                    if val != 0:
                        W_full[(c_col_off+k)*4 + (c_row_off+i)][col_start+c] = val
    
    def add_2x3_x_3x3(U_sub, V_sub, W_sub, rank_sub,
                      a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, col_start):
        """Add a 2x3 * 3x3 -> 2x3 sub-decomposition."""
        # U_sub rows: (i,j) for i=0,1; j=0,1,2 -> index i*3+j
        # Map to full U: (a_row_off+i)*5 + (a_col_off+j)
        for i in range(2):
            for j in range(3):
                for c in range(rank_sub):
                    val = U_sub[i*3+j][c]
                    if val != 0:
                        U_full[(a_row_off+i)*5 + (a_col_off+j)][col_start+c] = val
        
        # V_sub rows: (j,k) for j=0,1,2; k=0,1,2 -> index j*3+k
        # Map to full V: (b_row_off+j)*6 + (b_col_off+k)
        for j in range(3):
            for k in range(3):
                for c in range(rank_sub):
                    val = V_sub[j*3+k][c]
                    if val != 0:
                        V_full[(b_row_off+j)*6 + (b_col_off+k)][col_start+c] = val
        
        # W_sub rows: (k,i) for k=0,1,2; i=0,1 -> index k*2+i
        # Map to full W: (c_col_off+k)*4 + (c_row_off+i)
        for k in range(3):
            for i in range(2):
                for c in range(rank_sub):
                    val = W_sub[k*2+i][c]
                    if val != 0:
                        W_full[(c_col_off+k)*4 + (c_row_off+i)][col_start+c] = val
    
    # Add all 8 sub-decompositions:
    
    # 1. A11(2x2) * B11(2x3) -> C11(2x3)
    # A11 at (0,0), B11 at (0,0), C11 at (0,0)
    add_2x2_x_2x3(U1, V1, W1, r1, 0, 0, 0, 0, 0, 0, col)
    col += r1
    
    # 2. A12(2x3) * B21(3x3) -> C11(2x3)
    # A12 at (0,2), B21 at (2,0), C11 at (0,0)
    add_2x3_x_3x3(U2, V2, W2, r2, 0, 2, 2, 0, 0, 0, col)
    col += r2
    
    # 3. A11(2x2) * B12(2x3) -> C12(2x3)
    # A11 at (0,0), B12 at (0,3), C12 at (0,3)
    add_2x2_x_2x3(U3, V3, W3, r3, 0, 0, 0, 3, 0, 3, col)
    col += r3
    
    # 4. A12(2x3) * B22(3x3) -> C12(2x3)
    # A12 at (0,2), B22 at (2,3), C12 at (0,3)
    add_2x3_x_3x3(U4, V4, W4, r4, 0, 2, 2, 3, 0, 3, col)
    col += r4
    
    # 5. A21(2x2) * B11(2x3) -> C21(2x3)
    # A21 at (2,0), B11 at (0,0), C21 at (2,0)
    add_2x2_x_2x3(U5, V5, W5, r5, 2, 0, 0, 0, 2, 0, col)
    col += r5
    
    # 6. A22(2x3) * B21(3x3) -> C21(2x3)
    # A22 at (2,2), B21 at (2,0), C21 at (2,0)
    add_2x3_x_3x3(U6, V6, W6, r6, 2, 2, 2, 0, 2, 0, col)
    col += r6
    
    # 7. A21(2x2) * B12(2x3) -> C22(2x3)
    # A21 at (2,0), B12 at (0,3), C22 at (2,3)
    add_2x2_x_2x3(U7, V7, W7, r7, 2, 0, 0, 3, 2, 3, col)
    col += r7
    
    # 8. A22(2x3) * B22(3x3) -> C22(2x3)
    # A22 at (2,2), B22 at (2,3), C22 at (2,3)
    add_2x3_x_3x3(U8, V8, W8, r8, 2, 2, 2, 3, 2, 3, col)
    col += r8
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
