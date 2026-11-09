import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    # We'll try to find a rank-minimal decomposition.
    # Start with schoolbook (rank 60) as fallback.
    
    # Strategy: Use block decomposition.
    # Split middle dimension m=5 into m1=2, m2=3.
    # For the (n x m1) * (m1 x p) block: use Strassen if possible, else schoolbook.
    # For the (n x m2) * (m2 x p) block: use schoolbook.
    
    # Actually, let's think more carefully.
    # C = A * B where A is n x m, B is m x p.
    # Split A into A1 (n x m1) and A2 (n x m2)
    # Split B into B1 (m1 x p) and B2 (m2 x p)
    # C = A1 * B1 + A2 * B2
    
    # For A1 * B1: n=2, m1=2, p=6
    #   We can further split p=6 into p1=2, p2=2, p3=2
    #   So A1 * B1 = A1 * [B1_1, B1_2, B1_3] = [A1*B1_1, A1*B1_2, A1*B1_3]
    #   Each A1*B1_j is 2x2 * 2x2, which can use Strassen (rank 7 instead of 8).
    #   Total for A1*B1: 3 * 7 = 21
    
    # For A2 * B2: n=2, m2=3, p=6
    #   Split p=6 into p1=2, p2=2, p3=2
    #   A2 * B2 = [A2*B2_1, A2*B2_2, A2*B2_3]
    #   Each A2*B2_j is 2x3 * 3x2
    #   Schoolbook rank for 2x3*3x2 is 2*3*2 = 12
    #   Total for A2*B2: 3 * 12 = 36
    
    # Total rank: 21 + 36 = 57
    
    # Let's implement this.
    
    def strassen_2x2x2():
        """
        Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
        U: 4x7, V: 4x7, W: 4x7
        Indices: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c]
        """
        # Strassen formulas:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        # C00 = M1 - M5 + M6 - M7
        # C01 = M3 + M4 + M7
        # C10 = M2 + M5 - M4
        # C11 = M1 - M2 + M3 + M6
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1: (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2: (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = -1  # C11
        
        # M3: A00(B10-B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[3, 2] = -1  # B11
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4: A11(B01-B00)
        U[3, 3] = 1  # A11
        V[1, 3] = 1  # B01
        V[0, 3] = -1  # B00
        W[1, 3] = 1  # C01
        W[2, 3] = -1  # C10
        
        # M5: (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1  # C00
        W[2, 4] = 1  # C10
        
        # M6: (A10-A00)(B00+B10)
        U[2, 5] = 1  # A10
        U[0, 5] = -1  # A00
        V[0, 5] = 1  # B00
        V[2, 5] = 1  # B10
        W[0, 5] = 1  # C00
        W[3, 5] = 1  # C11
        
        # M7: (A01-A11)(B01+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1  # A11
        V[1, 6] = 1  # B01
        V[3, 6] = 1  # B11
        W[0, 6] = -1  # C00
        W[1, 6] = 1  # C01
        
        return U, V, W
    
    def schoolbook_nm_p(n, m, p):
        """
        Return schoolbook decomposition for n x m * m x p -> n x p.
        U: (n*m) x (n*m*p), V: (m*p) x (n*m*p), W: (p*n) x (n*m*p)
        """
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Build the block decomposition
    m1 = 2
    m2 = 3
    assert m1 + m2 == m
    
    # A1 is 2x2, B1 is 2x6
    # Split B1 into three 2x2 blocks: B1_0, B1_1, B1_2
    # For each, use Strassen
    
    # A2 is 2x3, B2 is 3x6
    # Split B2 into three 3x2 blocks: B2_0, B2_1, B2_2
    # For each, use schoolbook (2x3 * 3x2)
    
    # Collect all U, V, W blocks
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # For A1 * B1: three Strassen decompositions
    # The A part is shared across all three Strassen decomps
    # But in the tensor decomposition, each Strassen term is independent
    
    # Actually, we need to be careful. The full decomposition is:
    # C = A1*B1 + A2*B2
    # = [A1*B1_0, A1*B1_1, A1*B1_2] + [A2*B2_0, A2*B2_1, A2*B2_2]
    
    # Each A1*B1_j uses Strassen with its own U, V, W
    # Each A2*B2_j uses schoolbook with its own U, V, W
    
    # For A1*B1_j (j=0,1,2):
    #   A1 is 2x2, B1_j is 2x2
    #   The U matrix maps A1 indices to the Strassen factors
    #   The V matrix maps B1_j indices to the Strassen factors
    #   The W matrix maps the output (2x2) to C columns j*2 to j*2+1
    
    # Let's build this properly.
    
    # For each of the 3 Strassen blocks:
    for j in range(3):
        U_s, V_s, W_s = strassen_2x2x2()
        # U_s is 4x7, corresponds to A1 (2x2)
        # V_s is 4x7, corresponds to B1_j (2x2)
        # W_s is 4x7, corresponds to output C[:, j*2:(j+1)*2]
        
        # In the full U matrix (10 x total_rank), the rows for A1 are 0,1,2,3
        # In the full V matrix (30 x total_rank), the rows for B1_j are j*4 to j*4+3
        #   Wait, B1 is 2x6, so B1_j is columns j*2 to j*2+1 of B1
        #   B1_j indices: row 0,1 (from m1=2), col j*2, j*2+1
        #   In V, the index is j'*p + k where j' is row in B1_j (0 or 1), k is col (0 or 1)
        #   But in the full V, the index is j_full*p + k_full
        #   j_full = j' (since B1 is the first part of B, rows 0,1)
        #   k_full = j*2 + k (since B1_j is columns j*2, j*2+1 of B)
        
        # Let me re-index.
        # Full A is 2x5, rows 0-1, cols 0-4
        # A1 is cols 0-1, A2 is cols 2-4
        # Full B is 5x6, rows 0-4, cols 0-5
        # B1 is rows 0-1, B2 is rows 2-4
        # B1_j is B1 with cols j*2, j*2+1
        
        # U index: i*m + j_a where i is row of A (0,1), j_a is col of A (0-4)
        # For A1: j_a = 0,1
        # V index: j_b*p + k where j_b is row of B (0-4), k is col of B (0-5)
        # For B1_j: j_b = 0,1, k = j*2, j*2+1
        # W index: k_c*n + i where k_c is col of C (0-5), i is row of C (0,1)
        # For output of A1*B1_j: k_c = j*2, j*2+1, i = 0,1
        
        # So for Strassen block j:
        # U part: 4 rows (i=0,1; j_a=0,1) -> indices 0,1,2,3
        # V part: 4 rows (j_b=0,1; k=j*2, j*2+1) -> indices j_b*6 + k
        #   j_b=0, k=j*2 -> 0*6 + j*2 = j*2
        #   j_b=0, k=j*2+1 -> j*2+1
        #   j_b=1, k=j*2 -> 1*6 + j*2 = 6+j*2
        #   j_b=1, k=j*2+1 -> 6+j*2+1
        # W part: 4 rows (k_c=j*2, j*2+1; i=0,1) -> indices k_c*2 + i
        #   k_c=j*2, i=0 -> j*4
        #   k_c=j*2, i=1 -> j*4+1
        #   k_c=j*2+1, i=0 -> j*4+2
        #   k_c=j*2+1, i=1 -> j*4+3
        
        # We need to place U_s (4x7) into the full U at rows 0-3
        # Place V_s (4x7) into the full V at rows [j*2, j*2+1, 6+j*2, 6+j*2+1]
        # Place W_s (4x7) into the full W at rows [j*4, j*4+1, j*4+2, j*4+3]
        
        U_blocks.append((list(range(4)), U_s))  # rows 0-3
        v_rows = [j*2, j*2+1, 6+j*2, 6+j*2+1]
        V_blocks.append((v_rows, V_s))
        w_rows = [j*4, j*4+1, j*4+2, j*4+3]
        W_blocks.append((w_rows, W_s))
    
    # For A2 * B2: three schoolbook decompositions
    # A2 is 2x3, B2_j is 3x2
    for j in range(3):
        U_sb, V_sb, W_sb = schoolbook_nm_p(2, 3, 2)
        # U_sb is 6x12, corresponds to A2 (2x3)
        # V_sb is 6x12, corresponds to B2_j (3x2)
        # W_sb is 4x12, corresponds to output C[:, j*2:(j+1)*2]
        
        # U index for A2: i*m + j_a where i=0,1, j_a=2,3,4
        #   i=0, j_a=2 -> 0*5+2 = 2
        #   i=0, j_a=3 -> 3
        #   i=0, j_a=4 -> 4
        #   i=1, j_a=2 -> 5
        #   i=1, j_a=3 -> 6
        #   i=1, j_a=4 -> 7
        u_rows_a2 = [2, 3, 4, 5, 6, 7]
        
        # V index for B2_j: j_b*6 + k where j_b=2,3,4, k=j*2, j*2+1
        #   j_b=2, k=j*2 -> 12+j*2
        #   j_b=2, k=j*2+1 -> 12+j*2+1
        #   j_b=3, k=j*2 -> 18+j*2
        #   j_b=3, k=j*2+1 -> 18+j*2+1
        #   j_b=4, k=j*2 -> 24+j*2
        #   j_b=4, k=j*2+1 -> 24+j*2+1
        v_rows_b2 = [12+j*2, 12+j*2+1, 18+j*2, 18+j*2+1, 24+j*2, 24+j*2+1]
        
        # W index for output: k_c*2 + i where k_c=j*2, j*2+1, i=0,1
        #   k_c=j*2, i=0 -> j*4
        #   k_c=j*2, i=1 -> j*4+1
        #   k_c=j*2+1, i=0 -> j*4+2
        #   k_c=j*2+1, i=1 -> j*4+3
        w_rows_out = [j*4, j*4+1, j*4+2, j*4+3]
        
        U_blocks.append((u_rows_a2, U_sb))
        V_blocks.append((v_rows_b2, V_sb))
        W_blocks.append((w_rows_out, W_sb))
    
    # Now assemble the full U, V, W matrices
    total_rank = sum(block.shape[1] for _, block in U_blocks)
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (u_rows, u_block), (v_rows, v_block), (w_rows, w_block) in zip(U_blocks, V_blocks, W_blocks):
        r = u_block.shape[1]
        U_full[u_rows, col_offset:col_offset+r] = u_block
        V_full[v_rows, col_offset:col_offset+r] = v_block
        W_full[w_rows, col_offset:col_offset+r] = w_block
        col_offset += r
    
    # Convert to lists of lists with integer entries
    def to_int_list(mat):
        return [[int(x) for x in row] for row in mat.tolist()]
    
    return {"U": to_int_list(U_full), "V": to_int_list(V_full), "W": to_int_list(W_full)}

