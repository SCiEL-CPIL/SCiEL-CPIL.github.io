"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    
    def _strassen_2x2():
        """
        Returns U, V, W for 2x2 x 2x2 matrix multiplication with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def _schoolbook_block(n_u, n_v, n_w, offset_u, offset_v, offset_w):
        """
        Returns U, V, W for a schoolbook block of size n_u x n_v x n_w.
        U: (n_u, rank), V: (n_v, rank), W: (n_w, rank)
        """
        rank = n_u * n_v * n_w
        U = np.zeros((n_u, rank))
        V = np.zeros((n_v, rank))
        W = np.zeros((n_w, rank))
        
        col = 0
        for i in range(n_u):
            for j in range(n_v):
                for k in range(n_w):
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
        return U, V, W

    def _combine_parts(parts, n, m, p):
        """
        Combine a list of (U, V, W, u_offset, v_offset, w_offset) into full matrices.
        """
        total_rank = sum(part[0].shape[1] for part in parts)
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col = 0
        for U_part, V_part, W_part, u_off, v_off, w_off in parts:
            r = U_part.shape[1]
            U_full[u_off[0]:u_off[1], col:col+r] = U_part
            V_full[v_off[0]:v_off[1], col:col+r] = V_part
            W_full[w_off[0]:w_off[1], col:col+r] = W_part
            col += r
            
        return U_full, V_full, W_full

    # Strategy: Rank 31
    # Part 1: 3x3 x 3x3 (Rank 22)
    # Part 2: 3x3 x 3x1 (Rank 9)
    
    parts = []
    
    # --- Part 1: 3x3 x 3x3 ---
    # Sub-part 1a: 2x2 x 2x2 (Top-Left) using Strassen (Rank 7)
    S_U, S_V, S_W = _strassen_2x2()
    # U indices: A[0:2, 0:2] -> rows 0,1; cols 0,1. U index = i*3+j.
    # We need to map S_U (4x7) to U_full (9x...).
    # S_U row i*2+j corresponds to A[i,j].
    # U_full row i*3+j.
    U_1a = np.zeros((9, 7))
    V_1a = np.zeros((12, 7)) # m*p = 3*4
    W_1a = np.zeros((12, 7)) # p*n = 4*3
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * 3 + j
                v_idx = i * 4 + j
                w_idx = j * 3 + i # W index is k*n + i, where k is col of B (j here), i is row of A
                U_1a[u_idx, r] = S_U[i * 2 + j, r]
                V_1a[v_idx, r] = S_V[i * 2 + j, r]
                W_1a[w_idx, r] = S_W[i * 2 + j, r]
                
    parts.append((U_1a, V_1a, W_1a, (0, 9), (0, 12), (0, 12))) # Offsets not strictly needed if we place directly, but let's use a helper or just fill full matrices directly.
    
    # Let's just build the full matrices directly to avoid offset complexity errors.
    
    U_total = np.zeros((n * m, 31))
    V_total = np.zeros((m * p, 31))
    W_total = np.zeros((p * n, 31))
    
    col = 0
    
    # 1. Strassen 2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    S_U, S_V, S_W = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + j
                v_idx = i * p + j
                w_idx = j * n + i
                U_total[u_idx, col + r] = S_U[i * 2 + j, r]
                V_total[v_idx, col + r] = S_V[i * 2 + j, r]
                W_total[w_idx, col + r] = S_W[i * 2 + j, r]
    col += 7
    
    # 2. Schoolbook for A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    # A rows 0,1 col 2. B row 2 cols 0,1. C rows 0,1 cols 0,1.
    # Rank 4.
    for i in range(2):
        for k in range(2):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U_total[u_idx, col] = 1
            V_total[v_idx, col] = 1
            W_total[w_idx, col] = 1
            col += 1
    col += 0 # col is now 11
    
    # 3. Schoolbook for A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    # A row 2 cols 0,1. B rows 0,1 cols 0,1. C row 2 cols 0,1.
    # Rank 4.
    for j in range(2):
        for k in range(2):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U_total[u_idx, col] = 1
            V_total[v_idx, col] = 1
            W_total[w_idx, col] = 1
            col += 1
    col += 0 # col is now 15

    # 4. Schoolbook for A[2,2] * B[2,0:2] -> C[2,0:2]
    # A row 2 col 2. B row 2 cols 0,1. C row 2 cols 0,1.
    # Rank 2.
    for k in range(2):
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
    col += 0 # col is now 17

    # 5. Strassen 2x2 for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B cols 2,3.
    S_U, S_V, S_W = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + j
                v_idx = i * p + (2 + j) # B col offset 2
                w_idx = (2 + j) * n + i # C col offset 2
                U_total[u_idx, col + r] = S_U[i * 2 + j, r]
                V_total[v_idx, col + r] = S_V[i * 2 + j, r]
                W_total[w_idx, col + r] = S_W[i * 2 + j, r]
    col += 7 # col is now 24

    # 6. Schoolbook for A[0:2,2] * B[2,2:4] -> C[0:2,2:4]
    # A rows 0,1 col 2. B row 2 cols 2,3. C rows 0,1 cols 2,3.
    # Rank 4.
    for i in range(2):
        for k in range(2):
            u_idx = i * m + 2
            v_idx = 2 * p + (2 + k)
            w_idx = (2 + k) * n + i
            U_total[u_idx, col] = 1
            V_total[v_idx, col] = 1
            W_total[w_idx, col] = 1
            col += 1
    col += 0 # col is now 28

    # 7. Schoolbook for A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    # A row 2 cols 0,1. B rows 0,1 cols 2,3. C row 2 cols 2,3.
    # Rank 4.
    for j in range(2):
        for k in range(2):
            u_idx = 2 * m + j
            v_idx = j * p + (2 + k)
            w_idx = (2 + k) * n + 2
            U_total[u_idx, col] = 1
            V_total[v_idx, col] = 1
            W_total[w_idx, col] = 1
            col += 1
    col += 0 # col is now 32 -- WAIT. I planned Rank 31.
    
    # Let's re-count.
    # 1. Strassen 2x2 (7)
    # 2. A[0:2,2]B[2,0:2] (4) -> Total 11
    # 3. A[2,0:2]B[0:2,0:2] (4) -> Total 15
    # 4. A[2,2]B[2,0:2] (2) -> Total 17
    # 5. Strassen 2x2 (7) -> Total 24
    # 6. A[0:2,2]B[2,2:4] (4) -> Total 28
    # 7. A[2,0:2]B[0:2,2:4] (4) -> Total 32
    # 8. A[2,2]B[2,2:4] (2) -> Total 34
    
    # My previous "Rank 31" calculation was:
    # 3x3x3 (22) + 3x3x1 (9) = 31.
    # The block decomposition above is:
    # C[0:2,0:2]: 7 + 4 = 11
    # C[2,0:2]: 4 + 2 = 6
    # C[0:2,2:4]: 7 + 4 = 11
    # C[2,2:4]: 4 + 2 = 6
    # Total 34.
    
    # Why did I think 31?
    # 3x3x3 part:
    # C11 (2x2): 7
    # C12 (2x1): 4+2=6
    # C21 (1x2): 4+2=6
    # C22 (1x1): 2+1=3
    # Total 22.
    
    # 3x3x1 part (B col 3):
    # C[0,3]: A[0,0]B[0,3] + A[0,1]B[1,3] + A[0,2]B[2,3] (3 terms)
    # C[1,3]: A[1,0]B[0,3] + A[1,1]B[1,3] + A[1,2]B[2,3] (3 terms)
    # C[2,3]: A[2,0]B[0,3] + A[2,1]B[1,3] + A[2,2]B[2,3] (3 terms)
    # Total 9 terms.
    
    # Total 22 + 9 = 31.
    
    # The block decomposition above covers B[0:3] (cols 0,1,2).
    # It does NOT cover B col 3.
    # So I need to add the 9 terms for B col 3.
    # Current col index is 32 (from steps 1-7 above, which covered B cols 0-2).
    # Wait, step 7 covered B[0:2, 2:4] which is cols 2,3.
    # Step 6 covered B[2, 2:4] which is cols 2,3.
    # Step 5 covered B[0:2, 2:4] which is cols 2,3.
    # So B col 3 IS covered in steps 5,6,7.
    
    # Let's re-verify the "3x3x3 + 3x3x1" logic vs "Block" logic.
    # Block logic:
    # B is split into B_left (cols 0,1) and B_right (cols 2,3).
    # A is split into A_top (rows 0,1) and A_bot (row 2).
    
    # C_top_left (2x2): A_top B_left.
    #   A_top = [A_tl(2x2) A_tr(2x1)]
    #   B_left = [B_tl(2x2); B_bl(1x2)]
    #   C_tl = A_tl B_tl + A_tr B_bl.
    #   Rank(A_tl B_tl) = 7 (Strassen).
    #   Rank(A_tr B_bl) = 2*1*2 = 4 (Schoolbook).
    #   Total 11.
    
    # C_bot_left (1x2): A_bot B_left.
    #   A_bot = [A_bl(1x2) A_br(1x1)]
    #   B_left = [B_tl(2x2); B_bl(1x2)]
    #   C_bl = A_bl B_tl + A_br B_bl.
    #   Rank(A_bl B_tl) = 1*2*2 = 4.
    #   Rank(A_br B_bl) = 1*1*2 = 2.
    #   Total 6.
    
    # C_top_right (2x2): A_top B_right.
    #   B_right = [B_tr(2x2); B_br(1x2)]
    #   C_tr = A_tl B_tr + A_tr B_br.
    #   Rank(A_tl B_tr) = 7.
    #   Rank(A_tr B_br) = 4.
    #   Total 11.
    
    # C_bot_right (1x2): A_bot B_right.
    #   C_br = A_bl B_tr + A_br B_br.
    #   Rank(A_bl B_tr) = 4.
    #   Rank(A_br B_br) = 2.
    #   Total 6.
    
    # Total 11+6+11+6 = 34.
    
    # The "3x3x3 + 3x3x1" logic:
    # B = [B_3x3, b_3x1].
    # C = A B_3x3 + A b_3x1.
    # Rank(A B_3x3) = 22.
    # Rank(A b_3x1) = 9.
    # Total 31.
    
    # Why the difference?
    # In 3x3x3, B_3x3 uses cols 0,1,2.
    # In 3x3x1, b_3x1 uses col 3.
    # In Block logic, B_right uses cols 2,3.
    # So Block logic mixes col 2 and 3 in the Strassen/Schoolbook terms for the right half.
    # The 3x3x3 logic isolates col 3 completely.
    # The 3x3x3 logic is more efficient because it uses Strassen for the main 3x3 block.
    # The Block logic uses Strassen for 2x2 blocks.
    
    # So I should implement the 31 rank solution.
    
    # Reset col
    col = 0
    
    # 1. 3x3x3 Part (Rank 22)
    # C11 (2x2) from A[0:2,0:2] B[0:2,0:2] -> Strassen 7
    S_U, S_V, S_W = _strassen_2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + j
                v_idx = i * p + j
                w_idx = j * n + i
                U_total[u_idx, col + r] = S_U[i * 2 + j, r]
                V_total[v_idx, col + r] = S_V[i * 2 + j, r]
                W_total[w_idx, col + r] = S_W[i * 2 + j, r]
    col += 7
    
    # C12 (2x1) from A[0:2,0:2] B[0:2,2] + A[0:2,2] B[2,2]
    # Part A: A[0:2,0:2] B[0:2,2] (Rank 4)
    for i in range(2):
        for j in range(2):
            u_idx = i * m + j
            v_idx = j * p + 2
            w_idx = 2 * n + i
            U_total[u_idx, col] = 1
            V_total[v_idx, col] = 1
            W_total[w_idx, col] = 1
            col += 1
    # Part B: A[0:2,2] B[2,2] (Rank 2)
    for i in range(2):
        u_idx = i * m + 2
        v_idx = 2 * p + 2
        w_idx = 2 * n + i
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
        
    # C21 (1x2) from A[2,0:2] B[0:2,0:2] + A[2,2] B[2,0:2]
    # Part A: A[2,0:2] B[0:2,0:2] (Rank 4)
    for j in range(2):
        for k in range(2):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U_total[u_idx, col] = 1
            V_total[v_idx, col] = 1
            W_total[w_idx, col] = 1
            col += 1
    # Part B: A[2,2] B[2,0:2] (Rank 2)
    for k in range(2):
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
        
    # C22 (1x1) from A[2,0:2] B[0:2,2] + A[2,2] B[2,2]
    # Part A: A[2,0:2] B[0:2,2] (Rank 2)
    for j in range(2):
        u_idx = 2 * m + j
        v_idx = j * p + 2
        w_idx = 2 * n + 2
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
    # Part B: A[2,2] B[2,2] (Rank 1)
    u_idx = 2 * m + 2
    v_idx = 2 * p + 2
    w_idx = 2 * n + 2
    U_total[u_idx, col] = 1
    V_total[v_idx, col] = 1
    W_total[w_idx, col] = 1
    col += 1
    
    # Total so far: 7 + 4 + 2 + 4 + 2 + 2 + 1 = 22.
    # col is 22.
    
    # 2. 3x3x1 Part (Rank 9) for B col 3
    # C[0,3]: A[0,0]B[0,3] + A[0,1]B[1,3] + A[0,2]B[2,3]
    for j in range(3):
        u_idx = 0 * m + j
        v_idx = j * p + 3
        w_idx = 3 * n + 0
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
        
    # C[1,3]: A[1,0]B[0,3] + A[1,1]B[1,3] + A[1,2]B[2,3]
    for j in range(3):
        u_idx = 1 * m + j
        v_idx = j * p + 3
        w_idx = 3 * n + 1
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
        
    # C[2,3]: A[2,0]B[0,3] + A[2,1]B[1,3] + A[2,2]B[2,3]
    for j in range(3):
        u_idx = 2 * m + j
        v_idx = j * p + 3
        w_idx = 3 * n + 2
        U_total[u_idx, col] = 1
        V_total[v_idx, col] = 1
        W_total[w_idx, col] = 1
        col += 1
        
    # Total Rank 31.
    
    def to_half_integer_array(arr):
        rounded = np.round(arr * 2) / 2
        return [[float(x) for x in row] for row in rounded]
    
    return {
        "U": to_half_integer_array(U_total),
        "V": to_half_integer_array(V_total),
        "W": to_half_integer_array(W_total)
    }
# EVOLVE-BLOCK-END
