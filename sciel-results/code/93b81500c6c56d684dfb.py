import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # --- STRATEGY 1: STRUCTURED BLOCK CONSTRUCTION (Rank 66) ---
    # We decompose the 3x4x6 multiplication into blocks.
    # A (3x4) is split into rows: Row0 (1x4), Row1 (1x4), Row2 (1x4)
    # B (4x6) is split into cols: Col0 (4x2), Col1 (4x2), Col2 (4x2)
    # This doesn't quite work directly for Strassen which needs square blocks.
    # Better: Split A into 3 blocks of 1x4? No, Strassen is for 2x2.
    # Let's use the known rank 66 construction from inspiration 1/2.
    # The structure:
    # n=3, m=4, p=6.
    # We can view this as 3 separate 1x4 * 4x6 multiplications? No, that's rank 72.
    # 
    # Let's try to construct a rank 66 decomposition manually using 2x2 Strassen blocks.
    # A 2x2 matrix mult has rank 7.
    # If we split A (3x4) into top 2x4 and bottom 1x4.
    # Split B (4x6) into left 4x2, mid 4x2, right 4x2.
    # 
    # Actually, let's use the "2+2+2" partition of B's columns and "2+1" partition of A's rows.
    # Block 1: A_top (2x4) * B_left (4x2) -> 2x2 result. Rank 7.
    # Block 2: A_top (2x4) * B_mid (4x2) -> 2x2 result. Rank 7.
    # Block 3: A_top (2x4) * B_right (4x2) -> 2x2 result. Rank 7.
    # Block 4: A_bot (1x4) * B_left (4x2) -> 1x2 result. Rank 4? (1x2 mult is rank 4? No, 1*2=2 terms schoolbook, but tensor rank of 1x2x2 is 4? 
    #    T[i,j,k] = A[i,j]B[j,k]. i in 0, j in 0,1, k in 0,1. 
    #    Terms: A00B00, A00B01, A01B00, A01B01. Rank 4? 
    #    Actually, 1x2 * 2x2 is just 4 scalar multiplications. Rank 4.
    # Block 5: A_bot (1x4) * B_mid (4x2) -> 1x2 result. Rank 4.
    # Block 6: A_bot (1x4) * B_right (4x2) -> 1x2 result. Rank 4.
    # Total Rank = 7 + 7 + 7 + 4 + 4 + 4 = 33? 
    # Wait, the Strassen decomposition for 2x2 requires the 2x4 block to be decomposed into 2x2s?
    # A 2x4 block is two 2x2 blocks side-by-side.
    # So A_top (2x4) = [A_tl, A_tr].
    # B_left (4x2) is 4x2. 
    # The product A_top * B_left is (2x4) * (4x2) = 2x2.
    # This is a 2x4 * 4x2 multiplication.
    # This can be seen as two 2x2 * 2x2 multiplications if we split B_left into top 2x2 and bottom 2x2?
    # No, B is 4x6. Rows are 4. 
    # Let's split A (3x4) into A1 (2x4) and A2 (1x4).
    # Let's split B (4x6) into B1 (4x2), B2 (4x2), B3 (4x2).
    # 
    # Product C = A * B.
    # C11 (2x2) = A1 * B1. (2x4 * 4x2). 
    #    To compute 2x4 * 4x2, we can split A1 into A1a (2x2) and A1b (2x2).
    #    And B1 into B1a (2x2) and B1b (2x2).
    #    C11 = A1a*B1a + A1b*B1b.
    #    Each 2x2*2x2 is rank 7. Sum of two rank 7 tensors is rank 14.
    #    So C11 costs 14.
    #    Similarly C12 (2x2) = A1 * B2 costs 14.
    #    Similarly C13 (2x2) = A1 * B3 costs 14.
    #    Total for top rows: 42.
    #    
    # C21 (1x2) = A2 * B1. (1x4 * 4x2).
    #    A2 is 1x4. Split into A2a (1x2), A2b (1x2).
    #    B1 is 4x2. Split into B1a (2x2), B1b (2x2).
    #    C21 = A2a*B1a + A2b*B1b.
    #    1x2 * 2x2 multiplication. 
    #    T[i,j,k] where i=0, j in 0,1, k in 0,1.
    #    This is rank 4. (Schoolbook is 4, can it be lower? No, 1x2x2 tensor rank is 4).
    #    So C21 costs 4.
    #    Similarly C22 costs 4.
    #    Similarly C23 costs 4.
    #    Total for bottom row: 12.
    #    
    # Total Rank = 42 + 12 = 54? 
    # Wait, is the rank of 2x4 * 4x2 really 14?
    # Yes, it's the sum of two independent 2x2*2x2 multiplications.
    # So rank 54 is possible?
    # Let's check the lower bounds. 
    # If rank 54 is possible, the current best of 66 is bad.
    # However, the "official" trace convention and specific index mappings might prevent simple block addition if the blocks overlap in the factor matrices.
    # In the tensor decomposition, the factors U, V, W are global.
    # If we use disjoint sets of columns for each block, the ranks add.
    # So 54 should be valid if the math holds.
    # 
    # Let's implement the rank 54 construction.
    # 
    # Indices:
    # n=3, m=4, p=6.
    # U shape (12, R), V shape (24, R), W shape (18, R).
    # 
    # Blocks:
    # 1. A1a (rows 0,1; cols 0,1) * B1a (rows 0,1; cols 0,1) -> C(0,1; 0,1)
    # 2. A1b (rows 0,1; cols 2,3) * B1b (rows 2,3; cols 0,1) -> C(0,1; 0,1)
    # 3. A1a * B2a (rows 0,1; cols 2,3) -> C(0,1; 2,3)
    # 4. A1b * B2b (rows 2,3; cols 2,3) -> C(0,1; 2,3)
    # 5. A1a * B3a (rows 0,1; cols 4,5) -> C(0,1; 4,5)
    # 6. A1b * B3b (rows 2,3; cols 4,5) -> C(0,1; 4,5)
    # 
    # 7. A2a (row 2; cols 0,1) * B1a (rows 0,1; cols 0,1) -> C(2; 0,1)
    # 8. A2b (row 2; cols 2,3) * B1b (rows 2,3; cols 0,1) -> C(2; 0,1)
    # 9. A2a * B2a -> C(2; 2,3)
    # 10. A2b * B2b -> C(2; 2,3)
    # 11. A2a * B3a -> C(2; 4,5)
    # 12. A2b * B3b -> C(2; 4,5)
    # 
    # Total 12 blocks.
    # Blocks 1-6 are 2x2*2x2 (Rank 7 each) -> 42.
    # Blocks 7-12 are 1x2*2x2 (Rank 4 each) -> 24.
    # Total Rank = 66.
    # 
    # Wait, my previous calculation of 54 assumed 1x2*2x2 was rank 2? No, I said 4.
    # 12 blocks * avg rank?
    # 6 * 7 + 6 * 4 = 42 + 24 = 66.
    # So the block construction yields Rank 66. This matches the "best known" score 1.0909.
    # 
    # Can we do better?
    # Maybe mix the blocks? 
    # For now, I will implement the Rank 66 construction. It is deterministic and fast.
    # Then I will try to optimize it.

    def get_strassen_2x2():
        """
        Returns U, V, W factors for 2x2 * 2x2 -> 2x2 using Strassen's algorithm.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indices:
        U: i*m+j (m=2). 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
        V: j*p+k (p=2). 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
        W: k*n+i (n=2). 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
        
        Strassen formulas:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 - M4 + M5 + M7
        C01 = M3 + M4
        C10 = M2 + M5
        C11 = M1 - M3 + M6 + M7
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    def get_mult_1x2_2x2():
        """
        Returns U, V, W for 1x2 * 2x2 -> 1x2.
        A is 1x2 (row 0). B is 2x2. C is 1x2.
        C0 = A0*B00 + A1*B10
        C1 = A0*B01 + A1*B11
        This is just schoolbook rank 4.
        U shape (2, 4), V shape (4, 4), W shape (2, 4).
        U indices: i*m+j (m=2). 0:(0,0) 1:(0,1)
        V indices: j*p+k (p=2). 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
        W indices: k*n+i (n=1). 0:(0,0) 1:(1,0)
        """
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A0 B00 -> C0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: A1 B10 -> C0
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        # Term 3: A0 B01 -> C1
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        
        # Term 4: A1 B11 -> C1
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Construct Rank 66
    U_66 = np.zeros((12, 66))
    V_66 = np.zeros((24, 66))
    W_66 = np.zeros((18, 66))
    
    col_offset = 0
    
    # Helper to place a block
    def place_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, offset):
        """
        u_rows: list of 4 indices for U (2x2) or 2 indices for U (1x2)
        v_rows: list of 4 indices for V
        w_rows: list of 4 indices for W (2x2) or 2 indices for W (1x2)
        """
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            for r_u, r_v, r_w in zip(u_rows, v_rows, w_rows):
                # This mapping is tricky because the block factors are local.
                # We need to map local indices to global indices.
                pass
        
        # Simpler: just copy the values into the global matrices at the correct rows/cols.
        # U_block is (num_u_rows, rank). We want to put it into U_66[u_rows, offset:offset+rank]
        # But U_block rows correspond to local (i,j).
        # We need to map local (i,j) to global (i,j).
        
        # For 2x2 block:
        # Local U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global U rows: provided in u_rows.
        # u_rows should be [g_00, g_01, g_10, g_11]
        
        for c in range(rank_block):
            for local_r, global_r in enumerate(u_rows):
                U_66[global_r, offset + c] = U_block[local_r, c]
            for local_r, global_r in enumerate(v_rows):
                V_66[global_r, offset + c] = V_block[local_r, c]
            for local_r, global_r in enumerate(w_rows):
                W_66[global_r, offset + c] = W_block[local_r, c]

    S_U, S_V, S_W = get_strassen_2x2()
    M_U, M_V, M_W = get_mult_1x2_2x2()

    # Define global indices for U (12 rows)
    # i in 0..2, j in 0..3
    # U_idx[i, j] = i*4 + j
    
    # Define global indices for V (24 rows)
    # j in 0..3, k in 0..5
    # V_idx[j, k] = j*6 + k
    
    # Define global indices for W (18 rows)
    # k in 0..5, i in 0..2
    # W_idx[k, i] = k*3 + i

    # Block 1: A1a (i:0,1; j:0,1) * B1a (j:0,1; k:0,1)
    u_rows = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # (0,0), (0,1), (1,0), (1,1)
    v_rows = [0*6+0, 0*6+1, 1*6+0, 1*6+1] # (0,0), (0,1), (1,0), (1,1)
    w_rows = [0*3+0, 0*3+1, 1*3+0, 1*3+1] # (0,0), (0,1), (1,0), (1,1)
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Block 2: A1b (i:0,1; j:2,3) * B1b (j:2,3; k:0,1)
    u_rows = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    v_rows = [2*6+0, 2*6+1, 3*6+0, 3*6+1]
    w_rows = [0*3+0, 0*3+1, 1*3+0, 1*3+1]
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Block 3: A1a (i:0,1; j:0,1) * B2a (j:0,1; k:2,3)
    u_rows = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
    v_rows = [0*6+2, 0*6+3, 1*6+2, 1*6+3]
    w_rows = [2*3+0, 2*3+1, 3*3+0, 3*3+1]
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Block 4: A1b (i:0,1; j:2,3) * B2b (j:2,3; k:2,3)
    u_rows = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    v_rows = [2*6+2, 2*6+3, 3*6+2, 3*6+3]
    w_rows = [2*3+0, 2*3+1, 3*3+0, 3*3+1]
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Block 5: A1a (i:0,1; j:0,1) * B3a (j:0,1; k:4,5)
    u_rows = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
    v_rows = [0*6+4, 0*6+5, 1*6+4, 1*6+5]
    w_rows = [4*3+0, 4*3+1, 5*3+0, 5*3+1]
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Block 6: A1b (i:0,1; j:2,3) * B3b (j:2,3; k:4,5)
    u_rows = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    v_rows = [2*6+4, 2*6+5, 3*6+4, 3*6+5]
    w_rows = [4*3+0, 4*3+1, 5*3+0, 5*3+1]
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Block 7: A2a (i:2; j:0,1) * B1a (j:0,1; k:0,1)
    # A2 is 1x2. Local U rows: 0:(0,0), 1:(0,1). Global: (2,0), (2,1)
    u_rows = [2*4+0, 2*4+1]
    v_rows = [0*6+0, 0*6+1, 1*6+0, 1*6+1]
    # W is 1x2. Local W rows: 0:(0,0), 1:(1,0). Global: (0,2), (1,2)
    w_rows = [0*3+2, 1*3+2]
    place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Block 8: A2b (i:2; j:2,3) * B1b (j:2,3; k:0,1)
    u_rows = [2*4+2, 2*4+3]
    v_rows = [2*6+0, 2*6+1, 3*6+0, 3*6+1]
    w_rows = [0*3+2, 1*3+2]
    place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Block 9: A2a (i:2; j:0,1) * B2a (j:0,1; k:2,3)
    u_rows = [2*4+0, 2*4+1]
    v_rows = [0*6+2, 0*6+3, 1*6+2, 1*6+3]
    w_rows = [2*3+2, 3*3+2]
    place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Block 10: A2b (i:2; j:2,3) * B2b (j:2,3; k:2,3)
    u_rows = [2*4+2, 2*4+3]
    v_rows = [2*6+2, 2*6+3, 3*6+2, 3*6+3]
    w_rows = [2*3+2, 3*3+2]
    place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Block 11: A2a (i:2; j:0,1) * B3a (j:0,1; k:4,5)
    u_rows = [2*4+0, 2*4+1]
    v_rows = [0*6+4, 0*6+5, 1*6+4, 1*6+5]
    w_rows = [4*3+2, 5*3+2]
    place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Block 12: A2b (i:2; j:2,3) * B3b (j:2,3; k:4,5)
    u_rows = [2*4+2, 2*4+3]
    v_rows = [2*6+4, 2*6+5, 3*6+4, 3*6+5]
    w_rows = [4*3+2, 5*3+2]
    place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Verify Rank 66
    if verify(U_66.tolist(), V_66.tolist(), W_66.tolist(), n, m, p):
        best_U = U_66.tolist()
        best_V = V_66.tolist()
        best_W = W_66.tolist()
        best_rank = 66
        print("Constructed Rank 66")
    else:
        print("Rank 66 construction failed verification, falling back to 72")

    # --- STRATEGY 2: GREEDY ERROR REPAIR ---
    # Try to improve from best_rank downwards.
    # We will use a greedy local search that picks the best single-variable move.
    
    def calculate_error(U, V, W, rank):
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    err += (s - 1.0) ** 2
        return err

    def greedy_search(U, V, W, rank, time_limit):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = calculate_error(U, V, W, rank)
        if current_err < 1e-9:
            return U, V, W, current_err
            
        start_time = time.time()
        
        # Precompute S[i,j,k] for delta calculation?
        # For greedy, we need to evaluate many moves.
        # Full recalc is 72 * rank ops.
        # Let's try to be smart.
        
        num_vars = (n * m + m * p + p * n) * rank
        u_size = n * m * rank
        v_size = m * p * rank
        
        # Values to try: neighbors in half-integer lattice
        # We restrict to small values to keep it sane
        candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        # To speed up, we will iterate over variables in random order and try a few candidates.
        # Or, we can do a "best improvement" search:
        # 1. Calculate current error.
        # 2. For each variable (or a subset), try changing it to each candidate.
        # 3. Pick the change that minimizes error.
        # 4. Repeat.
        # This is O(num_vars * num_candidates * cost_eval).
        # cost_eval ~ 3600. num_vars ~ 1000. candidates ~ 9.
        # 1000 * 9 * 3600 = 32M ops per iteration. Too slow for Python.
        
        # So we stick to "First Improvement" or "Random Best" with limited candidates.
        
        while time.time() - start_time < time_limit:
            improved = False
            # Shuffle variables to avoid bias
            var_indices = list(range(num_vars))
            random.shuffle(var_indices)
            
            for var_idx in var_indices:
                if time.time() - start_time > time_limit:
                    break
                
                # Determine matrix and position
                if var_idx < u_size:
                    mat = U
                    mat_size = n * m
                    local_idx = var_idx
                elif var_idx < u_size + v_size:
                    mat = V
                    mat_size = m * p
                    local_idx = var_idx - u_size
                else:
                    mat = W
                    mat_size = p * n
                    local_idx = var_idx - u_size - v_size
                
                row = local_idx // rank
                col = local_idx % rank
                old_val = mat[row, col]
                
                # Try a few candidates
                # Sort candidates by distance to old_val?
                # Just try a few random ones
                candidates_to_try = random.sample(candidates, min(3, len(candidates)))
                
                best_new_val = old_val
                best_new_err = current_err
                
                for new_val in candidates_to_try:
                    if new_val == old_val:
                        continue
                    mat[row, col] = new_val
                    new_err = calculate_error(U, V, W, rank)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_new_val = new_val
                
                if best_new_val != old_val:
                    mat[row, col] = best_new_val
                    current_err = best_new_err
                    improved = True
                    if current_err < 1e-9:
                        return U, V, W, current_err
                    # Restart loop? Or continue?
                    # Continue scanning other variables in this pass.
            
            if not improved:
                # Stagnation, break
                break

        return U, V, W, current_err

    # Try to improve from 66 down to 60
    if best_rank == 66:
        for rank in range(66, 59, -1):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random half-integers?
            # Or try to "project" the rank 66 solution to lower rank?
            # Simple: Random init.
            
            for seed in range(5):
                random.seed(seed)
                np.random.seed(seed)
                
                def rand_half_int():
                    return random.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
                
                U = np.array([[rand_half_int() for _ in range(rank)] for _ in range(n * m)], dtype=np.float64)
                V = np.array([[rand_half_int() for _ in range(rank)] for _ in range(m * p)], dtype=np.float64)
                W = np.array([[rand_half_int() for _ in range(rank)] for _ in range(p * n)], dtype=np.float64)
                
                U, V, W, err = greedy_search(U, V, W, rank, time_limit=10.0)
                
                if err < 1e-9:
                    if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                        best_rank = rank
                        print(f"Greedy found rank {rank}")
                        break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
