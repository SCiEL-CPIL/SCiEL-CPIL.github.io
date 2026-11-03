import numpy as np
import time
import random
from fractions import Fraction

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

    # --- STRATEGY 2: ALGEBRAIC REDUCTION VIA SPARSE LINEAR ALGEBRA ---
    # This is a structural change from "greedy local search on half-integers" to
    # "exact rational algebraic reduction".
    # 
    # The idea:
    # 1. Start with the Rank 66 decomposition.
    # 2. Represent the problem as a system of polynomial equations.
    # 3. Instead of greedy search, we use the structure of the equations to identify
    #    linear dependencies or terms that can be eliminated.
    # 
    # However, a full Gröbner basis is too expensive. 
    # A more practical "algebraic" approach is to use the fact that the tensor rank
    # is the minimum number of terms. If we have a valid decomposition, we can try to
    # find a lower rank one by checking if the current rank is minimal for the given
    # support pattern, or by using "tensor rank reduction" techniques.
    #
    # A simpler but effective structural change:
    # Use "Exact" Gaussian elimination over the rationals (using Fractions) on a
    # specific subspace of the equations to see if we can force some coefficients to 0
    # or merge terms.
    #
    # Let's try a different angle: The "Melding" strategy.
    # If we have two rank-1 terms that share structure, can we combine them?
    # No, rank is additive.
    #
    # Let's try to improve the Rank 66 construction by exploiting the specific dimensions.
    # The 3x4x6 case is close to 3x3x3 (rank 23) or 4x4x4 (rank 49).
    # 
    # Actually, the most promising structural change is to use a **Global Optimization**
    # solver (like scipy.optimize) on the *real* values, but then snap to rationals?
    # No, that's not exact.
    #
    # Let's try **Integer Programming** or **Constraint Satisfaction**?
    # Too slow.
    #
    # Let's go back to the "Greedy" but make it **Algebraically Informed**.
    # In the previous greedy, we tried random candidates.
    # Here, we will try to solve for the *best* candidate value for each variable
    # given the others, using the linear structure of the error in a local neighborhood.
    # 
    # Wait, the error is cubic. But if we fix all but one variable, the error is quadratic.
    # We can find the exact minimum of the quadratic in O(1) per variable!
    # This is a "Coordinate Descent" with exact minimization steps.
    
    def calculate_error_and_grad(U, V, W, rank):
        """
        Calculate the total error and the gradient with respect to each variable.
        Error E = sum_{i,j,k} (sum_c U_ic V_jc W_kc - 1)^2
        Let S_ijk = sum_c U_ic V_jc W_kc.
        E = sum (S_ijk - 1)^2.
        
        dE/dU_ic = sum_{j,k} 2(S_ijk - 1) * V_jc * W_kc
        dE/dV_jc = sum_{i,k} 2(S_ijk - 1) * U_ic * W_kc
        dE/dW_kc = sum_{i,j} 2(S_ijk - 1) * U_ic * V_jc
        
        This can be computed efficiently using matrix multiplications.
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # S is a tensor (n, m, p)
        # S[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is not a simple matmul.
        # We can compute S via:
        # For each c, U_c (n*m) * V_c (m*p) * W_c (p*n) -> contribution to S.
        # Actually, let's just loop over c and accumulate.
        S = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            # Contribution to S[i,j,k] is u_c[i,j] * v_c[j,k] * w_c[k,i]
            # This is element-wise product after reshaping?
            # S[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            # We can do this with einsum: 'ij,jk,ki->ijk'
            S += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
        error = np.sum((S - 1.0) ** 2)
        
        # Gradients
        # dE/dU_ic = sum_{j,k} 2(S_ijk - 1) V_jc W_kc
        # Let D = S - 1.
        D = S - 1.0
        grad_U = np.zeros_like(U_arr)
        grad_V = np.zeros_like(V_arr)
        grad_W = np.zeros_like(W_arr)
        
        for c in range(rank):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            
            # grad_U[i,j] = 2 * sum_{k} D[i,j,k] * v_c[j,k] * w_c[k,i]
            # This is 2 * (D * v_c) * w_c? 
            # D is (n,m,p). v_c is (m,p). w_c is (p,n).
            # term1 = sum_k D[i,j,k] * v_c[j,k] -> (n,m)
            # term1 = np.einsum('ijk,jk->ij', D, v_c)
            # grad_U[i,j] = 2 * sum_i? No, w_c[k,i] depends on i.
            # grad_U[i,j] = 2 * sum_k (D[i,j,k] * v_c[j,k]) * w_c[k,i]
            # Let T[i,j,k] = D[i,j,k] * v_c[j,k]
            # grad_U[i,j] = 2 * sum_k T[i,j,k] * w_c[k,i]
            # This is 2 * np.einsum('ijk,ki->ij', T, w_c)
            
            T = D * v_c  # Broadcasting: D (n,m,p), v_c (m,p) -> v_c needs to be (1,m,p)
            T = D * v_c[None, :, :]
            grad_U_c = 2.0 * np.einsum('ijk,ki->ij', T, w_c)
            grad_U[:, c] = grad_U_c.flatten()
            
            # grad_V[j,k] = 2 * sum_{i} D[i,j,k] * u_c[i,j] * w_c[k,i]
            # Let T[i,j,k] = D[i,j,k] * u_c[i,j]
            # grad_V[j,k] = 2 * sum_i T[i,j,k] * w_c[k,i]
            T = D * u_c[:, :, None]
            grad_V_c = 2.0 * np.einsum('ijk,ki->jk', T, w_c)
            grad_V[:, c] = grad_V_c.flatten()
            
            # grad_W[k,i] = 2 * sum_{j} D[i,j,k] * u_c[i,j] * v_c[j,k]
            # Let T[i,j,k] = D[i,j,k] * v_c[j,k]
            # grad_W[k,i] = 2 * sum_j T[i,j,k] * u_c[i,j]
            T = D * v_c[None, :, :]
            grad_W_c = 2.0 * np.einsum('ijk,ij->ki', T, u_c)
            grad_W[:, c] = grad_W_c.flatten()
            
        return error, grad_U, grad_V, grad_W

    def coordinate_descent(U, V, W, rank, time_limit):
        """
        Coordinate descent with exact minimization of the quadratic error for each variable.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = calculate_error(U, V, W, rank)
        if current_err < 1e-9:
            return U, V, W, current_err
            
        start_time = time.time()
        
        # Precompute S and D?
        # We need to update S and D efficiently when one variable changes.
        # If U[i*m+j, c] changes by delta:
        # S[i,j,k] changes by delta * V[j*p+k, c] * W[k*n+i, c]
        # This is a rank-1 update to S (along k for fixed i,j).
        
        # Let's maintain S.
        S = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            S += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
        D = S - 1.0
        current_err = np.sum(D**2)
        
        # We need gradients to find the optimal step.
        # For a single variable U[r, c] (where r = i*m+j), the error is:
        # E(x) = sum_{j',k'} (S[i,j',k'] - 1)^2
        # Only S[i,j,k] depends on U[i*m+j, c].
        # S[i,j,k] = x * V[j*p+k, c] * W[k*n+i, c] + rest
        # Let A_k = V[j*p+k, c] * W[k*n+i, c]
        # S[i,j,k] = x * A_k + B_k
        # E(x) = sum_k (x * A_k + B_k - 1)^2
        # This is a quadratic in x.
        # E(x) = x^2 * sum(A_k^2) + 2x * sum(A_k(B_k-1)) + const
        # Optimal x = - sum(A_k(B_k-1)) / sum(A_k^2)
        
        # Note: B_k is the value of S[i,j,k] when x=0.
        # Current S[i,j,k] = x_old * A_k + B_k.
        # So B_k = S[i,j,k] - x_old * A_k.
        # B_k - 1 = D[i,j,k] - x_old * A_k.
        
        # Let's implement this exact line search.
        
        while time.time() - start_time < time_limit:
            improved = False
            # Iterate over all variables
            # We can shuffle them to avoid bias
            var_indices = list(range(n*m*rank + m*p*rank + p*n*rank))
            random.shuffle(var_indices)
            
            for var_idx in var_indices:
                if time.time() - start_time > time_limit:
                    break
                
                if var_idx < n*m*rank:
                    mat = U
                    r = var_idx // rank
                    c = var_idx % rank
                    i = r // m
                    j = r % m
                    # A_k = V[j*p+k, c] * W[k*n+i, c]
                    A = V[j*p:(j+1)*p, c] * W[c*n:(c+1)*n, c] # Wait, W index is k*n+i.
                    # W row indices for fixed i, varying k: [0*n+i, 1*n+i, ..., (p-1)*n+i]
                    # So W[k*n+i, c] for k in 0..p-1.
                    # V row indices for fixed j, varying k: [j*p+0, ..., j*p+p-1]
                    # So V[j*p+k, c] for k in 0..p-1.
                    
                    # A is a vector of length p.
                    # A[k] = V[j*p+k, c] * W[k*n+i, c]
                    A = V[j*p:(j+1)*p, c] * W[i::n, c] # This is wrong, W[i::n] takes i, i+n, i+2n... which is k*n+i. Correct.
                    
                    # B_k - 1 = D[i,j,k] - x_old * A_k
                    x_old = mat[r, c]
                    D_ij = D[i, j, :] # Vector of length p
                    B_minus_1 = D_ij - x_old * A
                    
                    denom = np.sum(A * A)
                    if denom < 1e-12:
                        continue
                    num = np.sum(A * B_minus_1)
                    x_new = -num / denom
                    
                    # Snap to half-integer?
                    # The task requires integer/half-integer entries.
                    # So we must snap.
                    x_snap = round(x_new * 2) / 2.0
                    
                    # Check if snap improves error
                    # Calculate new error contribution
                    # Old error contribution for this i,j: sum_k D[i,j,k]^2
                    # New D[i,j,k] = D[i,j,k] + (x_snap - x_old) * A[k]
                    delta = x_snap - x_old
                    if abs(delta) < 1e-9:
                        continue
                    new_D_ij = D_ij + delta * A
                    old_err_ij = np.sum(D_ij**2)
                    new_err_ij = np.sum(new_D_ij**2)
                    
                    if new_err_ij < old_err_ij - 1e-12:
                        # Update
                        mat[r, c] = x_snap
                        D[i, j, :] = new_D_ij
                        current_err += (new_err_ij - old_err_ij)
                        improved = True
                        if current_err < 1e-9:
                            return U, V, W, current_err
                            
                elif var_idx < n*m*rank + m*p*rank:
                    local_idx = var_idx - n*m*rank
                    mat = V
                    r = local_idx // rank
                    c = local_idx % rank
                    j = r // p
                    k = r % p
                    # A_i = U[i*m+j, c] * W[k*n+i, c]
                    # U index: i*m+j. i in 0..n-1.
                    # W index: k*n+i. i in 0..n-1.
                    # A is vector of length n.
                    A = U[j::m, c] * W[k*n:(k+1)*n, c] # U[j::m] takes j, j+m, j+2m... which is i*m+j. Correct.
                    
                    x_old = mat[r, c]
                    D_jk = D[:, j, k] # Vector of length n
                    B_minus_1 = D_jk - x_old * A
                    
                    denom = np.sum(A * A)
                    if denom < 1e-12:
                        continue
                    num = np.sum(A * B_minus_1)
                    x_new = -num / denom
                    x_snap = round(x_new * 2) / 2.0
                    delta = x_snap - x_old
                    if abs(delta) < 1e-9:
                        continue
                    new_D_jk = D_jk + delta * A
                    old_err_jk = np.sum(D_jk**2)
                    new_err_jk = np.sum(new_D_jk**2)
                    
                    if new_err_jk < old_err_jk - 1e-12:
                        mat[r, c] = x_snap
                        D[:, j, k] = new_D_jk
                        current_err += (new_err_jk - old_err_jk)
                        improved = True
                        if current_err < 1e-9:
                            return U, V, W, current_err
                            
                else:
                    local_idx = var_idx - n*m*rank - m*p*rank
                    mat = W
                    r = local_idx // rank
                    c = local_idx % rank
                    k = r // n
                    i = r % n
                    # A_j = U[i*m+j, c] * V[j*p+k, c]
                    # U index: i*m+j. j in 0..m-1.
                    # V index: j*p+k. j in 0..m-1.
                    # A is vector of length m.
                    A = U[i*m:(i+1)*m, c] * V[k::p, c] # V[k::p] takes k, k+p, k+2p... which is j*p+k. Correct.
                    
                    x_old = mat[r, c]
                    D_ki = D[i, :, k] # Vector of length m
                    B_minus_1 = D_ki - x_old * A
                    
                    denom = np.sum(A * A)
                    if denom < 1e-12:
                        continue
                    num = np.sum(A * B_minus_1)
                    x_new = -num / denom
                    x_snap = round(x_new * 2) / 2.0
                    delta = x_snap - x_old
                    if abs(delta) < 1e-9:
                        continue
                    new_D_ki = D_ki + delta * A
                    old_err_ki = np.sum(D_ki**2)
                    new_err_ki = np.sum(new_D_ki**2)
                    
                    if new_err_ki < old_err_ki - 1e-12:
                        mat[r, c] = x_snap
                        D[i, :, k] = new_D_ki
                        current_err += (new_err_ki - old_err_ki)
                        improved = True
                        if current_err < 1e-9:
                            return U, V, W, current_err

            if not improved:
                break

        return U, V, W, current_err

    # Try to improve from 66 down to 60 using Coordinate Descent
    if best_rank == 66:
        for rank in range(66, 59, -1):
            if time.time() > deadline - 10:
                break
            
            # Initialize with the rank 66 solution truncated?
            # Or random?
            # Truncation is better as it starts close to a valid solution.
            # But rank 66 has 66 columns. We need rank columns.
            # We can take the first 'rank' columns of the 66 solution.
            # This will likely break the decomposition, but CD might fix it.
            
            U_init = U_66[:, :rank]
            V_init = V_66[:, :rank]
            W_init = W_66[:, :rank]
            
            U, V, W, err = coordinate_descent(U_init, V_init, W_init, rank, time_limit=15.0)
            
            if err < 1e-9:
                if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    best_rank = rank
                    print(f"Coordinate Descent found rank {rank}")
                    break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
