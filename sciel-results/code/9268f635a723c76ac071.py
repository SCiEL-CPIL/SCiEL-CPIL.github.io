"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will construct a decomposition of rank 60 using block-strassen logic.
    # Partition A (3x3) into:
    # A11 (2x2), A12 (2x1)
    # A21 (1x2), A22 (1x1)
    # Partition B (3x7) into:
    # B1 (2x7) (rows 0,1)
    # B2 (1x7) (row 2)
    
    # C = A B
    # C1 (2x7) = A11 B1 + A12 B2
    # C2 (1x7) = A21 B1 + A22 B2
    
    # We need to compute:
    # 1. A11 B1 (2x2 * 2x7) -> Rank 25
    # 2. A12 B2 (2x1 * 1x7) -> Rank 14
    # 3. A21 B1 (1x2 * 2x7) -> Rank 14
    # 4. A22 B2 (1x1 * 1x7) -> Rank 7
    # Total Rank = 25 + 14 + 14 + 7 = 60
    
    U = []
    V = []
    W = []
    
    # Helper to add a decomposition
    def add_decomp(U_local, V_local, W_local, row_offset_U, row_offset_V, row_offset_W):
        # U_local: (n_local*m_local, rank)
        # V_local: (m_local*p_local, rank)
        # W_local: (p_local*n_local, rank)
        # We need to map local indices to global indices.
        # This is tricky because the local tensors are sub-tensors of the global one.
        # Instead of generic mapping, I will construct the global U,V,W directly by filling zeros and adding contributions.
        pass

    # Given the complexity of generic mapping, I will build the 60-rank decomposition explicitly.
    # Global U size: 3*3 = 9 rows.
    # Global V size: 3*7 = 21 rows.
    # Global W size: 7*3 = 21 rows.
    
    rank_total = 60
    U = [[0.0] * rank_total for _ in range(9)]
    V = [[0.0] * rank_total for _ in range(21)]
    W = [[0.0] * rank_total for _ in range(21)]
    
    col_offset = 0
    
    # --- Part 1: A11 (2x2) * B1 (2x7) ---
    # Indices:
    # A11 corresponds to A[0:2, 0:2] -> Global U rows: 0,1,2,3 (i=0,1; j=0,1)
    # B1 corresponds to B[0:2, 0:7] -> Global V rows: 0:14 (j=0,1; k=0..6)
    # Output C1_top corresponds to C[0:2, 0:7] -> Global W rows: 0:14 (k=0..6; i=0,1)
    
    # We use the 2x2x7 rank 25 decomposition.
    # It consists of 3 Strassen 2x2x2 blocks (rank 7 each) and 1 schoolbook 2x2x1 block (rank 4).
    
    def get_strassen_2x2x2_coefficients():
        # Returns U, V, W for 2x2x2 multiplication, rank 7.
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U_s = [
            [1, 0, 0, 0, 0, 0, 0], # (0,0)
            [0, 1, 0, 0, 0, 0, 0], # (0,1)
            [0, 0, 1, 0, 0, 0, 0], # (1,0)
            [0, 0, 0, 1, 0, 0, 0], # (1,1)
        ]
        # Wait, Strassen coefficients are specific.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # U (A factors):
        # M1: A00, A11
        # M2: A10, A11
        # M3: A00
        # M4: A11
        # M5: A00, A01
        # M6: A10, A00
        # M7: A01, A11
        
        U_s = [
            [1, 0, 1, 0, 1, 1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1], # A01
            [0, 1, 0, 0, 0, 1, 0], # A10
            [1, 1, 0, 1, 0, 0, 1], # A11
        ]
        
        # V (B factors):
        # M1: B00, B11
        # M2: B00
        # M3: B10, B11
        # M4: B01, B00
        # M5: B11
        # M6: B00, B01
        # M7: B10, B11
        
        V_s = [
            [1, 1, 0, 1, 0, 1, 0], # B00 (j=0,k=0)
            [0, 0, 0, -1, 0, 0, 1], # B01 (j=0,k=1)
            [0, 0, 1, 0, 0, 0, 1], # B10 (j=1,k=0)
            [1, 0, -1, 0, 1, 0, 0], # B11 (j=1,k=1)
        ]
        
        # W (C factors):
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        W_s = [
            [1, 0, 0, 1, -1, 0, -1], # C00
            [0, 0, 1, 0, 1, 0, 0],   # C01
            [0, 1, 0, 1, 0, 0, 0],   # C10
            [1, -1, 1, 0, 0, 0, 1]   # C11
        ]
        
        return U_s, V_s, W_s

    def get_schoolbook_2x2x1_coefficients():
        # 2x2 * 2x1
        # U: 2x2, V: 2x1, W: 1x2
        # Rank 4
        U_s = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_s = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        # W[k*n+i, r] -> n=2, p=1? No, 2x2 * 2x1.
        # Output is 2x1.
        # W rows: 1*2 = 2 rows.
        # W[0, r] for C0, W[1, r] for C1.
        W_s = [
            [1, 0, 0, 0],
            [0, 0, 1, 0] # Wait, C0 = A00*B0 + A01*B1. C1 = A10*B0 + A11*B1.
        ]
        # Correct W:
        # Term 0: A00 B0 -> C0
        # Term 1: A01 B1 -> C0
        # Term 2: A10 B0 -> C1
        # Term 3: A11 B1 -> C1
        W_s = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        return U_s, V_s, W_s

    # Add Strassen block for cols 0,1
    U_s, V_s, W_s = get_strassen_2x2x2_coefficients()
    # Map to global indices for A11 (rows 0,1; cols 0,1), B1 (rows 0,1; cols 0,1), C1_top (rows 0,1; cols 0,1)
    # Global U rows: 0,1,2,3
    # Global V rows: 0,1,7,8 (j=0,1; k=0,1) -> V[j*7+k]
    # Global W rows: 0,1,14,15? No, W[k*n+i]. n=3.
    # C1_top is C[0:2, 0:2].
    # W index: k*3 + i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 3
    # k=0, i=1 -> 1
    # k=1, i=1 -> 4
    
    # This mapping is error-prone. I will use a generic add function.
    
    def add_block(U_b, V_b, W_b, u_map, v_map, w_map, offset):
        # u_map: list of 4 global row indices for U_b rows 0..3
        # v_map: list of 4 global row indices for V_b rows 0..3
        # w_map: list of 4 global row indices for W_b rows 0..3
        rank_b = len(U_b[0])
        for r in range(rank_b):
            for i in range(4):
                U[u_map[i]][offset + r] += U_b[i][r]
            for j in range(4):
                V[v_map[j]][offset + r] += V_b[j][r]
            for k in range(4):
                W[w_map[k]][offset + r] += W_b[k][r]
        return rank_b

    # Block 1: A11(0:2,0:2) * B1(0:2,0:2) -> C(0:2,0:2)
    # U rows: 0,1,2,3
    # V rows: 0,1,7,8 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    # W rows: 0,3,1,4 (k=0,i=0; k=1,i=0; k=0,i=1; k=1,i=1)
    u_map = [0, 1, 2, 3]
    v_map = [0, 1, 7, 8]
    w_map = [0, 3, 1, 4]
    rank_used = add_block(U_s, V_s, W_s, u_map, v_map, w_map, col_offset)
    col_offset += rank_used

    # Block 2: A11(0:2,0:2) * B1(0:2,2:4) -> C(0:2,2:4)
    # U rows: 0,1,2,3
    # V rows: 2,3,9,10 (j=0,k=2; j=0,k=3; j=1,k=2; j=1,k=3)
    # W rows: 2,5,3,6 (k=2,i=0; k=3,i=0; k=2,i=1; k=3,i=1)
    u_map = [0, 1, 2, 3]
    v_map = [2, 3, 9, 10]
    w_map = [2, 5, 3, 6]
    rank_used = add_block(U_s, V_s, W_s, u_map, v_map, w_map, col_offset)
    col_offset += rank_used

    # Block 3: A11(0:2,0:2) * B1(0:2,4:6) -> C(0:2,4:6)
    # U rows: 0,1,2,3
    # V rows: 4,5,11,12
    # W rows: 4,7,5,8
    u_map = [0, 1, 2, 3]
    v_map = [4, 5, 11, 12]
    w_map = [4, 7, 5, 8]
    rank_used = add_block(U_s, V_s, W_s, u_map, v_map, w_map, col_offset)
    col_offset += rank_used

    # Block 4: A11(0:2,0:2) * B1(0:2,6:7) -> C(0:2,6:7)
    # Schoolbook 2x2x1
    U_s4, V_s4, W_s4 = get_schoolbook_2x2x1_coefficients()
    # U rows: 0,1,2,3
    # V rows: 6, 13 (j=0,k=6; j=1,k=6)
    # W rows: 6, 9 (k=6,i=0; k=6,i=1)
    # But Schoolbook 2x2x1 has V size 2x1=2 rows? No, V is (m*p, rank). m=2, p=1. V has 2 rows.
    # My get_schoolbook_2x2x1 returns V with 2 rows?
    # V_s was 2x2 in 2x2x2. Here it is 2x1.
    # Let's re-verify get_schoolbook_2x2x1.
    # U: 2x2. V: 2x1. W: 1x2.
    # V rows: 0, 1.
    # W rows: 0, 1.
    # Mapping:
    # V global: 6 (j=0,k=6), 13 (j=1,k=6).
    # W global: 6 (k=6,i=0), 9 (k=6,i=1).
    u_map = [0, 1, 2, 3]
    v_map = [6, 13] # Only 2 rows for V
    w_map = [6, 9] # Only 2 rows for W
    # But add_block expects 4 rows for V and W?
    # I need a different add function or pad.
    # Let's just do it manually for this small block.
    for r in range(4):
        for i in range(4):
            U[u_map[i]][col_offset + r] += U_s4[i][r]
        # V has only 2 rows
        V[v_map[0]][col_offset + r] += V_s4[0][r]
        V[v_map[1]][col_offset + r] += V_s4[1][r]
        # W has only 2 rows
        W[w_map[0]][col_offset + r] += W_s4[0][r]
        W[w_map[1]][col_offset + r] += W_s4[1][r]
    col_offset += 4

    # --- Part 2: A12 (2x1) * B2 (1x7) ---
    # A12 is A[0:2, 2]. Global U rows: 2, 5 (i=0,j=2; i=1,j=2)
    # B2 is B[2, 0:7]. Global V rows: 14,15,16,17,18,19,20 (j=2, k=0..6)
    # Output C1_bot? No, C1 is 2x7. C1[0:2, 0:7].
    # C1 = A11 B1 + A12 B2.
    # So we add to C1.
    # W rows: 0..6, 3..9, 6..12?
    # C1 rows are 0,1.
    # W[k*3+i].
    # For i=0,1 and k=0..6.
    # W rows: 0,3,6,9,12,15,18 (i=0) and 1,4,7,10,13,16,19 (i=1).
    
    # Rank 14.
    # U: 2x1. V: 1x7. W: 7x2.
    # Schoolbook.
    for k in range(7):
        # Term: A[0,2] * B[2,k] -> C[0,k]
        U[2][col_offset] = 1
        V[14+k][col_offset] = 1
        W[k*3+0][col_offset] = 1
        col_offset += 1
        # Term: A[1,2] * B[2,k] -> C[1,k]
        U[5][col_offset] = 1
        V[14+k][col_offset] = 1
        W[k*3+1][col_offset] = 1
        col_offset += 1

    # --- Part 3: A21 (1x2) * B1 (2x7) ---
    # A21 is A[2, 0:2]. Global U rows: 6, 7 (i=2, j=0,1)
    # B1 is B[0:2, 0:7]. Global V rows: 0..13.
    # Output C2 (1x7). C2[0, 0:7].
    # W rows: k*3+2 for k=0..6.
    
    # Rank 14.
    for k in range(7):
        # Term: A[2,0] * B[0,k] -> C[2,k]
        U[6][col_offset] = 1
        V[k][col_offset] = 1
        W[k*3+2][col_offset] = 1
        col_offset += 1
        # Term: A[2,1] * B[1,k] -> C[2,k]
        U[7][col_offset] = 1
        V[7+k][col_offset] = 1
        W[k*3+2][col_offset] = 1
        col_offset += 1

    # --- Part 4: A22 (1x1) * B2 (1x7) ---
    # A22 is A[2,2]. Global U row: 8.
    # B2 is B[2, 0:7]. Global V rows: 14..20.
    # Output C2[0, 0:7].
    # W rows: k*3+2.
    
    # Rank 7.
    for k in range(7):
        U[8][col_offset] = 1
        V[14+k][col_offset] = 1
        W[k*3+2][col_offset] = 1
        col_offset += 1

    # Total rank used: 7+7+7+4 + 14 + 14 + 7 = 60.
    # col_offset should be 60.
    
    # Verify
    def verify(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    s = 0
                    for r in range(60):
                        s += U[i*3+j][r] * V[j*7+k][r] * W[k*3+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {'U': U, 'V': V, 'W': W}
    else:
        # Fallback to schoolbook
        rank = 63
        U_sb = [[0]*rank for _ in range(9)]
        V_sb = [[0]*rank for _ in range(21)]
        W_sb = [[0]*rank for _ in range(21)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    U_sb[i*3+j][col] = 1
                    V_sb[j*7+k][col] = 1
                    W_sb[k*3+i][col] = 1
                    col += 1
        return {'U': U_sb, 'V': V_sb, 'W': W_sb}

# EVOLVE-BLOCK-END
