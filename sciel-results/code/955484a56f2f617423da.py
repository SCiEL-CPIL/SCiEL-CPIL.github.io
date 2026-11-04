import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_UVW():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Rank = 7.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _trivial_2x2_1_UVW():
    """
    Return trivial decomposition for 2x2 * 2x1 -> 2x1.
    Rank = 4.
    U: 4x4, V: 2x4, W: 2x4
    A: 2x2 (idx 0..3)
    B: 2x1 (idx 0..1)
    C: 2x1 (idx 0..1)
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C0 = A00 B0 + A01 B1
    # Term 0: A00 B0 -> C0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 1: A01 B1 -> C0
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # C1 = A10 B0 + A11 B1
    # Term 2: A10 B0 -> C1
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # Term 3: A11 B1 -> C1
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def _m_2x2_3_UVW():
    """
    Construct decomposition for 2x2 * 2x3 -> 2x3.
    Rank = 11 (7 from Strassen for first 2 cols, 4 from Trivial for 3rd col).
    U: 4x11, V: 6x11, W: 6x11
    A: 2x2 (idx 0..3)
    B: 2x3 (idx 0..5). Cols 0,1 for Strassen, Col 2 for Trivial.
    C: 2x3 (idx 0..5). Cols 0,1 for Strassen, Col 2 for Trivial.
    Note: W index is k*2+i? No, local 2x3 output.
    Local C indices: 
    Col 0: 0, 1
    Col 1: 2, 3
    Col 2: 4, 5
    """
    U_s, V_s, W_s = _strassen_2x2_2x2_UVW()
    U_t, V_t, W_t = _trivial_2x2_1_UVW()
    
    rank = 11
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Copy Strassen part (first 7 ranks)
    # U: 4x7 -> 4x11
    for i in range(4):
        for r in range(7):
            U[i][r] = U_s[i][r]
            
    # V: 4x7 -> 6x11
    # Strassen V uses B cols 0,1.
    # Local B idx: 0,1 (Col 0), 2,3 (Col 1)? 
    # Wait, Strassen 2x2x2 assumes B is 2x2.
    # B indices in Strassen: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1).
    # In 2x3 B, Col 0 is indices 0,1. Col 1 is indices 2,3.
    # So V_s[0] (B00) -> V[0]
    # V_s[1] (B01) -> V[2]
    # V_s[2] (B10) -> V[1]
    # V_s[3] (B11) -> V[3]
    
    mapping_V_s = {0:0, 1:2, 2:1, 3:3}
    for i_s in range(4):
        i_v = mapping_V_s[i_s]
        for r in range(7):
            V[i_v][r] = V_s[i_s][r]
            
    # W: 4x7 -> 6x11
    # Strassen W uses C cols 0,1.
    # W_s indices: 0,1 (Col 0), 2,3 (Col 1).
    # In 2x3 C, Col 0 is indices 0,1. Col 1 is indices 2,3.
    # So W_s[0] -> W[0], W_s[1] -> W[1], W_s[2] -> W[2], W_s[3] -> W[3].
    for i in range(4):
        for r in range(7):
            W[i][r] = W_s[i][r]
            
    # Copy Trivial part (next 4 ranks, offset 7)
    # U: 4x4 -> 4x11 (offset 7)
    for i in range(4):
        for r in range(4):
            U[i][7+r] = U_t[i][r]
            
    # V: 2x4 -> 6x11 (offset 7)
    # Trivial V uses B col 2.
    # Local B idx for Col 2: 4, 5.
    # V_t[0] (B0 of col 2) -> V[4]
    # V_t[1] (B1 of col 2) -> V[5]
    mapping_V_t = {0:4, 1:5}
    for i_t in range(2):
        i_v = mapping_V_t[i_t]
        for r in range(4):
            V[i_v][7+r] = V_t[i_t][r]
            
    # W: 2x4 -> 6x11 (offset 7)
    # Trivial W uses C col 2.
    # Local C idx for Col 2: 4, 5.
    # W_t[0] (C0 of col 2) -> W[4]
    # W_t[1] (C1 of col 2) -> W[5]
    mapping_W_t = {0:4, 1:5}
    for i_t in range(2):
        i_w = mapping_W_t[i_t]
        for r in range(4):
            W[i_w][7+r] = W_t[i_t][r]
            
    return U, V, W

def _construct_4x4_4x7_UVW():
    """
    Construct the decomposition for 4x4 * 4x7 -> 4x7.
    Rank = 93 (49 from 4x4x4 Strassen + 44 from 4x4x3 block decomposition).
    """
    n, m, p = 4, 4, 7
    rank_head = 49
    rank_tail = 44
    total_rank = rank_head + rank_tail
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Head Part: 4x4 * 4x4 -> 4x4 (Strassen) ---
    # Uses B cols 0-3, C cols 0-3.
    U_h, V_h, W_h = _strassen_4x4_4x4_UVW()
    
    for i in range(16):
        for r in range(rank_head):
            U[i][r] = U_h[i][r]
            
    # V_h is 16x49. B indices 0-15 (cols 0-3).
    # Global V indices for cols 0-3:
    # Row 0: 0,1,2,3
    # Row 1: 7,8,9,10
    # Row 2: 14,15,16,17
    # Row 3: 21,22,23,24
    # V_h row j corresponds to B row j.
    # V_h[j*4+k] is B[j, k] for k in 0..3.
    # Global V index for B[j, k] is j*7 + k.
    for j in range(4):
        for k in range(4):
            local_v_idx = j * 4 + k
            global_v_idx = j * 7 + k
            for r in range(rank_head):
                V[global_v_idx][r] = V_h[local_v_idx][r]
                
    # W_h is 16x49. C indices 0-15 (cols 0-3).
    # W_h index k*n+i.
    # Global W index k*n+i.
    for i in range(16):
        for r in range(rank_head):
            W[i][r] = W_h[i][r]

    # --- Tail Part: 4x4 * 4x3 -> 4x3 (4 blocks of 2x2x3) ---
    # Uses B cols 4-6, C cols 4-6.
    U_t_2x2_3, V_t_2x2_3, W_t_2x2_3 = _m_2x2_3_UVW()
    
    # Blocks:
    # 1. A00 * B_top -> C_top
    # 2. A01 * B_bot -> C_top
    # 3. A10 * B_top -> C_bot
    # 4. A11 * B_bot -> C_bot
    
    # A00: A rows 0,1 cols 0,1. Indices 0,1,4,5.
    # A01: A rows 0,1 cols 2,3. Indices 2,3,6,7.
    # A10: A rows 2,3 cols 0,1. Indices 8,9,12,13.
    # A11: A rows 2,3 cols 2,3. Indices 10,11,14,15.
    
    # B_top: B rows 0,1 cols 4,5,6.
    # B indices:
    # Row 0: 4,5,6
    # Row 1: 11,12,13
    # Local B indices for 2x3:
    # (0,0)->4, (0,1)->5, (0,2)->6
    # (1,0)->11, (1,1)->12, (1,2)->13
    
    # B_bot: B rows 2,3 cols 4,5,6.
    # B indices:
    # Row 2: 18,19,20
    # Row 3: 25,26,27
    # Local B indices for 2x3:
    # (0,0)->18, (0,1)->19, (0,2)->20
    # (1,0)->25, (1,1)->26, (1,2)->27
    
    # C_top: C rows 0,1 cols 4,5,6.
    # C indices (k*n+i):
    # Col 4: 16,17
    # Col 5: 20,21
    # Col 6: 24,25
    # Local C indices for 2x3:
    # (0,0)->16, (0,1)->20, (0,2)->24
    # (1,0)->17, (1,1)->21, (1,2)->25
    
    # C_bot: C rows 2,3 cols 4,5,6.
    # C indices (k*n+i):
    # Col 4: 18,19
    # Col 5: 22,23
    # Col 6: 26,27
    # Local C indices for 2x3:
    # (0,0)->18, (0,1)->22, (0,2)->26
    # (1,0)->19, (1,1)->23, (1,2)->27
    
    def add_block(a_start, b_start, c_start, rank_offset):
        # a_start: index of A00 in global A (0..15)
        # b_start: index of B00 in global B (0..27)
        # c_start: index of C00 in global C (0..27)
        
        # A mapping: Local 0,1,2,3 -> Global a_start + [0,1,4,5]?
        # No, A is 4x4.
        # A00 is rows 0,1 cols 0,1.
        # Local A idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # Global A idx:
        # (0,0) -> 0
        # (0,1) -> 1
        # (1,0) -> 4
        # (1,1) -> 5
        # So for A00, global indices are 0,1,4,5.
        # For A01, global indices are 2,3,6,7.
        # For A10, global indices are 8,9,12,13.
        # For A11, global indices are 10,11,14,15.
        
        a_map = {
            0: a_start + 0,
            1: a_start + 1,
            2: a_start + 4,
            3: a_start + 5
        }
        
        # B mapping: Local 0..5 -> Global b_start + ...
        # Local B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # Global B for B_top (rows 0,1, cols 4,5,6):
        # (0,0) -> 0*7+4 = 4
        # (0,1) -> 0*7+5 = 5
        # (0,2) -> 0*7+6 = 6
        # (1,0) -> 1*7+4 = 11
        # (1,1) -> 1*7+5 = 12
        # (1,2) -> 1*7+6 = 13
        # b_start is 4 for B_top.
        # b_start is 18 for B_bot.
        
        b_map = {
            0: b_start + 0,
            1: b_start + 1,
            2: b_start + 2,
            3: b_start + 7,
            4: b_start + 8,
            5: b_start + 9
        }
        
        # C mapping: Local 0..5 -> Global c_start + ...
        # Local C: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        # Global C for C_top (rows 0,1, cols 4,5,6):
        # (0,0) -> 4*4+0 = 16
        # (0,1) -> 5*4+0 = 20
        # (0,2) -> 6*4+0 = 24
        # (1,0) -> 4*4+1 = 17
        # (1,1) -> 5*4+1 = 21
        # (1,2) -> 6*4+1 = 25
        # c_start is 16 for C_top.
        # c_start is 18 for C_bot.
        
        c_map = {
            0: c_start + 0,
            1: c_start + 4,
            2: c_start + 8,
            3: c_start + 1,
            4: c_start + 5,
            5: c_start + 9
        }
        
        for r in range(11):
            global_r = rank_offset + r
            # U
            for local_a in range(4):
                global_a = a_map[local_a]
                U[global_a][global_r] += U_t_2x2_3[local_a][r]
            # V
            for local_b in range(6):
                global_b = b_map[local_b]
                V[global_b][global_r] += V_t_2x2_3[local_b][r]
            # W
            for local_c in range(6):
                global_c = c_map[local_c]
                W[global_c][global_r] += W_t_2x2_3[local_c][r]
                
    # Block 1: A00 * B_top -> C_top
    # A00 start 0. B_top start 4. C_top start 16.
    add_block(0, 4, 16, rank_head)
    
    # Block 2: A01 * B_bot -> C_top
    # A01 start 2. B_bot start 18. C_top start 16.
    add_block(2, 18, 16, rank_head + 11)
    
    # Block 3: A10 * B_top -> C_bot
    # A10 start 8. B_top start 4. C_bot start 18.
    add_block(8, 4, 18, rank_head + 22)
    
    # Block 4: A11 * B_bot -> C_bot
    # A11 start 10. B_bot start 18. C_bot start 18.
    add_block(10, 18, 18, rank_head + 33)
    
    return U, V, W

def _strassen_4x4_4x4_UVW():
    """
    Return the Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    Uses block Strassen: treat 4x4 as 2x2 blocks of 2x2 matrices.
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    block_A_coeffs = {
        (0,0): [1, 0, 1, 0, 1, 0, 0],
        (0,1): [0, 0, 0, 0, 1, 0, 1],
        (1,0): [0, 1, 0, 0, 0, -1, 0],
        (1,1): [1, 1, 0, 1, 0, -1, 0],
    }
    
    block_B_coeffs = {
        (0,0): [1, 1, 0, 0, 0, 1, 0],
        (0,1): [0, 0, 1, 0, 0, 1, 0],
        (1,0): [0, 0, 0, 1, 0, 0, 1],
        (1,1): [1, -1, 0, 0, 1, 0, 1],
    }
    
    block_C_coeffs = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    for i in range(4):
        for j in range(4):
            bi, bj = i // 2, j // 2
            ii, jj = i % 2, j % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    U[i*4+j][combined_r] = block_A_coeffs[(bi,bj)][rb] * U2[inner_idx][ri]
    
    for j in range(4):
        for k in range(4):
            bi, bj = j // 2, k // 2
            ii, jj = j % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    V[j*4+k][combined_r] = block_B_coeffs[(bi,bj)][rb] * V2[inner_idx][ri]
    
    for i in range(4):
        for k in range(4):
            bi, bj = i // 2, k // 2
            ii, jj = i % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    W[k*4+i][combined_r] = block_C_coeffs[(bi,bj)][rb] * W2[inner_idx][ri]
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    Returns True if valid, False otherwise.
    """
    try:
        np.random.seed(12345)
        
        for trial in range(5):
            A = np.random.randint(-10, 10, (n, m), dtype=float)
            B = np.random.randint(-10, 10, (m, p), dtype=float)
            C_expected = A @ B
            
            rank = len(U[0])
            C_computed = np.zeros((n, p), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_val = W[k * n + i]
                        for r in range(rank):
                            s += u_row[r] * v_row[r] * w_val[r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    except Exception:
        return False

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # Try the optimized decomposition (Rank 93)
    try:
        U, V, W = _construct_4x4_4x7_UVW()
        if _verify_decomposition(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        print(f"Optimized construction failed: {e}")
        
    # Fallback to trivial (Rank 112)
    U, V, W = _trivial_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
