"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target dimensions
    n, m, p = DIMENSIONS
    
    # We use a decomposition for 2x4x8 with rank 28
    # and restrict it to 2x3x7.
    
    n_ext = 2
    m_ext = 4
    p_ext = 8
    
    # Strassen decomposition for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We have 4 blocks of 2x2x2
    # Block 0: A1 B11 -> C11 (part of C1:4)
    # Block 1: A2 B21 -> C11 (part of C1:4)
    # Block 2: A1 B12 -> C12 (part of C5:8)
    # Block 3: A2 B22 -> C12 (part of C5:8)
    
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n_ext * m_ext)]
    V = [[0] * rank for _ in range(m_ext * p_ext)]
    W = [[0] * rank for _ in range(p_ext * n_ext)]
    
    # Helper to add a Strassen block
    def add_strassen_block(offset_u, offset_v, offset_w, block_idx):
        # block_idx determines which A, B blocks are used
        # A blocks: 0 (cols 0,1), 1 (cols 2,3)
        # B blocks: 0 (rows 0,1, cols 0,1), 1 (rows 0,1, cols 2,3), 2 (rows 2,3, cols 0,1), 3 (rows 2,3, cols 2,3)
        
        a_block = 0 if block_idx in [0, 2] else 1
        b_block = block_idx
        
        # A indices in global U
        # A is 2x4. U row index = i * 4 + j
        # A1 (block 0): i in {0,1}, j in {0,1}
        # A2 (block 1): i in {0,1}, j in {2,3}
        
        # B indices in global V
        # B is 4x8. V row index = j * 8 + k
        # B11 (block 0): j in {0,1}, k in {0,1}
        # B12 (block 1): j in {0,1}, k in {2,3}
        # B21 (block 2): j in {2,3}, k in {0,1}
        # B22 (block 3): j in {2,3}, k in {2,3}
        
        # C indices in global W
        # C is 2x8. W row index = k * 2 + i
        # C1:4 (blocks 0,1): k in {0..3}
        # C5:8 (blocks 2,3): k in {4..7}
        # Specifically:
        # Block 0,1 contribute to k in {0..3}
        # Block 2,3 contribute to k in {4..7}
        
        c_k_offset = 0 if block_idx in [0, 1] else 4
        
        # Strassen terms
        # 1: (A11 + A22)(B11 + B22)
        # 2: (A21 + A22) B11
        # 3: A11 (B12 - B22)
        # 4: A22 (B21 - B11)
        # 5: (A11 + A12) B22
        # 6: (A21 - A11)(B11 + B12)
        # 7: (A12 - A22)(B21 + B22)
        
        # Local indices in 2x2 block:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # Global A index: i * 4 + (a_block*2 + j_local)
        # Global B index: (b_block_row_offset + j_local) * 8 + (b_block_col_offset + k_local)
        # Global C index: (c_k_offset + k_local) * 2 + i
        
        b_row_off = 0 if b_block in [0, 1] else 2
        b_col_off = 0 if b_block in [0, 2] else 2
        
        terms = [
            # (coeff_A, coeff_B, coeff_C, sign_A, sign_B, sign_C)
            # We need to map to U, V, W
            # U[i, col], V[j, col], W[k, col]
            
            # Term 1: (A00 + A11)(B00 + B11)
            # U: A00 (+), A11 (+)
            # V: B00 (+), B11 (+)
            # W: C00 (+), C01 (-), C10 (-), C11 (+) -> Wait, Strassen C formulas:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for M1:
            # C00: +1, C01: 0, C10: 0, C11: +1
            # Wait, M1 is in C00 and C11.
            # C00 += M1
            # C11 += M1
            # So W: C00 (+1), C11 (+1)
            
            # Let's define the contributions to W for each M
            # M1: C00(+), C11(+)
            # M2: C10(+), C11(-)
            # M3: C01(+), C11(+)
            # M4: C00(+), C10(+)
            # M5: C00(-), C01(+)
            # M6: C11(+)
            # M7: C00(+)
            
            # Re-check Strassen:
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
            
            # Correct.
        ]
        
        # We will manually assign the 7 columns for this block
        base_col = block_idx * 7
        
        # M1
        col = base_col + 0
        # A: A00, A11
        u_i00 = 0 * m_ext + (a_block * 2 + 0)
        u_i11 = 1 * m_ext + (a_block * 2 + 1)
        U[u_i00][col] = 1
        U[u_i11][col] = 1
        
        # B: B00, B11
        v_j00 = (b_row_off + 0) * p_ext + (b_col_off + 0)
        v_j11 = (b_row_off + 1) * p_ext + (b_col_off + 1)
        V[v_j00][col] = 1
        V[v_j11][col] = 1
        
        # C: C00, C11
        w_k00 = (c_k_offset + 0) * n_ext + 0
        w_k11 = (c_k_offset + 1) * n_ext + 1
        W[w_k00][col] = 1
        W[w_k11][col] = 1
        
        # M2
        col = base_col + 1
        # A: A10, A11
        u_i10 = 1 * m_ext + (a_block * 2 + 0)
        u_i11 = 1 * m_ext + (a_block * 2 + 1)
        U[u_i10][col] = 1
        U[u_i11][col] = 1
        
        # B: B00
        v_j00 = (b_row_off + 0) * p_ext + (b_col_off + 0)
        V[v_j00][col] = 1
        
        # C: C10 (+), C11 (-)
        w_k10 = (c_k_offset + 0) * n_ext + 1
        w_k11 = (c_k_offset + 1) * n_ext + 1
        W[w_k10][col] = 1
        W[w_k11][col] = -1
        
        # M3
        col = base_col + 2
        # A: A00
        u_i00 = 0 * m_ext + (a_block * 2 + 0)
        U[u_i00][col] = 1
        
        # B: B01 (+), B11 (-)
        v_j01 = (b_row_off + 0) * p_ext + (b_col_off + 1)
        v_j11 = (b_row_off + 1) * p_ext + (b_col_off + 1)
        V[v_j01][col] = 1
        V[v_j11][col] = -1
        
        # C: C01 (+), C11 (+)
        w_k01 = (c_k_offset + 1) * n_ext + 0
        w_k11 = (c_k_offset + 1) * n_ext + 1
        W[w_k01][col] = 1
        W[w_k11][col] = 1
        
        # M4
        col = base_col + 3
        # A: A11
        u_i11 = 1 * m_ext + (a_block * 2 + 1)
        U[u_i11][col] = 1
        
        # B: B10 (+), B00 (-)
        v_j10 = (b_row_off + 1) * p_ext + (b_col_off + 0)
        v_j00 = (b_row_off + 0) * p_ext + (b_col_off + 0)
        V[v_j10][col] = 1
        V[v_j00][col] = -1
        
        # C: C00 (+), C10 (+)
        w_k00 = (c_k_offset + 0) * n_ext + 0
        w_k10 = (c_k_offset + 0) * n_ext + 1
        W[w_k00][col] = 1
        W[w_k10][col] = 1
        
        # M5
        col = base_col + 4
        # A: A00 (+), A01 (+)
        u_i00 = 0 * m_ext + (a_block * 2 + 0)
        u_i01 = 0 * m_ext + (a_block * 2 + 1)
        U[u_i00][col] = 1
        U[u_i01][col] = 1
        
        # B: B11
        v_j11 = (b_row_off + 1) * p_ext + (b_col_off + 1)
        V[v_j11][col] = 1
        
        # C: C00 (-), C01 (+)
        w_k00 = (c_k_offset + 0) * n_ext + 0
        w_k01 = (c_k_offset + 1) * n_ext + 0
        W[w_k00][col] = -1
        W[w_k01][col] = 1
        
        # M6
        col = base_col + 5
        # A: A10 (-), A00 (-) -> (A10 - A00)
        u_i10 = 1 * m_ext + (a_block * 2 + 0)
        u_i00 = 0 * m_ext + (a_block * 2 + 0)
        U[u_i10][col] = 1
        U[u_i00][col] = -1
        
        # B: B00 (+), B01 (+)
        v_j00 = (b_row_off + 0) * p_ext + (b_col_off + 0)
        v_j01 = (b_row_off + 0) * p_ext + (b_col_off + 1)
        V[v_j00][col] = 1
        V[v_j01][col] = 1
        
        # C: C11 (+)
        w_k11 = (c_k_offset + 1) * n_ext + 1
        W[w_k11][col] = 1
        
        # M7
        col = base_col + 6
        # A: A01 (+), A11 (-) -> (A01 - A11)
        u_i01 = 0 * m_ext + (a_block * 2 + 1)
        u_i11 = 1 * m_ext + (a_block * 2 + 1)
        U[u_i01][col] = 1
        U[u_i11][col] = -1
        
        # B: B10 (+), B11 (+)
        v_j10 = (b_row_off + 1) * p_ext + (b_col_off + 0)
        v_j11 = (b_row_off + 1) * p_ext + (b_col_off + 1)
        V[v_j10][col] = 1
        V[v_j11][col] = 1
        
        # C: C00 (+)
        w_k00 = (c_k_offset + 0) * n_ext + 0
        W[w_k00][col] = 1

    # Add the 4 blocks
    add_strassen_block(0, 0, 0, 0) # A1 B11
    add_strassen_block(0, 0, 0, 1) # A2 B21
    add_strassen_block(0, 0, 0, 2) # A1 B12
    add_strassen_block(0, 0, 0, 3) # A2 B22
    
    # Now restrict to 2x3x7
    # Keep U rows where j < 3
    # U row index = i * 4 + j. Keep if j < 3.
    U_restricted = []
    for i in range(n_ext):
        for j in range(m_ext):
            if j < m:
                U_restricted.append(U[i * m_ext + j])
                
    # Keep V rows where j < 3 and k < 7
    # V row index = j * 8 + k. Keep if j < 3 and k < 7.
    V_restricted = []
    for j in range(m_ext):
        for k in range(p_ext):
            if j < m and k < p:
                V_restricted.append(V[j * p_ext + k])
                
    # Keep W rows where k < 7
    # W row index = k * 2 + i. Keep if k < 7.
    W_restricted = []
    for k in range(p_ext):
        for i in range(n_ext):
            if k < p:
                W_restricted.append(W[k * n_ext + i])
                
    return {"U": U_restricted, "V": V_restricted, "W": W_restricted}
# EVOLVE-BLOCK-END
