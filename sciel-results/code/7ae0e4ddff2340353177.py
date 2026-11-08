"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 28 for 2x4x4 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 7-term local matrices (2x2x2).
    # Local U index: i_loc*2 + j_loc, i_loc,j_loc in {0,1} -> indices 0,1,2,3
    # Local V index: j_loc*2 + k_loc, j_loc,k_loc in {0,1} -> indices 0,1,2,3
    # Local W index: k_loc*2 + i_loc, k_loc,i_loc in {0,1} -> indices 0,1,2,3
    
    # Term 0: (A00 + A11)(B00 + B11)
    U0 = [1, 0, 0, 1]
    V0 = [1, 0, 0, 1]
    W0 = [1, 0, 0, 1]
    
    # Term 1: (A10 + A11)B00
    U1 = [0, 0, 1, 1]
    V1 = [1, 0, 0, 0]
    W1 = [1, 0, 0, 0]
    
    # Term 2: A00(B10 - B11)
    U2 = [1, 0, 0, 0]
    V2 = [0, 0, 1, -1]
    W2 = [0, 1, 0, 0]
    
    # Term 3: A11(B00 - B01)
    U3 = [0, 0, 0, 1]
    V3 = [1, -1, 0, 0]
    W3 = [0, 0, 0, 1]
    
    # Term 4: (A00 + A01)B11
    U4 = [1, 1, 0, 0]
    V4 = [0, 0, 0, 1]
    W4 = [0, 0, 1, 0]
    
    # Term 5: (A10 - A00)(B01 + B11)
    U5 = [-1, 0, 1, 0]
    V5 = [0, 1, 0, 1]
    W5 = [0, 1, 0, 0]
    
    # Term 6: (A01 - A11)(B10 + B00)
    U6 = [0, 1, 0, -1]
    V6 = [1, 0, 1, 0]
    W6 = [0, 0, 0, 1]
    
    # Four subproblems: (j_off, k_off)
    subproblems = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    for sub, (j_off, k_off) in enumerate(subproblems):
        for s in range(7):
            col = sub * 7 + s
            U_vals = [U0[s], U1[s], U2[s], U3[s], U4[s], U5[s], U6[s]][s]
            V_vals = [V0[s], V1[s], V2[s], V3[s], V4[s], V5[s], V6[s]][s]
            W_vals = [W0[s], W1[s], W2[s], W3[s], W4[s], W5[s], W6[s]][s]
            
            # Fill U: global index i*4 + j, local index i_loc*2 + j_loc
            for i_loc in range(2):
                for j_loc in range(2):
                    local_idx = i_loc * 2 + j_loc
                    global_idx = i_loc * 4 + (j_off + j_loc)
                    U[global_idx][col] = U_vals if local_idx == 0 else U[global_idx][col]
            
            # This approach is getting messy. Let me redo it cleanly.
            pass
    
    # Cleaner approach:
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_terms = [
        [1, 0, 0, 1],   # term 0
        [0, 0, 1, 1],   # term 1
        [1, 0, 0, 0],   # term 2
        [0, 0, 0, 1],   # term 3
        [1, 1, 0, 0],   # term 4
        [-1, 0, 1, 0],  # term 5
        [0, 1, 0, -1]   # term 6
    ]
    V_terms = [
        [1, 0, 0, 1],   # term 0
        [1, 0, 0, 0],   # term 1
        [0, 0, 1, -1],  # term 2
        [1, -1, 0, 0],  # term 3
        [0, 0, 0, 1],   # term 4
        [0, 1, 0, 1],   # term 5
        [1, 0, 1, 0]    # term 6
    ]
    W_terms = [
        [1, 0, 0, 1],   # term 0
        [1, 0, 0, 0],   # term 1
        [0, 1, 0, 0],   # term 2
        [0, 0, 0, 1],   # term 3
        [0, 0, 1, 0],   # term 4
        [0, 1, 0, 0],   # term 5
        [0, 0, 0, 1]    # term 6
    ]
    
    for sub, (j_off, k_off) in enumerate(subproblems):
        for s in range(7):
            col = sub * 7 + s
            # U: local index i_loc*2+j_loc -> global i_loc*4 + j_off+j_loc
            for i_loc in range(2):
                for j_loc in range(2):
                    local_idx = i_loc * 2 + j_loc
                    global_idx = i_loc * 4 + (j_off + j_loc)
                    U[global_idx][col] = U_terms[s][local_idx]
            # V: local index j_loc*2+k_loc -> global (j_off+j_loc)*4 + k_off+k_loc
            for j_loc in range(2):
                for k_loc in range(2):
                    local_idx = j_loc * 2 + k_loc
                    global_idx = (j_off + j_loc) * 4 + (k_off + k_loc)
                    V[global_idx][col] = V_terms[s][local_idx]
            # W: local index k_loc*2+i_loc -> global (k_off+k_loc)*2 + i_loc
            for k_loc in range(2):
                for i_loc in range(2):
                    local_idx = k_loc * 2 + i_loc
                    global_idx = (k_off + k_loc) * 2 + i_loc
                    W[global_idx][col] = W_terms[s][local_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
