"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Indices in local 2x2x2 block
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # 1. M1
        u_vec = [1, 0, 0, 1]
        v_vec = [1, 0, 0, 1]
        w_vec = [1, 0, 0, 1]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1
        
        # 2. M2
        u_vec = [0, 0, 1, 1]
        v_vec = [1, 0, 0, 0]
        w_vec = [0, 0, 1, -1]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1
        
        # 3. M3
        u_vec = [1, 0, 0, 0]
        v_vec = [0, 1, 0, -1]
        w_vec = [0, 1, 0, 1]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1
        
        # 4. M4
        u_vec = [0, 0, 0, 1]
        v_vec = [-1, 0, 1, 0]
        w_vec = [1, 0, 1, 0]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1
        
        # 5. M5
        u_vec = [1, 1, 0, 0]
        v_vec = [0, 0, 0, 1]
        w_vec = [-1, 1, 0, 0]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1
        
        # 6. M6
        u_vec = [-1, 0, 1, 0]
        v_vec = [1, 1, 0, 0]
        w_vec = [0, 0, 0, 1]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1
        
        # 7. M7
        u_vec = [0, 1, 0, -1]
        v_vec = [0, 0, 1, 1]
        w_vec = [1, 0, 0, 0]
        for l in range(4):
            if u_vec[l]: U[(i_off + l//2) * m + (j_off + l%2)][col] += u_vec[l]
            if v_vec[l]: V[(j_off + l//2) * p + (k_off + l%2)][col] += v_vec[l]
            if w_vec[l]: W[(k_off + l//2) * n + (i_off + l%2)][col] += w_vec[l]
        col += 1

    # Helper to add a 2xMxP schoolbook block
    def add_schoolbook_2MP(i_off, j_off, k_off, m_b, p_b):
        nonlocal col
        for j in range(m_b):
            for k in range(p_b):
                U[(i_off + 0) * m + (j_off + j)][col] = 1
                U[(i_off + 1) * m + (j_off + j)][col] = 1
                V[(j_off + j) * p + (k_off + k)][col] = 1
                W[(k_off + k) * n + (i_off + 0)][col] = 1
                W[(k_off + k) * n + (i_off + 1)][col] = 1
                col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_222(0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_222(0, 2, 0)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_222(0, 0, 2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_222(0, 2, 2)
    
    # Block 5: A[0:2, 4] * B[4, 0:5]
    # This is 2x1x5
    add_schoolbook_2MP(0, 4, 0, 1, 5)
    
    # Block 6: A[0:2, 0:4] * B[0:4, 4]
    # This is 2x4x1
    add_schoolbook_2MP(0, 0, 4, 4, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
