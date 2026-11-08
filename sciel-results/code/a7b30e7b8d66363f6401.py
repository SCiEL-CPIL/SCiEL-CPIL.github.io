import numpy as np

def construct():
    n, m, p = 3, 4, 5
    
    # Total rank 40
    rank = 40
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a block
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk, offset):
        """
        u_idx, v_idx, w_idx: lists of global indices corresponding to local indices 0..len-1
        U_blk, V_blk, W_blk: block factor matrices
        offset: starting rank index
        """
        r_count = U_blk.shape[1]
        for r in range(r_count):
            for i, u_g in enumerate(u_idx):
                U[u_g, offset + r] = U_blk[i, r]
            for i, v_g in enumerate(v_idx):
                V[v_g, offset + r] = V_blk[i, r]
            for i, w_g in enumerate(w_idx):
                W[w_g, offset + r] = W_blk[i, r]
        return offset + r_count

    # --- Block 1: A[0, 0:2] * B[0:2, :] (1x2x5) ---
    # U indices: 0, 1 (A[0,0], A[0,1])
    # V indices: 0..4 (B[0,0..4]), 5..9 (B[1,0..4])
    # W indices: k*3 + 0 for k=0..4 (C[0,k])
    
    # Rank 2
    U_blk = np.zeros((2, 2))
    V_blk = np.zeros((10, 2))
    W_blk = np.zeros((5, 2))
    
    # r=0: A[0,0] * B[0,k]
    U_blk[0, 0] = 1
    V_blk[0:5, 0] = 1 # B[0,0..4]
    W_blk[:, 0] = 1   # C[0,0..4]
    
    # r=1: A[0,1] * B[1,k]
    U_blk[1, 1] = 1
    V_blk[5:10, 1] = 1 # B[1,0..4]
    W_blk[:, 1] = 1   # C[0,0..4]
    
    u_idx = [0, 1]
    v_idx = list(range(0, 10))
    w_idx = [k * 3 + 0 for k in range(5)]
    
    offset = add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk, 0)

    # --- Block 2: A[1:3, 0:2] * B[0:2, :] (2x2x5) ---
    # U indices: 
    # A[1,0]=4, A[1,1]=5
    # A[2,0]=8, A[2,1]=9
    # Local order: 0:4, 1:5, 2:8, 3:9
    u_idx = [4, 5, 8, 9]
    
    # V indices: 0..9 (B[0:2, 0..4])
    # Split into 2x2 (cols 0,1), 2x2 (cols 2,3), 2x1 (col 4)
    v_idx_full = list(range(0, 10))
    
    # W indices:
    # C[1,k] -> k*3+1
    # C[2,k] -> k*3+2
    # Local order for 2x2x2: C[1,0], C[1,1], C[2,0], C[2,1]
    # Global: 1, 4, 2, 5 for cols 0,1
    #         7, 10, 8, 11 for cols 2,3
    
    # Sub-block 2a: Strassen 2x2x2 (cols 0,1)
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    # M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1
    # M3
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[1, 2] = 1; W_strassen[3, 2] = 1
    # M4
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[2, 3] = 1
    # M5
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[1, 4] = 1
    # M6
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1
    # M7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1
    
    # V indices for 2x2x2 (cols 0,1):
    # B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    v_idx_2a = [0, 1, 5, 6]
    # W indices for 2x2x2 (cols 0,1):
    # C[1,0]=1, C[1,1]=4, C[2,0]=2, C[2,1]=5
    w_idx_2a = [1, 4, 2, 5]
    
    offset = add_block(u_idx, v_idx_2a, w_idx_2a, U_strassen, V_strassen, W_strassen, offset)
    
    # Sub-block 2b: Strassen 2x2x2 (cols 2,3)
    # V indices: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    v_idx_2b = [2, 3, 7, 8]
    # W indices: C[1,2]=7, C[1,3]=10, C[2,2]=8, C[2,3]=11
    w_idx_2b = [7, 10, 8, 11]
    
    offset = add_block(u_idx, v_idx_2b, w_idx_2b, U_strassen, V_strassen, W_strassen, offset)
    
    # Sub-block 2c: Schoolbook 2x2x1 (col 4)
    U_school = np.zeros((4, 4))
    V_school = np.zeros((2, 4))
    W_school = np.zeros((2, 4))
    
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # C[2,4] = A[2,0]B[0,4] + A[2,1]B[1,4]
    
    # r=0: A[1,0]B[0,4] -> C[1,4]
    U_school[0, 0] = 1; V_school[0, 0] = 1; W_school[0, 0] = 1
    # r=1: A[1,1]B[1,4] -> C[1,4]
    U_school[1, 1] = 1; V_school[1, 1] = 1; W_school[0, 1] = 1
    # r=2: A[2,0]B[0,4] -> C[2,4]
    U_school[2, 2] = 1; V_school[0, 2] = 1; W_school[1, 2] = 1
    # r=3: A[2,1]B[1,4] -> C[2,4]
    U_school[3, 3] = 1; V_school[1, 3] = 1; W_school[1, 3] = 1
    
    # V indices: B[0,4]=4, B[1,4]=9
    v_idx_2c = [4, 9]
    # W indices: C[1,4]=13, C[2,4]=14
    w_idx_2c = [13, 14]
    
    offset = add_block(u_idx, v_idx_2c, w_idx_2c, U_school, V_school, W_school, offset)

    # --- Block 3: A[0, 2:4] * B[2:4, :] (1x2x5) ---
    # U indices: A[0,2]=2, A[0,3]=3
    u_idx = [2, 3]
    
    # V indices: B[2,0..4]=10..14, B[3,0..4]=15..19
    v_idx = list(range(10, 20))
    
    # W indices: C[0,k] -> k*3+0
    w_idx = [k * 3 + 0 for k in range(5)]
    
    # Rank 2
    U_blk = np.zeros((2, 2))
    V_blk = np.zeros((10, 2))
    W_blk = np.zeros((5, 2))
    
    # r=0: A[0,2] * B[2,k]
    U_blk[0, 0] = 1
    V_blk[0:5, 0] = 1 # B[2,0..4]
    W_blk[:, 0] = 1   # C[0,0..4]
    
    # r=1: A[0,3] * B[3,k]
    U_blk[1, 1] = 1
    V_blk[5:10, 1] = 1 # B[3,0..4]
    W_blk[:, 1] = 1   # C[0,0..4]
    
    offset = add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk, offset)

    # --- Block 4: A[1:3, 2:4] * B[2:4, :] (2x2x5) ---
    # U indices:
    # A[1,2]=6, A[1,3]=7
    # A[2,2]=10, A[2,3]=11
    u_idx = [6, 7, 10, 11]
    
    # V indices: 10..19 (B[2:4, 0..4])
    
    # W indices:
    # C[1,k] -> k*3+1
    # C[2,k] -> k*3+2
    
    # Sub-block 4a: Strassen 2x2x2 (cols 0,1)
    # V indices: B[2,0]=10, B[2,1]=11, B[3,0]=15, B[3,1]=16
    v_idx_4a = [10, 11, 15, 16]
    # W indices: C[1,0]=1, C[1,1]=4, C[2,0]=2, C[2,1]=5
    w_idx_4a = [1, 4, 2, 5]
    
    offset = add_block(u_idx, v_idx_4a, w_idx_4a, U_strassen, V_strassen, W_strassen, offset)
    
    # Sub-block 4b: Strassen 2x2x2 (cols 2,3)
    # V indices: B[2,2]=12, B[2,3]=13, B[3,2]=17, B[3,3]=18
    v_idx_4b = [12, 13, 17, 18]
    # W indices: C[1,2]=7, C[1,3]=10, C[2,2]=8, C[2,3]=11
    w_idx_4b = [7, 10, 8, 11]
    
    offset = add_block(u_idx, v_idx_4b, w_idx_4b, U_strassen, V_strassen, W_strassen, offset)
    
    # Sub-block 4c: Schoolbook 2x2x1 (col 4)
    # V indices: B[2,4]=14, B[3,4]=19
    v_idx_4c = [14, 19]
    # W indices: C[1,4]=13, C[2,4]=14
    w_idx_4c = [13, 14]
    
    offset = add_block(u_idx, v_idx_4c, w_idx_4c, U_school, V_school, W_school, offset)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
