"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct the decomposition for M_{2,4,6} as 6 copies of M_{2,2,2}
    # M_{2,4,6} = M_{2,2,6} \oplus M_{2,2,6}
    # M_{2,2,6} = M_{2,2,4} \oplus M_{2,2,2}
    # M_{2,2,4} = M_{2,2,2} \oplus M_{2,2,2}
    # So total 6 blocks of M_{2,2,2}.
    
    rank_block = 7
    total_rank = 6 * rank_block
    
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 42
    V = [[0] * total_rank for _ in range(m * p)] # 24 x 42
    W = [[0] * total_rank for _ in range(p * n)] # 12 x 42
    
    # Define Strassen tensors for 2x2x2
    # Indices: i,j in 0..1, k in 0..1
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # r=0..6
    # M1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=0, (1,0)=1, (1,1)=1  -> C00, C01, C10, C11
    # C00 = M1... C11 = M1...
    
    # Let's build the 7x3 matrices for U,V,W for one block
    u_block = [[0]*rank_block for _ in range(4)]
    v_block = [[0]*rank_block for _ in range(4)]
    w_block = [[0]*rank_block for _ in range(4)]
    
    # r=0 (M1)
    u_block[0][0] = 1
    u_block[3][0] = 1
    v_block[0][0] = 1
    v_block[3][0] = 1
    w_block[0][0] = 1 # C00
    w_block[2][0] = 1 # C10
    w_block[3][0] = 1 # C11
    # w_block[1][0] = 0 # C01
    
    # r=1 (M2): (A10+A11)B00
    u_block[2][1] = 1
    u_block[3][1] = 1
    v_block[0][1] = 1
    # v_block[1][1] = 0
    # v_block[2][1] = 0
    # v_block[3][1] = 0
    w_block[2][1] = 1 # C10
    w_block[3][1] = -1 # C11
    
    # r=2 (M3): A00(B10-B11)
    u_block[0][2] = 1
    v_block[2][2] = 1
    v_block[3][2] = -1
    w_block[1][2] = 1 # C01
    w_block[3][2] = 1 # C11
    
    # r=3 (M4): A11(B01-B00)
    u_block[3][3] = 1
    v_block[1][3] = 1
    v_block[0][3] = -1
    w_block[0][3] = 1 # C00
    w_block[2][3] = 1 # C10
    
    # r=4 (M5): (A00+A01)B11
    u_block[0][4] = 1
    u_block[1][4] = 1
    v_block[3][4] = 1
    w_block[0][4] = -1 # C00
    w_block[1][4] = 1 # C01
    
    # r=5 (M6): (A10-A00)(B00+B01)
    u_block[2][5] = 1
    u_block[0][5] = -1
    v_block[0][5] = 1
    v_block[1][5] = 1
    w_block[3][5] = 1 # C11
    
    # r=6 (M7): (A01-A11)(B10+B11)
    u_block[1][6] = 1
    u_block[3][6] = -1
    v_block[2][6] = 1
    v_block[3][6] = 1
    w_block[0][6] = 1 # C00
    # w_block[1][6] = 0
    # w_block[2][6] = 0
    # w_block[3][6] = 0
    
    # Now map these 6 blocks into the full 2x4x6 space.
    # Block 1: A rows 0-1, B cols 0-1 (j=0-1), C cols 0-1 (k=0-1)?
    # No, the decomposition is:
    # M_{2,4,6} splits middle dim 4 into two 2s.
    # Block 1: j in 0..1.
    # Block 2: j in 2..3.
    # Each block is M_{2,2,6}.
    # M_{2,2,6} splits last dim 6 into 2,2,2.
    # Sub-block 1: k in 0..1.
    # Sub-block 2: k in 2..3.
    # Sub-block 3: k in 4..5.
    
    # So we have 2 (j-splits) * 3 (k-splits) = 6 blocks of M_{2,2,2}.
    
    block_id = 0
    for j_offset in [0, 2]:
        for k_offset in [0, 2, 4]:
            # Map block_id to columns in U,V,W
            col_start = block_id * rank_block
            
            # U indices: i in 0..1, j in j_offset..j_offset+1
            for i in range(2):
                for j_local in range(2):
                    u_idx = i * 4 + (j_offset + j_local)
                    for r in range(rank_block):
                        U[u_idx][col_start + r] = u_block[i * 2 + j_local][r]
                        
            # V indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1
            for j_local in range(2):
                for k_local in range(2):
                    v_idx = (j_offset + j_local) * 6 + (k_offset + k_local)
                    for r in range(rank_block):
                        V[v_idx][col_start + r] = v_block[j_local * 2 + k_local][r]
                        
            # W indices: k in k_offset..k_offset+1, i in 0..1
            for k_local in range(2):
                for i in range(2):
                    w_idx = (k_offset + k_local) * 2 + i
                    for r in range(rank_block):
                        W[w_idx][col_start + r] = w_block[k_local * 2 + i][r]
                        
            block_id += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
