import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 84
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper to add a decomposition
    def add_decomp(n_blk, m_blk, p_blk, u_off, v_off, w_off, decomp):
        nonlocal col_offset
        for u, v, w in decomp:
            U[u_off + u, col_offset] = u_val
            V[v_off + v, col_offset] = v_val
            W[w_off + w, col_offset] = w_val
            col_offset += 1

    # Strassen 2x2x2 decomposition
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (a-c)(e+g)
    # C00 = m1 + m3 - m7 - m4
    # C01 = m1 + m4
    # C10 = m2 + m5
    # C11 = m3 + m6 - m1 - m2
    
    # Indices for 2x2:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_decomp = [
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, -1]), # C00=1, C01=1, C10=0, C11=-1? 
        # Wait, W index is k*n + i.
        # C00: k=0, i=0 -> 0.
        # C01: k=1, i=0 -> 2.
        # C10: k=0, i=1 -> 1.
        # C11: k=1, i=1 -> 3.
        # m1 in C00: 1. In C01: 1. In C10: 0. In C11: -1.
        # So W = [1, 0, 1, -1]. Correct.
        
        # m2 = (c+d)e. C10=1, C11=-1.
        # W = [0, -1, 1, 0]? 
        # C00=0, C01=0, C10=1, C11=-1.
        # Indices: 0:0, 1:0, 2:1, 3:-1.
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        
        # m3 = a(f-h). C00=1, C01=0, C10=0, C11=1.
        # W = [1, 0, 0, 1].
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 1]),
        
        # m4 = d(g-e). C00=-1, C01=1, C10=0, C11=0.
        # W = [-1, 0, 1, 0].
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]),
        
        # m5 = (a+b)h. C00=0, C01=0, C10=1, C11=0.
        # W = [0, 0, 1, 0].
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
        
        # m6 = (c-d)(f+h). C00=0, C01=0, C10=0, C11=1.
        # W = [0, 0, 0, 1].
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        
        # m7 = (a-c)(e+g). C00=-1, C01=0, C10=0, C11=0.
        # W = [-1, 0, 0, 0].
        ([1, 0, -1, 0], [1, 0, 1, 0], [-1, 0, 0, 0]),
    ]
    
    # Trivial 2x1x2
    # A 2x1, B 1x2.
    # A00 B00 -> C00
    # A00 B01 -> C01
    # A10 B00 -> C10
    # A10 B01 -> C11
    trivial_2_1_2 = [
        ([1, 0], [1, 0], [1, 0, 0, 0]),
        ([1, 0], [0, 1], [0, 1, 0, 0]),
        ([0, 1], [1, 0], [0, 0, 1, 0]),
        ([0, 1], [0, 1], [0, 0, 0, 1]),
    ]
    
    # Trivial 1x2x2
    # A 1x2, B 2x2.
    # A00 B00 -> C00
    # A00 B01 -> C01
    # A01 B00 -> C10
    # A01 B01 -> C11
    trivial_1_2_2 = [
        ([1, 0], [1, 0, 0, 0], [1, 0, 0, 0]),
        ([1, 0], [0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([0, 1], [0, 1, 0, 0], [0, 0, 0, 1]),
    ]
    
    # Trivial 1x1x2
    # A 1x1, B 1x2.
    # A00 B00 -> C00
    # A00 B01 -> C01
    trivial_1_1_2 = [
        ([1], [1, 0], [1, 0]),
        ([1], [0, 1], [0, 1]),
    ]
    
    # Mapping of blocks to global indices
    # A is 3x5. U index = i*5 + j.
    # B is 5x6. V index = j*6 + k.
    # C is 3x6. W index = k*3 + i.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # u_off: 0*5+0=0. v_off: 0*6+0=0. w_off: 0*3+0=0.
    for u, v, w in strassen_decomp:
        for i in range(2):
            U[u_off + i, col_offset] = u[i]
            for j in range(2):
                V[v_off + j, col_offset] = v[j]
                for k in range(2):
                    W[w_off + k*2 + i, col_offset] = w[k*2 + i]
        col_offset += 1
    # This loop structure is wrong. The decomposition vectors are already full.
    
    # Let's rewrite the add logic properly.
    col_offset = 0
    def add_block(n_b, m_b, p_b, u_start, v_start, w_start, decomp):
        nonlocal col_offset
        for u_vec, v_vec, w_vec in decomp:
            for idx_u in range(len(u_vec)):
                U[u_start + idx_u, col_offset] = u_vec[idx_u]
            for idx_v in range(len(v_vec)):
                V[v_start + idx_v, col_offset] = v_vec[idx_v]
            for idx_w in range(len(w_vec)):
                W[w_start + idx_w, col_offset] = w_vec[idx_w]
            col_offset += 1
            
    # Strassen blocks
    # 1. A[0:2, 0:2] * B[0:2, 0:2]
    # u_start=0, v_start=0, w_start=0
    add_block(2, 2, 2, 0, 0, 0, strassen_decomp)
    
    # 2. A[0:2, 0:2] * B[0:2, 2:4]
    # u_start=0, v_start=2, w_start=2
    add_block(2, 2, 2, 0, 2, 2, strassen_decomp)
    
    # 3. A[0:2, 0:2] * B[0:2, 4:6]
    # u_start=0, v_start=4, w_start=4
    add_block(2, 2, 2, 0, 4, 4, strassen_decomp)
    
    # 4. A[0:2, 2:4] * B[2:4, 0:2]
    # u_start=2, v_start=12, w_start=0
    add_block(2, 2, 2, 2, 12, 0, strassen_decomp)
    
    # 5. A[0:2, 2:4] * B[2:4, 2:4]
    # u_start=2, v_start=14, w_start=2
    add_block(2, 2, 2, 2, 14, 2, strassen_decomp)
    
    # 6. A[0:2, 2:4] * B[2:4, 4:6]
    # u_start=2, v_start=16, w_start=4
    add_block(2, 2, 2, 2, 16, 4, strassen_decomp)
    
    # Trivial 2x1x2 blocks (A[0:2, 4:5] * B[4:5, 0:2])
    # 7. A[0:2, 4:5] * B[4:5, 0:2]
    # u_start=4, v_start=24, w_start=0
    add_block(2, 1, 2, 4, 24, 0, trivial_2_1_2)
    
    # 8. A[0:2, 4:5] * B[4:5, 2:4]
    # u_start=4, v_start=26, w_start=2
    add_block(2, 1, 2, 4, 26, 2, trivial_2_1_2)
    
    # 9. A[0:2, 4:5] * B[4:5, 4:6]
    # u_start=4, v_start=28, w_start=4
    add_block(2, 1, 2, 4, 28, 4, trivial_2_1_2)
    
    # Trivial 1x2x2 blocks (A[2:3, 0:2] * B[0:2, 0:2])
    # 10. A[2:3, 0:2] * B[0:2, 0:2]
    # u_start=10, v_start=0, w_start=6
    add_block(1, 2, 2, 10, 0, 6, trivial_1_2_2)
    
    # 11. A[2:3, 0:2] * B[0:2, 2:4]
    # u_start=10, v_start=2, w_start=8
    add_block(1, 2, 2, 10, 2, 8, trivial_1_2_2)
    
    # 12. A[2:3, 0:2] * B[0:2, 4:6]
    # u_start=10, v_start=4, w_start=10
    add_block(1, 2, 2, 10, 4, 10, trivial_1_2_2)
    
    # 13. A[2:3, 2:4] * B[2:4, 0:2]
    # u_start=12, v_start=12, w_start=6
    add_block(1, 2, 2, 12, 12, 6, trivial_1_2_2)
    
    # 14. A[2:3, 2:4] * B[2:4, 2:4]
    # u_start=12, v_start=14, w_start=8
    add_block(1, 2, 2, 12, 14, 8, trivial_1_2_2)
    
    # 15. A[2:3, 2:4] * B[2:4, 4:6]
    # u_start=12, v_start=16, w_start=10
    add_block(1, 2, 2, 12, 16, 10, trivial_1_2_2)
    
    # Trivial 1x1x2 blocks (A[2:3, 4:5] * B[4:5, 0:2])
    # 16. A[2:3, 4:5] * B[4:5, 0:2]
    # u_start=14, v_start=24, w_start=6
    add_block(1, 1, 2, 14, 24, 6, trivial_1_1_2)
    
    # 17. A[2:3, 4:5] * B[4:5, 2:4]
    # u_start=14, v_start=26, w_start=8
    add_block(1, 1, 2, 14, 26, 8, trivial_1_1_2)
    
    # 18. A[2:3, 4:5] * B[4:5, 4:6]
    # u_start=14, v_start=28, w_start=10
    add_block(1, 1, 2, 14, 28, 10, trivial_1_1_2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
