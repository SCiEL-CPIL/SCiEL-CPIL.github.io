"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # Split m=3 into m1=2, m2=1
    # A1 is 2x2, A2 is 2x1
    # B1 is 2x4, B2 is 1x4
    # C = A1 B1 + A2 B2
    # B1 split into B1L (2x2), B1R (2x2)
    # C = A1 B1L + A1 B1R + A2 B2
    
    total_rank = 7 + 7 + 8
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Helper to add Strassen block
    def add_strassen_block(u_map, v_map, w_map, offset):
        # u_map: local 0-3 to global u index
        # v_map: local 0-3 to global v index
        # w_map: local 0-3 to global w index
        
        # M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # Z00, Z11
        
        # M2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, 1] # Z10, Z11 (with sign - in Z11? No, Z10=M2+M4, Z11=M1-M2+M3+M6)
        
        # M3
        u3 = [1, 0, 0, 0]
        v3 = [0, -1, 1, 0] # Y10 - Y11? Y indices: 00, 01, 10, 11. Y10 is idx 2, Y11 is idx 3.
        # Wait, local Y indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # M3 = X00 (Y10 - Y11)
        # v3 should be 0, 0, 1, -1.
        v3 = [0, 0, 1, -1]
        w3 = [0, 1, 0, 0] # Z01 (M3+M5), Z11 (M1-M2+M3+M6)
        
        # M4
        u4 = [0, 0, 0, 1]
        v4 = [0, 1, -1, 0] # Y01 - Y00
        w4 = [1, 0, 1, 0] # Z00 (M1+M4-M5+M7), Z10 (M2+M4)
        
        # M5
        u5 = [1, 1, 0, 0] # X00 + X01
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0] # Z00 (-M5), Z01 (M5)
        
        # M6
        u6 = [0, 0, 1, -1] # X10 - X00
        v6 = [1, 0, 1, 0] # Y00 + Y10
        w6 = [0, 0, 0, 1] # Z11 (M6)
        
        # M7
        u7 = [0, 1, 0, -1] # X01 - X11
        v7 = [0, 1, 0, 1] # Y11 + Y01
        w7 = [1, 0, 0, 0] # Z00 (M7)
        
        blocks = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for i, (ub, vb, wb) in enumerate(blocks):
            col = offset + i
            for li, val in enumerate(ub):
                if val != 0:
                    U[u_map[li]][col] = val
            for li, val in enumerate(vb):
                if val != 0:
                    V[v_map[li]][col] = val
            for li, val in enumerate(wb):
                if val != 0:
                    W[w_map[li]][col] = val

    # Block 1: A1 B1L
    # A1 local 0-3 -> Global u
    # Local 0(0,0)->0, 1(0,1)->1, 2(1,0)->3, 3(1,1)->4
    u_map1 = [0, 1, 3, 4]
    # B1L local 0-3 -> Global v
    # Local 0(0,0)->0, 1(0,1)->1, 2(1,0)->4, 3(1,1)->5
    v_map1 = [0, 1, 4, 5]
    # C1L local 0-3 -> Global w
    # Local 0(i=0,k=0)->0, 1(i=0,k=1)->2, 2(i=1,k=0)->1, 3(i=1,k=1)->3
    w_map1 = [0, 2, 1, 3]
    
    add_strassen_block(u_map1, v_map1, w_map1, rank_offset)
    rank_offset += 7
    
    # Block 2: A1 B1R
    # u_map same
    u_map2 = [0, 1, 3, 4]
    # B1R local 0-3 -> Global v
    # Local 0(0,2)->2, 1(0,3)->3, 2(1,2)->6, 3(1,3)->7
    v_map2 = [2, 3, 6, 7]
    # C1R local 0-3 -> Global w
    # Local 0(i=0,k=2)->4, 1(i=0,k=3)->6, 2(i=1,k=2)->5, 3(i=1,k=3)->7
    w_map2 = [4, 6, 5, 7]
    
    add_strassen_block(u_map2, v_map2, w_map2, rank_offset)
    rank_offset += 7
    
    # Block 3: A2 B2 (Rank 8)
    # A2 indices: i=0->u=2, i=1->u=5
    # B2 indices: k=0->v=8, k=1->v=9, k=2->v=10, k=3->v=11
    # W indices: (k,i)
    for i in range(2):
        for k in range(4):
            col = rank_offset
            u_idx = i * m + 2 # j=2
            v_idx = 2 * p + k # j=2
            w_idx = k * n + i
            
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0
            rank_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
