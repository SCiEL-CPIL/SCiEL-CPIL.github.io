import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 4, 5, 6

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        U_cols.append(u_sparse)
        V_cols.append(v_sparse)
        W_cols.append(w_sparse)

    def schoolbook_block(i_off, j_off, k_off, i_size, j_size, k_size):
        for i in range(i_size):
            for j in range(j_size):
                for k in range(k_size):
                    add_term(
                        {((i_off + i) * m + (j_off + j)): 1},
                        {((j_off + j) * p + (k_off + k)): 1},
                        {((k_off + k) * n + (i_off + i)): 1}
                    )

    def laderman_block(i_off, j_off, k_off):
        # Verify Laderman correctness
        # A is 3x3, B is 3x3, C is 3x3
        # Local indices 0..2
        # Global A: (i_off+r)*m + (j_off+c)
        # Global B: (j_off+r)*p + (k_off+c)
        # Global C: (k_off+c)*n + (i_off+r)
        
        # Laderman terms
        # M1 = (a00+a10+a11)(b00+b10+b11)
        u1 = {(i_off*3+0)*m + (j_off+0): 1, (i_off+1)*m + (j_off+0): 1, (i_off+1)*m + (j_off+1): 1}
        v1 = {(j_off+0)*p + (k_off+0): 1, (j_off+1)*p + (k_off+0): 1, (j_off+1)*p + (k_off+1): 1}
        w1 = {(k_off+0)*n + (i_off+0): 1, (k_off+0)*n + (i_off+1): 1, (k_off+0)*n + (i_off+2): 1, (k_off+1)*n + (i_off+0): 1, (k_off+1)*n + (i_off+1): 1, (k_off+1)*n + (i_off+2): 1, (k_off+2)*n + (i_off+0): 1, (k_off+2)*n + (i_off+1): 1, (k_off+2)*n + (i_off+2): 1}
        # Wait, W is sparse. We need to add terms for each output element.
        # C00 = M1 - M2 - M3 + M4
        # So for C00, we add M1 with coeff 1, M2 with -1, etc.
        
        # It's easier to define the 7 products and then add them to the output with coefficients.
        
        # Product 1
        u1 = {(i_off*3+0)*m + (j_off+0): 1, (i_off+1)*m + (j_off+0): 1, (i_off+1)*m + (j_off+1): 1}
        v1 = {(j_off+0)*p + (k_off+0): 1, (j_off+1)*p + (k_off+0): 1, (j_off+1)*p + (k_off+1): 1}
        # C00 += M1
        add_term({(i_off*3+0)*m + (j_off+0): 1, (i_off+1)*m + (j_off+0): 1, (i_off+1)*m + (j_off+1): 1}, 
                 {(j_off+0)*p + (k_off+0): 1, (j_off+1)*p + (k_off+0): 1, (j_off+1)*p + (k_off+1): 1}, 
                 {(k_off+0)*n + (i_off+0): 1})
        
        # This is getting messy. Let's just use schoolbook for safety and simplicity.
        # Rank 120 is the baseline. Any improvement is good.
        # I will use the 3x3 Laderman partition with schoolbook fallback.
        
        # For now, I will just use schoolbook for all blocks to ensure correctness.
        # The goal is to beat 120.
        # I will use the 3x3 Laderman partition.
        # If Laderman fails verification, I use schoolbook.
        
        # Let's just use schoolbook for the 3x3 blocks too, but with the partition.
        # This gives rank 120.
        # I need to use Laderman.
        
        # I will implement Laderman carefully.
        
        # M1
        u1 = {(i_off*3+0)*m + (j_off+0): 1, (i_off+1)*m + (j_off+0): 1, (i_off+1)*m + (j_off+1): 1}
        v1 = {(j_off+0)*p + (k_off+0): 1, (j_off+1)*p + (k_off+0): 1, (j_off+1)*p + (k_off+1): 1}
        # C00 += M1
        add_term(u1, v1, {(k_off+0)*n + (i_off+0): 1})
        
        # M2
        u2 = {(i_off*3+0)*m + (j_off+1): 1, (i_off*3+0)*m + (j_off+2): 1, (i_off+1)*m + (j_off+2): 1}
        v2 = {(j_off+0)*p + (k_off+1): 1, (j_off+1)*p + (k_off+2): 1}
        # C00 -= M2
        add_term({k: -v for k, v in u2.items()}, {k: -v for k, v in v2.items()}, {(k_off+0)*n + (i_off+0): 1})
        
        # M3
        u3 = {(i_off*3+0)*m + (j_off+0): 1, (i_off*3+0)*m + (j_off+1): 1, (i_off+1)*m + (j_off+1): 1}
        v3 = {(j_off+0)*p + (k_off+0): 1, (j_off+0)*p + (k_off+1): 1, (j_off+1)*p + (k_off+1): 1}
        # C00 -= M3
        add_term({k: -v for k, v in u3.items()}, {k: -v for k, v in v3.items()}, {(k_off+0)*n + (i_off+0): 1})
        
        # M4
        u4 = {(i_off*3+0)*m + (j_off+2): 1, (i_off+1)*m + (j_off+2): 1, (i_off+2)*m + (j_off+2): 1}
        v4 = {(j_off+0)*p + (k_off+2): 1, (j_off+1)*p + (k_off+2): 1, (j_off+2)*p + (k_off+2): 1}
        # C00 += M4
        add_term(u4, v4, {(k_off+0)*n + (i_off+0): 1})
        
        # C01 = M2 - M4
        add_term(u2, v2, {(k_off+1)*n + (i_off+0): 1})
        add_term({k: -v for k, v in u4.items()}, {k: -v for k, v in v4.items()}, {(k_off+1)*n + (i_off+0): 1})
        
        # C02 = M4
        add_term(u4, v4, {(k_off+2)*n + (i_off+0): 1})
        
        # C10 = M3 - M1
        add_term(u3, v3, {(k_off+0)*n + (i_off+1): 1})
        add_term({k: -v for k, v in u1.items()}, {k: -v for k, v in v1.items()}, {(k_off+0)*n + (i_off+1): 1})
        
        # C11 = M1 - M3 - M6 + M7
        # M6
        u6 = {(i_off*3+0)*m + (j_off+1): 1, (i_off+1)*m + (j_off+1): 1, (i_off+2)*m + (j_off+1): 1}
        v6 = {(j_off+0)*p + (k_off+1): 1, (j_off+1)*p + (k_off+1): 1, (j_off+2)*p + (k_off+1): 1}
        # M7
        u7 = {(i_off+1)*m + (j_off+0): 1, (i_off+2)*m + (j_off+0): 1, (i_off+2)*m + (j_off+1): 1}
        v7 = {(j_off+0)*p + (k_off+0): 1, (j_off+1)*p + (k_off+0): 1, (j_off+2)*p + (k_off+1): 1}
        
        add_term(u1, v1, {(k_off+1)*n + (i_off+1): 1})
        add_term({k: -v for k, v in u3.items()}, {k: -v for k, v in v3.items()}, {(k_off+1)*n + (i_off+1): 1})
        add_term({k: -v for k, v in u6.items()}, {k: -v for k, v in v6.items()}, {(k_off+1)*n + (i_off+1): 1})
        add_term(u7, v7, {(k_off+1)*n + (i_off+1): 1})
        
        # C12 = M6 - M7
        add_term(u6, v6, {(k_off+2)*n + (i_off+1): 1})
        add_term({k: -v for k, v in u7.items()}, {k: -v for k, v in v7.items()}, {(k_off+2)*n + (i_off+1): 1})
        
        # C20 = M5 - M3
        # M5
        u5 = {(i_off*3+0)*m + (j_off+0): 1, (i_off+1)*m + (j_off+0): 1, (i_off+2)*m + (j_off+0): 1}
        v5 = {(j_off+0)*p + (k_off+0): 1, (j_off+1)*p + (k_off+0): 1, (j_off+2)*p + (k_off+0): 1}
        
        add_term(u5, v5, {(k_off+0)*n + (i_off+2): 1})
        add_term({k: -v for k, v in u3.items()}, {k: -v for k, v in v3.items()}, {(k_off+0)*n + (i_off+2): 1})
        
        # C21 = M3 - M5 - M7 + M1
        add_term(u3, v3, {(k_off+1)*n + (i_off+2): 1})
        add_term({k: -v for k, v in u5.items()}, {k: -v for k, v in v5.items()}, {(k_off+1)*n + (i_off+2): 1})
        add_term({k: -v for k, v in u7.items()}, {k: -v for k, v in v7.items()}, {(k_off+1)*n + (i_off+2): 1})
        add_term(u1, v1, {(k_off+1)*n + (i_off+2): 1})
        
        # C22 = M5 - M7
        add_term(u5, v5, {(k_off+2)*n + (i_off+2): 1})
        add_term({k: -v for k, v in u7.items()}, {k: -v for k, v in v7.items()}, {(k_off+2)*n + (i_off+2): 1})

    # Partition:
    # A: 4x5, B: 5x6, C: 4x6
    # Split A rows: 3, 1
    # Split A cols: 3, 2
    # Split B rows: 3, 2
    # Split B cols: 3, 3
    # Split C rows: 3, 1
    # Split C cols: 3, 3
    
    # Blocks:
    # (0,0,0): 3x3 x 3x3 -> 3x3 (Laderman)
    laderman_block(0, 0, 0)
    # (0,0,1): 3x3 x 3x3 -> 3x3 (Laderman)
    laderman_block(0, 0, 3)
    # (0,1,0): 3x2 x 2x3 -> 3x3 (Schoolbook)
    schoolbook_block(0, 3, 0, 3, 2, 3)
    # (0,1,1): 3x2 x 2x3 -> 3x3 (Schoolbook)
    schoolbook_block(0, 3, 3, 3, 2, 3)
    # (1,0,0): 1x3 x 3x3 -> 1x3 (Schoolbook)
    schoolbook_block(3, 0, 0, 1, 3, 3)
    # (1,0,1): 1x3 x 3x3 -> 1x3 (Schoolbook)
    schoolbook_block(3, 0, 3, 1, 3, 3)
    # (1,1,0): 1x2 x 2x3 -> 1x3 (Schoolbook)
    schoolbook_block(3, 3, 0, 1, 2, 3)
    # (1,1,1): 1x2 x 2x3 -> 1x3 (Schoolbook)
    schoolbook_block(3, 3, 3, 1, 2, 3)

    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
