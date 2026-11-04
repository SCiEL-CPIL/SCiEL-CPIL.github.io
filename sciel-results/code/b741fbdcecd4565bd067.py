"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_list = []
    V_list = []
    W_list = []
    
    def append_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    def make_vec(dim, updates):
        vec = [0] * dim
        for idx, val in updates.items():
            vec[idx] = val
        return vec

    # Strassen 2x2x2 terms
    # Returns lists of (index, coeff) for U, V, W
    # U is 2x2 (4 elems), V is 2x2 (4 elems), W is 2x2 (4 elems)
    # Indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    def strassen_2x2_offsets(oi, oj, ok):
        # oi: row offset in A (and C)
        # oj: col offset in A (and row in B)
        # ok: col offset in B (and C)
        
        # 7 Strassen terms
        terms = [
            # u_dict, v_dict, w_dict
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # m1
            ({2:1, 3:1}, {0:1},    {2:1, 3:-1}),   # m2
            ({0:1},    {1:1, 3:-1},{1:1, 3:1}),    # m3
            ({3:1},    {2:1, 0:-1},{0:1, 2:1}),    # m4
            ({0:1, 1:1},{3:1},     {0:-1, 1:1}),   # m5
            ({2:1, 0:-1},{0:1, 1:1},{3:1}),        # m6
            ({1:1, 3:-1},{2:1, 3:1},{0:1})         # m7
        ]
        
        for u_d, v_d, w_d in terms:
            u_vec = [0] * 16
            v_vec = [0] * 24
            w_vec = [0] * 24
            
            # Map local 2x2 indices to global
            # U global index: (oi + r) * 4 + (oj + c)
            for loc_idx, val in u_d.items():
                r, c = divmod(loc_idx, 2)
                g_idx = (oi + r) * 4 + (oj + c)
                u_vec[g_idx] = val
                
            # V global index: (oj + r) * 6 + (ok + c)
            for loc_idx, val in v_d.items():
                r, c = divmod(loc_idx, 2)
                g_idx = (oj + r) * 6 + (ok + c)
                v_vec[g_idx] = val
                
            # W global index: (ok + c) * 4 + (oi + r)  (Trace convention: W[k*n+i, r])
            # Wait, the task says W[k*n+i, r]. 
            # In the original code: w_col[k * n + i] = coeff.
            # Here i is row (from U), k is col (from V).
            # My local w_d indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # r is row in C (0,1), c is col in C (0,1)
            for loc_idx, val in w_d.items():
                r, c = divmod(loc_idx, 2)
                i = oi + r
                k = ok + c
                g_idx = k * 4 + i
                w_vec[g_idx] = val

            append_term(u_vec, v_vec, w_vec)

    # Recursive Strassen for 4x4x4
    # We can call strassen_2x2_offsets on the 4 blocks of the 2x2 super-matrix
    # But we need to handle the 4x4 structure.
    # The original code used a super_term approach.
    # Let's stick to the 4x4x4 rank 49 by using 7 super-terms of rank 7.
    
    def strassen_4x4_4x4_4x4():
        # Blocks of A, B, C are 2x2.
        # A: A00 A01 / A10 A11
        # B: B00 B01 / B10 B11
        # C: C00 C01 / C10 C11
        
        # Super terms
        # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
        # This requires adding rank-7 terms for each block multiplication.
        # But the super term is a sum of block products.
        # We need to decompose the bilinear form of the super term.
        
        # Instead, we can just call strassen_2x2_offsets for the 4x4 case?
        # No, strassen_2x2_offsets computes 2x2 * 2x2.
        # To do 4x4 * 4x4, we need to apply Strassen to the 2x2 block structure.
        # Each "element" of the block-matrix multiplication is a 2x2 * 2x2 multiplication.
        # Strassen on 2x2 block matrices uses 7 block multiplications.
        # Each block multiplication is a 2x2 * 2x2 tensor (rank 7).
        # So total rank 7 * 7 = 49.
        
        # We need to implement the 7 block multiplications.
        # Block indices: 00, 01, 10, 11
        # Offsets:
        # Block 00: oi=0, oj=0
        # Block 01: oi=0, oj=2
        # Block 10: oi=2, oj=0
        # Block 11: oi=2, oj=2
        
        # Super term 1: (A00+A11)(B00+B11)
        # This corresponds to computing M1 = (A00+A11)(B00+B11).
        # This is a 2x2 * 2x2 multiplication?
        # A_block = A00 + A11 (2x2 matrix)
        # B_block = B00 + B11 (2x2 matrix)
        # Result M1 is 2x2.
        # C00 += M1, C11 += M1.
        
        # So we need to decompose the product of two linear combinations of blocks.
        # This is equivalent to 7 terms of rank 7?
        # Yes.
        
        # Helper to add a super-term
        def add_super(u_blocks, v_blocks, w_blocks):
            # u_blocks: dict {block_idx: coeff}
            # block_idx: 0:00, 1:01, 2:10, 3:11
            # For each of the 7 Strassen terms (local), we generate a global term.
            
            local_terms = [
                ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
                ({2:1, 3:1}, {0:1},    {2:1, 3:-1}),
                ({0:1},    {1:1, 3:-1},{1:1, 3:1}),
                ({3:1},    {2:1, 0:-1},{0:1, 2:1}),
                ({0:1, 1:1},{3:1},     {0:-1, 1:1}),
                ({2:1, 0:-1},{0:1, 1:1},{3:1}),
                ({1:1, 3:-1},{2:1, 3:1},{0:1})
            ]
            
            for u_l, v_l, w_l in local_terms:
                u_vec = [0] * 16
                v_vec = [0] * 24
                w_vec = [0] * 24
                
                # Accumulate U
                for bl, cu in u_blocks.items():
                    r, c = divmod(bl, 2) # 00->0,0; 01->0,1; 10->1,0; 11->1,1
                    oi = r * 2
                    oj = c * 2
                    for loc_idx, cv in u_l.items():
                        lr, lc = divmod(loc_idx, 2)
                        gi = (oi + lr) * 4 + (oj + lc)
                        u_vec[gi] += cu * cv
                        
                # Accumulate V
                for bl, cv in v_blocks.items():
                    r, c = divmod(bl, 2)
                    oj = r * 2
                    ok = c * 2
                    for loc_idx, cvv in v_l.items():
                        lr, lc = divmod(loc_idx, 2)
                        gj = (oj + lr) * 6 + (ok + lc)
                        v_vec[gj] += cv * cvv
                        
                # Accumulate W
                for bl, cw in w_blocks.items():
                    r, c = divmod(bl, 2)
                    oi = r * 2
                    ok = c * 2
                    for loc_idx, cvv in w_l.items():
                        lr, lc = divmod(loc_idx, 2)
                        gi = oi + lr
                        gk = ok + lc
                        gw = gk * 4 + gi
                        w_vec[gw] += cw * cvv

                append_term(u_vec, v_vec, w_vec)

        # 7 Super Terms
        add_super({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        add_super({2:1, 3:1}, {0:1},    {2:1, 3:-1})
        add_super({0:1},    {1:1, 3:-1},{1:1, 3:1})
        add_super({3:1},    {2:1, 0:-1},{0:1, 2:1})
        add_super({0:1, 1:1},{3:1},     {0:-1, 1:1})
        add_super({2:1, 0:-1},{0:1, 1:1},{3:1})
        add_super({1:1, 3:-1},{2:1, 3:1},{0:1})

    # Execute 4x4x4 part
    if _time_remaining() > 5.0:
        strassen_4x4_4x4_4x4()
        
    # Execute 4x4x2 part (Cols 4,5)
    # Split A into 2x2 blocks, B into 2x2 blocks (cols 4-5)
    # We need to compute C[0:2, 4:6] and C[2:4, 4:6]
    # C_top = A00 B0 + A01 B1
    # C_bot = A10 B0 + A11 B1
    # Where B0 is B[0:2, 4:6], B1 is B[2:4, 4:6]
    
    # We can use the same add_super logic?
    # No, this is simpler. Just 4 independent 2x2x2 multiplications?
    # A00 B0 (rank 7), A01 B1 (rank 7), A10 B0 (rank 7), A11 B1 (rank 7).
    # Total 28.
    
    def add_2x2x2(oi, oj, ok):
        # oi: row offset in A (0 or 2)
        # oj: col offset in A (0 or 2)
        # ok: col offset in B (4)
        strassen_2x2_offsets(oi, oj, ok)
        
    if _time_remaining() > 2.0:
        add_2x2x2(0, 0, 4)
        add_2x2x2(0, 2, 4)
        add_2x2x2(2, 0, 4)
        add_2x2x2(2, 2, 4)
        
    rank = len(U_list)
    
    U_final = [[U_list[r][i] for r in range(rank)] for i in range(16)]
    V_final = [[V_list[r][i] for r in range(rank)] for i in range(24)]
    W_final = [[W_list[r][i] for r in range(rank)] for i in range(24)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
