import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_2x2(off_i, off_j, off_k, u_offsets, v_offsets, w_offsets, sign):
        # u_offsets, v_offsets, w_offsets are lists of (di, dj) offsets for the 2x2 block
        # sign is the multiplier for the whole term
        
        # Generate 7 Strassen terms for 2x2 multiplication
        # Each term is (u_sparse, v_sparse, w_sparse) of length 4
        # We map them to global 25-dim vectors
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [0]*4; v1 = [0]*4; w1 = [0]*4
        u1[0] = 1; u1[3] = 1 # A00[0,0] + A11[1,1] -> wait, 2x2 indices are 0..1
        # A00 is top-left, A11 is bottom-right of the 2x2 virtual matrix?
        # No, the virtual matrix IS the 2x2.
        # The "A00" in the 4x4 context corresponds to the (0,0) element of the virtual 2x2.
        # So for M1, U_virtual[0,0] = 1, U_virtual[1,1] = 1.
        u1[0] = 1; u1[3] = 1
        v1[0] = 1; v1[3] = 1
        w1[0] = 1; w1[3] = 1
        add_mapped_2x2_term(u1, v1, w1, u_offsets, v_offsets, w_offsets, sign)

        # M2 = (A10+A11)B00
        u2 = [0]*4; v2 = [0]*4; w2 = [0]*4
        u2[2] = 1; u2[3] = 1 # A10[0,0] + A11[1,1]? 
        # In 4x4, A10 is bottom-left, A11 is bottom-right.
        # Virtual 2x2: A10 is (0,0)?? No.
        # The virtual matrix is formed by summing the sub-blocks.
        # For M2, A_virtual = A10 + A11.
        # A10 is at (2,0)..(3,1) in global. In virtual 2x2, it's the whole matrix?
        # No, A10 and A11 are 2x2 blocks.
        # A_virtual[i,j] = A10[i,j] + A11[i,j].
        # So A_virtual is a 2x2 matrix.
        # For M2, B_virtual = B00.
        # B00 is a 2x2 block. So B_virtual[i,j] = B00[i,j].
        # So u2 corresponds to A10+A11.
        # But we don't need to populate u2 with 1s for every element.
        # We need to generate the 7 Strassen terms for the multiplication of 
        # matrix X (where X[i,j] = A10[i,j]+A11[i,j]) and Y (where Y[i,j] = B00[i,j]).
        # The Strassen terms depend on the structure of X and Y.
        # But X and Y are "dense" in the sense that they use all 4 elements.
        # So we can just use the standard 2x2 Strassen templates, 
        # where the "indices" refer to the virtual 2x2 matrix.
        
        # Wait, the standard 2x2 Strassen templates (like u1[0]=1, u1[3]=1) 
        # assume that the input matrices are the standard basis vectors?
        # No, the templates are for the COEFFICIENTS of the rank-1 terms.
        # The rank-1 term is u * v * w.
        # u, v, w are vectors of length 4 (flattened 2x2).
        # The template u1=[1,0,0,1] means the term picks up the (0,0) and (1,1) elements of A.
        # This is correct for the Strassen algorithm on a generic 2x2 matrix.
        # So we can reuse the templates.
        
        # We just need to map the virtual indices to global indices.
        pass

    # Helper to add a single rank-1 term from 2x2 Strassen to global vectors
    def add_mapped_2x2_term(u_2x2, v_2x2, w_2x2, u_maps, v_maps, w_maps, sign):
        # u_maps: list of (offset_i, offset_j) for A
        # v_maps: list of (offset_j, offset_k) for B
        # w_maps: list of (offset_k, offset_i, sign_factor) for C
        
        U_global = [0]*25
        V_global = [0]*25
        W_global = [0]*25
        
        for idx in range(4):
            ui, uj = divmod(idx, 2)
            val = u_2x2[idx]
            if val != 0:
                for (oi, oj) in u_maps:
                    gi = oi + ui
                    gj = oj + uj
                    U_global[gi*5 + gj] += val * sign
                    
        for idx in range(4):
            vi, vj = divmod(idx, 2)
            val = v_2x2[idx]
            if val != 0:
                for (oj, ok) in v_maps:
                    gi = oj + vi
                    gj = ok + vj
                    V_global[gi*5 + gj] += val * sign
                    
        for idx in range(4):
            wi, wj = divmod(idx, 2)
            val = w_2x2[idx]
            if val != 0:
                for (ok, oi, sf) in w_maps:
                    gi = ok + wi
                    gj = oi + wj
                    W_global[gi*5 + gj] += val * sign * sf

        if any(U_global) or any(V_global) or any(W_global):
            add_col(U_global, V_global, W_global)

    # Define 4x4 Strassen blocks
    # Sub-block offsets for 4x4 (0,0,0)
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    u_maps_1 = [(0,0), (2,2)]
    v_maps_1 = [(0,0), (2,2)]
    w_maps_1 = [(0,0,1), (2,2,1)]
    
    # M2: (A10+A11)B00 -> C10+C11
    u_maps_2 = [(2,0), (2,2)]
    v_maps_2 = [(0,0)]
    w_maps_2 = [(2,0,1), (2,2,1)]
    
    # M3: A00(B01-B11) -> C01
    u_maps_3 = [(0,0)]
    v_maps_3 = [(0,2), (2,2)] # B01 + (-1)*B11? 
    # Wait, v_maps needs to handle signs?
    # My helper adds val * sign. It doesn't handle per-element signs in the map.
    # I should include the sign in the map or handle it in the template.
    # Better: include sign in the map tuple: (oi, oj, s)
    
    # Let's redefine maps with signs
    # u_maps: [(oi, oj, s), ...]
    # v_maps: [(oj, ok, s), ...]
    # w_maps: [(ok, oi, s), ...]

    # M1
    u_maps_1 = [(0,0,1), (2,2,1)]
    v_maps_1 = [(0,0,1), (2,2,1)]
    w_maps_1 = [(0,0,1), (2,2,1)]
    
    # M2
    u_maps_2 = [(2,0,1), (2,2,1)]
    v_maps_2 = [(0,0,1)]
    w_maps_2 = [(2,0,1), (2,2,1)]
    
    # M3
    u_maps_3 = [(0,0,1)]
    v_maps_3 = [(0,2,1), (2,2,-1)]
    w_maps_3 = [(0,2,1)]
    
    # M4
    u_maps_4 = [(2,2,1)]
    v_maps_4 = [(2,0,1), (0,0,-1)]
    w_maps_4 = [(0,0,1), (2,0,1)]
    
    # M5
    u_maps_5 = [(0,0,1), (0,2,1)]
    v_maps_5 = [(2,2,1)]
    w_maps_5 = [(0,0,-1), (0,2,1)]
    
    # M6
    u_maps_6 = [(2,0,1), (0,0,-1)]
    v_maps_6 = [(0,0,1), (0,2,1)]
    w_maps_6 = [(2,2,1)]
    
    # M7
    u_maps_7 = [(0,2,1), (2,2,-1)]
    v_maps_7 = [(2,0,1), (2,2,1)]
    w_maps_7 = [(0,0,1)]

    # Standard 2x2 Strassen templates (u, v, w) for the 7 terms
    # u, v, w are length 4 lists
    
    # T1: (A00+A11)(B00+B11)
    t1_u = [1,0,0,1]; t1_v = [1,0,0,1]; t1_w = [1,0,0,1]
    # T2: (A10+A11)B00
    t2_u = [0,0,1,1]; t2_v = [1,0,0,0]; t2_w = [0,0,1,1]
    # T3: A00(B01-B11)
    t3_u = [1,0,0,0]; t3_v = [0,1,0,-1]; t3_w = [0,1,0,1]
    # T4: A11(B10-B00)
    t4_u = [0,0,0,1]; t4_v = [0,0,1,-1]; t4_w = [1,0,1,0]
    # T5: (A00+A01)B11
    t5_u = [1,1,0,0]; t5_v = [0,0,0,1]; t5_w = [0,-1,1,0]
    # T6: (A10-A00)(B00+B01)
    t6_u = [0,0,1,-1]; t6_v = [1,1,0,0]; t6_w = [0,0,0,1]
    # T7: (A01-A11)(B10+B11)
    t7_u = [0,1,0,-1]; t7_v = [0,0,1,1]; t7_w = [1,0,0,0]

    templates = [
        (t1_u, t1_v, t1_w, u_maps_1, v_maps_1, w_maps_1),
        (t2_u, t2_v, t2_w, u_maps_2, v_maps_2, w_maps_2),
        (t3_u, t3_v, t3_w, u_maps_3, v_maps_3, w_maps_3),
        (t4_u, t4_v, t4_w, u_maps_4, v_maps_4, w_maps_4),
        (t5_u, t5_v, t5_w, u_maps_5, v_maps_5, w_maps_5),
        (t6_u, t6_v, t6_w, u_maps_6, v_maps_6, w_maps_6),
        (t7_u, t7_v, t7_w, u_maps_7, v_maps_7, w_maps_7)
    ]

    for (tu, tv, tw, um, vm, wm) in templates:
        if time.time() > deadline: break
        add_mapped_2x2_term(tu, tv, tw, um, vm, wm, 1)

    # Schoolbook for remaining
    covered = set()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                covered.add((i,j,k))
                
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if (i,j,k) not in covered:
                    if time.time() > deadline: break
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[i*5+j] = 1
                    v_vec[j*5+k] = 1
                    w_vec[k*5+i] = 1
                    add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline: break
        if time.time() > deadline: break

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        for row_idx in range(25):
            U[row_idx][col_idx] = U_cols[col_idx][row_idx]
            V[row_idx][col_idx] = V_cols[col_idx][row_idx]
            W[row_idx][col_idx] = W_cols[col_idx][row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
