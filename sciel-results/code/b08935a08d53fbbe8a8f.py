"""
Program for finding a rank-minimal tensor decomposition of the 3x4 by 4x8 matrix multiplication.
Uses a block-strategy: Strassen for 2x2 sub-blocks where possible, schoolbook for the 1x4 rows.
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We will build the decomposition column by column (rank-1 terms).
    # Each term is a triple (u, v, w) where:
    # u is a sparse vector in R^{n*m}
    # v is a sparse vector in R^{m*p}
    # w is a sparse vector in R^{p*n}
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen 2x2 decomposition parameters
    # For A (2x2) and B (2x2), C = A*B
    # Indices in A: (0,0)=tl, (0,1)=tr, (1,0)=bl, (1,1)=br
    # Indices in B: (0,0)=tl, (0,1)=tr, (1,0)=bl, (1,1)=br
    # Indices in C: (0,0)=tl, (0,1)=tr, (1,0)=bl, (1,1)=br
    
    # Strassen terms:
    # M1 = (A_tl + A_br) * (B_tl + B_br)
    # M2 = (A_bl + A_br) * B_tl
    # M3 = A_tl * (B_tr - B_br)
    # M4 = A_br * (B_bl - B_tl)
    # M5 = (A_tl + A_tr) * B_br
    # M6 = (A_bl - A_tl) * (B_tl + B_tr)
    # M7 = (A_tr - A_br) * (B_bl + B_br)
    
    # C_tl = M1 + M4 - M5 + M7
    # C_tr = M3 + M5
    # C_bl = M2 + M4
    # C_br = M1 - M2 + M3 + M6
    
    # Each M_i is a bilinear form. We decompose it into rank-1 terms.
    # For M1: (A_tl + A_br) * (B_tl + B_br)
    # This expands to 4 terms:
    # A_tl*B_tl, A_tl*B_br, A_br*B_tl, A_br*B_br
    # But in Strassen, we treat M1 as a single "multiplier" which is a product of sums.
    # However, the rank of the tensor for 2x2 multiplication is 7.
    # This means we can represent the 8 schoolbook terms with 7 rank-1 terms.
    # The 7 rank-1 terms correspond to the 7 multipliers M1..M7.
    # Each multiplier M_i is a product of a linear form in A and a linear form in B.
    # So each M_i contributes ONE rank-1 term to the decomposition?
    # No, the output C is a sum of these multipliers.
    # C_tl = M1 + M4 - M5 + M7
    # This means the tensor for C_tl is the sum of 4 rank-1 tensors (from M1, M4, M5, M7).
    # So each Strassen multiplier M_i corresponds to a rank-1 tensor in the "intermediate" space?
    # No, the standard interpretation is:
    # We have 7 variables x1..x7 corresponding to the multipliers.
    # Each x_i is computed by a bilinear map.
    # The final C is a linear combination of x_i.
    # In the tensor decomposition U, V, W:
    # Each column r corresponds to one multiplier.
    # U[:, r] encodes the linear form in A for that multiplier.
    # V[:, r] encodes the linear form in B for that multiplier.
    # W[:, r] encodes the linear combination coefficients for the output C.
    
    # So for a 2x2 block at A-offset (ar, ac), B-offset (br, bc), C-offset (cr, cc):
    # We add 7 terms.
    # Term 1 (M1):
    #   u: A_tl + A_br -> {(ar,ac):1, (ar+1,ac+1):1}
    #   v: B_tl + B_br -> {(br,bc):1, (br+1,bc+1):1}
    #   w: C_tl - C_br? No.
    #   M1 contributes to C_tl (+1) and C_br (+1).
    #   w: {(cr,cc):1, (cr+1,cc+1):1}
    
    # Let's verify the Strassen formulas for W contributions.
    # C_tl = M1 + M4 - M5 + M7
    # C_tr = M3 + M5
    # C_bl = M2 + M4
    # C_br = M1 - M2 + M3 + M6
    
    # So:
    # M1: w = {C_tl: 1, C_br: 1}
    # M2: w = {C_bl: 1, C_br: -1}
    # M3: w = {C_tr: 1, C_br: 1}
    # M4: w = {C_tl: 1, C_bl: 1}
    # M5: w = {C_tl: -1, C_tr: 1}
    # M6: w = {C_br: 1}
    # M7: w = {C_tl: 1}
    
    # And the U, V for each M:
    # M1: u = {A_tl: 1, A_br: 1}, v = {B_tl: 1, B_br: 1}
    # M2: u = {A_bl: 1, A_br: 1}, v = {B_tl: 1}
    # M3: u = {A_tl: 1}, v = {B_tr: 1, B_br: -1}
    # M4: u = {A_br: 1}, v = {B_bl: 1, B_tl: -1}
    # M5: u = {A_tl: 1, A_tr: 1}, v = {B_br: 1}
    # M6: u = {A_bl: 1, A_tl: -1}, v = {B_tl: 1, B_tr: 1}
    # M7: u = {A_tr: 1, A_br: -1}, v = {B_bl: 1, B_br: 1}
    
    def add_strassen_block(ar, ac, br, bc, cr, cc):
        # ar, ac: top-left row/col of A block
        # br, bc: top-left row/col of B block
        # cr, cc: top-left row/col of C block
        
        # M1
        u = {(ar, ac): 1, (ar+1, ac+1): 1}
        v = {(br, bc): 1, (br+1, bc+1): 1}
        w = {(cr, cc): 1, (cr+1, cc+1): 1}
        add_term(u, v, w)
        
        # M2
        u = {(ar+1, ac): 1, (ar+1, ac+1): 1}
        v = {(br, bc): 1}
        w = {(cr+1, cc): 1, (cr+1, cc+1): -1}
        add_term(u, v, w)
        
        # M3
        u = {(ar, ac): 1}
        v = {(br, bc+1): 1, (br+1, bc+1): -1}
        w = {(cr, cc+1): 1, (cr+1, cc+1): 1}
        add_term(u, v, w)
        
        # M4
        u = {(ar+1, ac+1): 1}
        v = {(br+1, bc): 1, (br, bc): -1}
        w = {(cr, cc): 1, (cr+1, cc): 1}
        add_term(u, v, w)
        
        # M5
        u = {(ar, ac): 1, (ar, ac+1): 1}
        v = {(br+1, bc+1): 1}
        w = {(cr, cc): -1, (cr, cc+1): 1}
        add_term(u, v, w)
        
        # M6
        u = {(ar+1, ac): 1, (ar, ac): -1}
        v = {(br, bc): 1, (br, bc+1): 1}
        w = {(cr+1, cc+1): 1}
        add_term(u, v, w)
        
        # M7
        u = {(ar, ac+1): 1, (ar+1, ac+1): -1}
        v = {(br+1, bc): 1, (br+1, bc+1): 1}
        w = {(cr, cc): 1}
        add_term(u, v, w)

    def add_schoolbook_block(ar, ac, br, bc, cr, cc, rows_a, cols_a, rows_b, cols_b, rows_c, cols_c):
        # Schoolbook for a block of size rows_a x cols_a * rows_b x cols_b -> rows_c x cols_c
        # where cols_a == rows_b
        for i in range(rows_a):
            for j in range(cols_a):
                for k in range(cols_b):
                    u = {(ar+i, ac+j): 1}
                    v = {(br+j, bc+k): 1}
                    w = {(cr+i, cc+k): 1}
                    add_term(u, v, w)

    # Now, apply to our 3x4 * 4x8 case.
    # A is 3x4. B is 4x8. C is 3x8.
    
    # Split A into top 2 rows (A_top, 2x4) and bottom 1 row (A_bot, 1x4).
    # Split B into left 4 cols (B_left, 4x4) and right 4 cols (B_right, 4x4).
    # Split C into top 2 rows (C_top, 2x8) and bottom 1 row (C_bot, 1x8).
    
    # C_top = A_top * B = A_top * B_left + A_top * B_right
    # C_bot = A_bot * B = A_bot * B_left + A_bot * B_right
    
    # 1. A_top * B_left (2x4 * 4x4 -> 2x4)
    # Split A_top into left 2 cols (A_tl, 2x2) and right 2 cols (A_tr, 2x2).
    # Split B_left into top 2 rows (B_t, 2x4) and bottom 2 rows (B_b, 2x4).
    # C_top_left (2x4) is split into left 2 cols (C_tl, 2x2) and right 2 cols (C_tr, 2x2).
    
    # A_tl * B_t -> C_tl (2x2 * 2x2 -> 2x2)
    # A is at (0,0), B at (0,0), C at (0,0)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # A_tl * B_b -> C_tl? No, A_tl (2x2) * B_b (2x4) is not square.
    # Wait, B_left is 4x4.
    # B_left = [B_tl; B_bl] where B_tl is 2x4, B_bl is 2x4.
    # A_top = [A_tl, A_tr] where A_tl is 2x2, A_tr is 2x2.
    # C_top_left = A_top * B_left = A_tl * B_tl + A_tr * B_bl? No.
    # Matrix multiplication:
    # A_top (2x4) * B_left (4x4)
    # = [A_tl A_tr] * [B_tl; B_bl] where B_tl is 2x4, B_bl is 2x4.
    # This is not a block multiplication in the standard sense because the inner dimension is 4.
    # We need to partition the inner dimension (4) into 2+2.
    # Let B_left = [B1; B2] where B1, B2 are 2x4.
    # Let A_top = [A1 A2] where A1, A2 are 2x2.
    # Then A_top * B_left = A1 * B1 + A2 * B2.
    # A1 (2x2) * B1 (2x4) -> 2x4.
    # A2 (2x2) * B2 (2x4) -> 2x4.
    # So we have two 2x2 * 2x4 multiplications.
    
    # Let's refine the block strategy.
    # A_top (2x4) * B_left (4x4)
    # Partition A_top cols into 2+2: A1 (2x2), A2 (2x2).
    # Partition B_left rows into 2+2: B1 (2x4), B2 (2x4).
    # C_top_left = A1 * B1 + A2 * B2.
    
    # A1 (2x2) * B1 (2x4) -> 2x4.
    # We can split B1 into two 2x2 blocks: B1_l (2x2), B1_r (2x2).
    # C_top_left_l (2x2) = A1 * B1_l.
    # C_top_left_r (2x2) = A1 * B1_r.
    # So we need 2 Strassen blocks for A1 * B1.
    
    # Similarly for A2 * B2.
    
    # Let's list all blocks:
    # A1 is at A[0:2, 0:2].
    # A2 is at A[0:2, 2:4].
    # B1 is at B[0:2, 0:4].
    # B2 is at B[2:4, 0:4].
    # C_top_left is at C[0:2, 0:4].
    # C_top_left_l is at C[0:2, 0:2].
    # C_top_left_r is at C[0:2, 2:4].
    
    # Block 1: A1 * B1_l -> C_top_left_l
    # A1 (0,0), B1_l (0,0), C (0,0)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 * B1_r -> C_top_left_r
    # A1 (0,0), B1_r (0,2), C (0,2)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: A2 * B2_l -> C_top_left_l
    # A2 (0,2), B2_l (2,0), C (0,0)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 4: A2 * B2_r -> C_top_left_r
    # A2 (0,2), B2_r (2,2), C (0,2)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Now C_top_right = A_top * B_right (2x4 * 4x4 -> 2x4)
    # Same structure.
    # A1 (0,0), A2 (0,2).
    # B3 (0:2, 4:8), B4 (2:4, 4:8).
    # C_top_right_l (0:2, 4:6), C_top_right_r (0:2, 6:8).
    
    # Block 5: A1 * B3_l -> C_top_right_l
    # A1 (0,0), B3_l (0,4), C (0,4)
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # Block 6: A1 * B3_r -> C_top_right_r
    # A1 (0,0), B3_r (0,6), C (0,6)
    add_strassen_block(0, 0, 0, 6, 0, 6)
    
    # Block 7: A2 * B4_l -> C_top_right_l
    # A2 (0,2), B4_l (2,4), C (0,4)
    add_strassen_block(0, 2, 2, 4, 0, 4)
    
    # Block 8: A2 * B4_r -> C_top_right_r
    # A2 (0,2), B4_r (2,6), C (0,6)
    add_strassen_block(0, 2, 2, 6, 0, 6)
    
    # Now C_bot_left = A_bot * B_left (1x4 * 4x4 -> 1x4)
    # A_bot is at A[2, 0:4].
    # B_left is at B[0:4, 0:4].
    # C_bot_left is at C[2, 0:4].
    # This is a 1x4 * 4x4 multiplication.
    # We can use schoolbook.
    # A_bot is 1x4. B_left is 4x4.
    # C[2, k] = sum_j A[2, j] B[j, k].
    # This is 4 dot products of length 4.
    # Total 16 terms.
    add_schoolbook_block(2, 0, 0, 0, 2, 0, 1, 4, 4, 4, 1, 4)
    
    # C_bot_right = A_bot * B_right (1x4 * 4x4 -> 1x4)
    # A_bot is at A[2, 0:4].
    # B_right is at B[0:4, 4:8].
    # C_bot_right is at C[2, 4:8].
    add_schoolbook_block(2, 0, 0, 4, 2, 4, 1, 4, 4, 4, 1, 4)
    
    # Total rank:
    # 8 Strassen blocks * 7 terms = 56 terms.
    # 2 Schoolbook blocks * 16 terms = 32 terms.
    # Total = 88 terms.
    
    # Convert to matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] += val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] += val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] += val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
