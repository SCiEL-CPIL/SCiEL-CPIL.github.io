import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 + Rank 16 for 4x4x1 = 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 using Strassen 49
    # We need to define the 49 terms.
    
    # Strassen 2x2x2 coefficients
    # q1 = (x11+x22)(y11+y22)
    # q2 = (x21+x22)y11
    # q3 = x11(y12-y22)
    # q4 = x22(y21-y11)
    # q5 = (x11+x12)y22
    # q6 = (x21-x11)(y11+y12)
    # q7 = (x12-x22)(y21+y22)
    
    # Z11 = q1 + q4 - q5 + q7
    # Z12 = q3 + q5
    # Z21 = q2 + q4
    # Z22 = q1 - q2 + q3 + q6
    
    # X, Y definitions for 4x4 blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # X1 = A00 + A11
    # X2 = A10 + A11
    # X3 = A00
    # X4 = A11
    # X5 = A00 + A01
    # X6 = A10 - A00
    # X7 = A01 - A11
    
    # Y1 = B00 + B11
    # Y2 = B00
    # Y3 = B01 - B11
    # Y4 = B10 - B00
    # Y5 = B11
    # Y6 = B00 + B01
    # Y7 = B10 + B11
    
    # We represent linear forms as dicts { (i,j): coeff }
    
    def add_to_vec(vec, idx, val):
        if idx in vec:
            vec[idx] += val
        else:
            vec[idx] = val

    # Define X_l as list of 4 dicts (for 2x2 matrix entries)
    # Each dict maps (r,c) in A to coeff
    X_defs = []
    Y_defs = []
    
    # Helper to create a block matrix definition
    # block(r_block, c_block) returns a dict with 1s at (2*r_block+r, 2*c_block+c)
    def block_dict(rb, cb, sign=1):
        d = {}
        for r in range(2):
            for c in range(2):
                add_to_vec(d, (2*rb+r, 2*cb+c), sign)
        return d

    # X1 = A00 + A11
    X1 = [
        {**block_dict(0,0), **block_dict(1,1)}, # X1_11 = A00_11 + A11_11? No.
        # X_l is a 2x2 matrix.
        # X_l[0,0] is the top-left 2x2 block of X_l?
        # No, X_l is a 2x2 matrix of 2x2 blocks?
        # No, X_l is a 2x2 matrix where each entry is a 2x2 block of A.
        # So X_l[0,0] is a 2x2 matrix.
        # In Strassen 2x2, x11 is a scalar.
        # Here x11 is a 2x2 block.
        # So the "scalar" multiplication is actually a block multiplication.
        # But we are decomposing the block multiplication into scalar multiplications.
        # So X_l[0,0] is a linear form in A.
        
        # Let's define X_l as a 2x2 list of dicts.
    ]
    
    # Actually, it's easier to think of X_l as a 4x4 matrix of coefficients?
    # No, X_l is a 2x2 matrix of blocks.
    # Each block is a linear combination of A's blocks.
    
    # Let's define the linear forms for the 4 entries of X_l.
    # X_l[0,0], X_l[0,1], X_l[1,0], X_l[1,1].
    
    # X1 = A00 + A11 (scalar sum of blocks)
    # So X1[0,0] = A00[0,0] + A11[0,0]?
    # No, X1 is the sum of matrices A00 and A11.
    # So X1 is a 2x2 matrix where (r,c) entry is A00[r,c] + A11[r,c].
    
    def sum_blocks(blocks):
        # blocks is list of (rb, cb, sign)
        res = [[None for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                d = {}
                for rb, cb, s in blocks:
                    add_to_vec(d, (2*rb+r, 2*cb+c), s)
                res[r][c] = d
        return res

    X_defs = [
        sum_blocks([(0,0,1), (1,1,1)]), # X1
        sum_blocks([(1,0,1), (1,1,1)]), # X2
        sum_blocks([(0,0,1)]),          # X3
        sum_blocks([(1,1,1)]),          # X4
        sum_blocks([(0,0,1), (0,1,1)]), # X5
        sum_blocks([(1,0,1), (0,0,-1)]),# X6
        sum_blocks([(0,1,1), (1,1,-1)]) # X7
    ]
    
    Y_defs = [
        sum_blocks([(0,0,1), (1,1,1)]), # Y1
        sum_blocks([(0,0,1)]),          # Y2
        sum_blocks([(0,1,1), (1,1,-1)]),# Y3
        sum_blocks([(1,0,1), (0,0,-1)]),# Y4
        sum_blocks([(1,1,1)]),          # Y5
        sum_blocks([(0,0,1), (0,1,1)]), # Y6
        sum_blocks([(1,0,1), (1,1,1)])  # Y7
    ]
    
    # Now for each l in 0..6, we have X_l and Y_l (2x2 matrices of dicts).
    # We apply Strassen 7 to compute Z_l = X_l Y_l.
    # The 7 products q_{l,1}..q_{l,7} are formed from linear combinations of the 4 entries of X_l and Y_l.
    # Each entry of X_l is a dict (linear form in A).
    # So a linear combination of entries is a dict (linear form in A).
    
    # Strassen coefficients for 2x2:
    # q1: (x00+x11) * (y00+y11)
    # q2: (x10+x11) * y00
    # q3: x00 * (y01-y11)
    # q4: x11 * (y10-y00)
    # q5: (x00+x01) * y11
    # q6: (x10-x00) * (y00+y01)
    # q7: (x01-x11) * (y10+y11)
    
    def combine_dicts(dicts, coeffs):
        res = {}
        for d, c in zip(dicts, coeffs):
            if c == 0: continue
            for idx, val in d.items():
                add_to_vec(res, idx, val * c)
        return res

    strassen_terms = [
        ([(0,0), (1,1)], [(0,0), (1,1)]), # q1
        ([(1,0), (1,1)], [(0,0)]),        # q2
        ([(0,0)], [(0,1), (1,1)]),        # q3 (coeffs 1, -1)
        ([(1,1)], [(1,0), (0,0)]),        # q4 (coeffs 1, -1)
        ([(0,0), (0,1)], [(1,1)]),        # q5
        ([(1,0), (0,0)], [(0,0), (0,1)]), # q6 (coeffs 1, -1 for x; 1, 1 for y)
        ([(0,1), (1,1)], [(1,0), (1,1)])  # q7 (coeffs 1, -1 for x; 1, 1 for y)
    ]
    
    # Need to handle signs in the linear forms.
    # q3: x00 * (y01 - y11)
    # q4: x11 * (y10 - y00)
    # q6: (x10 - x00) * (y00 + y01)
    # q7: (x01 - x11) * (y10 + y11)
    
    # Let's define the linear forms explicitly with signs.
    # X_l is 2x2 list of dicts.
    # Y_l is 2x2 list of dicts.
    
    for l in range(7):
        Xl = X_defs[l]
        Yl = Y_defs[l]
        
        # Define the 7 linear forms for X and Y
        # xf[t] is dict for X part of q_t
        # yf[t] is dict for Y part of q_t
        
        xf = [None]*7
        yf = [None]*7
        
        # q1
        xf[0] = combine_dicts([Xl[0][0], Xl[1][1]], [1, 1])
        yf[0] = combine_dicts([Yl[0][0], Yl[1][1]], [1, 1])
        
        # q2
        xf[1] = combine_dicts([Xl[1][0], Xl[1][1]], [1, 1])
        yf[1] = Yl[0][0]
        
        # q3
        xf[2] = Xl[0][0]
        yf[2] = combine_dicts([Yl[0][1], Yl[1][1]], [1, -1])
        
        # q4
        xf[3] = Xl[1][1]
        yf[3] = combine_dicts([Yl[1][0], Yl[0][0]], [1, -1])
        
        # q5
        xf[4] = combine_dicts([Xl[0][0], Xl[0][1]], [1, 1])
        yf[4] = Yl[1][1]
        
        # q6
        xf[5] = combine_dicts([Xl[1][0], Xl[0][0]], [1, -1])
        yf[5] = combine_dicts([Yl[0][0], Yl[0][1]], [1, 1])
        
        # q7
        xf[6] = combine_dicts([Xl[0][1], Xl[1][1]], [1, -1])
        yf[6] = combine_dicts([Yl[1][0], Yl[1][1]], [1, 1])
        
        # Now for each t in 0..6, we have a term q_{l,t} = xf[t] * yf[t].
        # This term contributes to Z_l entries.
        # Z_l[0][0] = q1 + q4 - q5 + q7
        # Z_l[0][1] = q3 + q5
        # Z_l[1][0] = q2 + q4
        # Z_l[1][1] = q1 - q2 + q3 + q6
        
        z_coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # Z00
            [0, 0, 1, 0, 1, 0, 0],  # Z01
            [0, 1, 0, 1, 0, 0, 0],  # Z10
            [1, -1, 1, 0, 0, 1, 0]  # Z11
        ]
        
        # Z_l is a 2x2 matrix of "values" (which are linear forms in C? No, they are the products).
        # Z_l contributes to C blocks.
        # C00 = Z1 + Z4 - Z5 + Z7
        # C01 = Z3 + Z5
        # C10 = Z2 + Z4
        # C11 = Z1 - Z2 + Z3 + Z6
        
        c_block_coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # C00 gets Z1, Z4, -Z5, Z7
            [0, 0, 1, 0, 1, 0, 0],  # C01 gets Z3, Z5
            [0, 1, 0, 1, 0, 0, 0],  # C10 gets Z2, Z4
            [1, -1, 1, 0, 0, 1, 0]  # C11 gets Z1, -Z2, Z3, Z6
        ]
        
        # For each t (0..6), q_{l,t} is a scalar product of linear forms.
        # It contributes to Z_l[rb][cb] with coefficient z_coeffs[rb][cb][t].
        # Z_l[rb][cb] contributes to C_block[K][L] with coefficient c_block_coeffs[K][L][l].
        # So q_{l,t} contributes to C_block[K][L] with coeff c_block_coeffs[K][L][l] * z_coeffs[rb][cb][t].
        # Wait, Z_l[rb][cb] is an entry of the 2x2 matrix Z_l.
        # Z_l is added to C_block[K][L].
        # So the entry Z_l[rb][cb] is added to C_block[K][L][rb][cb].
        # So for a fixed t, and fixed l:
        # The term q_{l,t} has a U part (xf[t]) and V part (yf[t]).
        # Its W part is a linear combination of C entries.
        # Which C entries?
        # For each rb, cb in 0..1:
        #   coeff_z = z_coeffs[rb][cb][t]
        #   If coeff_z != 0:
        #     For each K, L in 0..1:
        #       coeff_c = c_block_coeffs[K][L][l]
        #       If coeff_c != 0:
        #         Total coeff for C[K][L][rb][cb] is coeff_z * coeff_c.
        #         Global index i = 2*K + rb, j = 2*L + cb.
        #         W index = j*4 + i.
        
        for t in range(7):
            u_dict = xf[t]
            v_dict = yf[t]
            
            w_dict = {}
            for rb in range(2):
                for cb in range(2):
                    coeff_z = z_coeffs[rb][cb][t]
                    if coeff_z == 0: continue
                    for K in range(2):
                        for L in range(2):
                            coeff_c = c_block_coeffs[K][L][l]
                            if coeff_c == 0: continue
                            total_coeff = coeff_z * coeff_c
                            i = 2*K + rb
                            j = 2*L + cb
                            w_idx = j*4 + i
                            add_to_vec(w_dict, w_idx, total_coeff)
            
            # Now populate U, V, W
            for idx, val in u_dict.items():
                U[idx][col] = val
            for idx, val in v_dict.items():
                V[idx][col] = val
            for idx, val in w_dict.items():
                W[idx][col] = val
            col += 1
            
    # Part 2: 4x4x1
    # A is 4x4, B is 4x1 (col 4), C is 4x1 (col 4)
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
