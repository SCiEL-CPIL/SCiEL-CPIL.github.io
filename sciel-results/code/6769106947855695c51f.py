import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 outer Strassen steps
    # Each step k defines X and Y as 2x2 matrices of linear forms
    # X and Y are 2x2, entries are dicts mapping (block_row, block_col, local_row, local_col) to coeff
    # Or simpler: just list of (A_idx, coeff) and (B_idx, coeff)
    
    # Helper to get A index from block coords
    def a_idx(br, bc, lr, lc):
        return (br * 2 + lr) * 4 + (bc * 2 + lc)
        
    def b_idx(br, bc, lr, lc):
        return (br * 2 + lr) * 4 + (bc * 2 + lc)

    # Define the 7 pairs (X, Y) for the outer Strassen
    # X and Y are 2x2 arrays of lists of (idx, coeff)
    
    steps = []
    
    # M1 = (A00 + A11) (B00 + B11)
    x1 = [
        [(a_idx(0,0,0,0), 1), (a_idx(1,1,0,0), 1)], # x00
        [(a_idx(0,0,0,1), 1), (a_idx(1,1,0,1), 1)], # x01
        [(a_idx(0,0,1,0), 1), (a_idx(1,1,1,0), 1)], # x10
        [(a_idx(0,0,1,1), 1), (a_idx(1,1,1,1), 1)]  # x11
    ]
    y1 = [
        [(b_idx(0,0,0,0), 1), (b_idx(1,1,0,0), 1)], # y00
        [(b_idx(0,0,0,1), 1), (b_idx(1,1,0,1), 1)], # y01
        [(b_idx(0,0,1,0), 1), (b_idx(1,1,1,0), 1)], # y10
        [(b_idx(0,0,1,1), 1), (b_idx(1,1,1,1), 1)]  # y11
    ]
    steps.append((x1, y1))
    
    # M2 = (A10 + A11) B00
    x2 = [
        [(a_idx(1,0,0,0), 1), (a_idx(1,1,0,0), 1)],
        [(a_idx(1,0,0,1), 1), (a_idx(1,1,0,1), 1)],
        [(a_idx(1,0,1,0), 1), (a_idx(1,1,1,0), 1)],
        [(a_idx(1,0,1,1), 1), (a_idx(1,1,1,1), 1)]
    ]
    y2 = [
        [(b_idx(0,0,0,0), 1)],
        [(b_idx(0,0,0,1), 1)],
        [(b_idx(0,0,1,0), 1)],
        [(b_idx(0,0,1,1), 1)]
    ]
    steps.append((x2, y2))
    
    # M3 = A00 (B01 - B11)
    x3 = [
        [(a_idx(0,0,0,0), 1)],
        [(a_idx(0,0,0,1), 1)],
        [(a_idx(0,0,1,0), 1)],
        [(a_idx(0,0,1,1), 1)]
    ]
    y3 = [
        [(b_idx(0,1,0,0), 1), (b_idx(1,1,0,0), -1)],
        [(b_idx(0,1,0,1), 1), (b_idx(1,1,0,1), -1)],
        [(b_idx(0,1,1,0), 1), (b_idx(1,1,1,0), -1)],
        [(b_idx(0,1,1,1), 1), (b_idx(1,1,1,1), -1)]
    ]
    steps.append((x3, y3))
    
    # M4 = A11 (B10 - B00)
    x4 = [
        [(a_idx(1,1,0,0), 1)],
        [(a_idx(1,1,0,1), 1)],
        [(a_idx(1,1,1,0), 1)],
        [(a_idx(1,1,1,1), 1)]
    ]
    y4 = [
        [(b_idx(1,0,0,0), 1), (b_idx(0,0,0,0), -1)],
        [(b_idx(1,0,0,1), 1), (b_idx(0,0,0,1), -1)],
        [(b_idx(1,0,1,0), 1), (b_idx(0,0,1,0), -1)],
        [(b_idx(1,0,1,1), 1), (b_idx(0,0,1,1), -1)]
    ]
    steps.append((x4, y4))
    
    # M5 = (A00 + A01) B11
    x5 = [
        [(a_idx(0,0,0,0), 1), (a_idx(0,1,0,0), 1)],
        [(a_idx(0,0,0,1), 1), (a_idx(0,1,0,1), 1)],
        [(a_idx(0,0,1,0), 1), (a_idx(0,1,1,0), 1)],
        [(a_idx(0,0,1,1), 1), (a_idx(0,1,1,1), 1)]
    ]
    y5 = [
        [(b_idx(1,1,0,0), 1)],
        [(b_idx(1,1,0,1), 1)],
        [(b_idx(1,1,1,0), 1)],
        [(b_idx(1,1,1,1), 1)]
    ]
    steps.append((x5, y5))
    
    # M6 = (A10 - A00) (B00 + B01)
    x6 = [
        [(a_idx(1,0,0,0), 1), (a_idx(0,0,0,0), -1)],
        [(a_idx(1,0,0,1), 1), (a_idx(0,0,0,1), -1)],
        [(a_idx(1,0,1,0), 1), (a_idx(0,0,1,0), -1)],
        [(a_idx(1,0,1,1), 1), (a_idx(0,0,1,1), -1)]
    ]
    y6 = [
        [(b_idx(0,0,0,0), 1), (b_idx(0,1,0,0), 1)],
        [(b_idx(0,0,0,1), 1), (b_idx(0,1,0,1), 1)],
        [(b_idx(0,0,1,0), 1), (b_idx(0,1,1,0), 1)],
        [(b_idx(0,0,1,1), 1), (b_idx(0,1,1,1), 1)]
    ]
    steps.append((x6, y6))
    
    # M7 = (A01 - A11) (B10 + B11)
    x7 = [
        [(a_idx(0,1,0,0), 1), (a_idx(1,1,0,0), -1)],
        [(a_idx(0,1,0,1), 1), (a_idx(1,1,0,1), -1)],
        [(a_idx(0,1,1,0), 1), (a_idx(1,1,1,0), -1)],
        [(a_idx(0,1,1,1), 1), (a_idx(1,1,1,1), -1)]
    ]
    y7 = [
        [(b_idx(1,0,0,0), 1), (b_idx(1,1,0,0), 1)],
        [(b_idx(1,0,0,1), 1), (b_idx(1,1,0,1), 1)],
        [(b_idx(1,0,1,0), 1), (b_idx(1,1,1,0), 1)],
        [(b_idx(1,0,1,1), 1), (b_idx(1,1,1,1), 1)]
    ]
    steps.append((x7, y7))
    
    # Inner Strassen coefficients for Z = XY
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    inner_coeffs = [
        # S1
        {0: 1, 1: 0, 2: 0, 3: -1}, # Z00, Z01, Z10, Z11 coeffs? No.
        # Let's store coeffs for Z00, Z01, Z10, Z11 for each Sj
    ]
    
    # Correctly:
    # S1: Z00+=1, Z11+=1
    # S2: Z10+=1, Z11-=1
    # S3: Z01+=1, Z11+=1
    # S4: Z00+=1, Z10+=1
    # S5: Z01+=1, Z00-=1
    # S6: Z11+=1
    # S7: Z00+=1
    
    z_coeffs = [
        [1, 0, 0, 1], # S1 -> Z00, Z01, Z10, Z11
        [0, 0, 1, -1], # S2
        [0, 1, 0, 1], # S3
        [1, 0, 1, 0], # S4
        [-1, 1, 0, 0], # S5
        [0, 0, 0, 1], # S6
        [1, 0, 0, 0]  # S7
    ]
    
    # Outer coeffs for C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    outer_coeffs = [
        [1, 0, 0, 0], # M1 -> C00, C01, C10, C11
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Wait, check M7 in C00.
    # C00 = M1 + M4 - M5 + M7. Yes.
    # M7 coeff in C00 is 1.
    
    for k in range(7):
        x, y = steps[k]
        for j in range(7):
            r = 7 * k + j
            # Determine L_A and L_B for Sj
            # x is 2x2 list of lists of (idx, coeff)
            # y is 2x2 list of lists of (idx, coeff)
            
            la = []
            lb = []
            
            if j == 0: # S1
                # (x00+x11)(y00+y11)
                for idx, c in x[0]: la.append((idx, c))
                for idx, c in x[3]: la.append((idx, c))
                for idx, c in y[0]: lb.append((idx, c))
                for idx, c in y[3]: lb.append((idx, c))
            elif j == 1: # S2
                # (x10+x11)y00
                for idx, c in x[2]: la.append((idx, c))
                for idx, c in x[3]: la.append((idx, c))
                for idx, c in y[0]: lb.append((idx, c))
            elif j == 2: # S3
                # x00(y01-y11)
                for idx, c in x[0]: la.append((idx, c))
                for idx, c in y[1]: lb.append((idx, c))
                for idx, c in y[3]: lb.append((idx, -c))
            elif j == 3: # S4
                # x11(y10-y00)
                for idx, c in x[3]: la.append((idx, c))
                for idx, c in y[2]: lb.append((idx, c))
                for idx, c in y[0]: lb.append((idx, -c))
            elif j == 4: # S5
                # (x00+x01)y11
                for idx, c in x[0]: la.append((idx, c))
                for idx, c in x[1]: la.append((idx, c))
                for idx, c in y[3]: lb.append((idx, c))
            elif j == 5: # S6
                # (x10-x00)(y00+y01)
                for idx, c in x[2]: la.append((idx, c))
                for idx, c in x[0]: la.append((idx, -c))
                for idx, c in y[0]: lb.append((idx, c))
                for idx, c in y[1]: lb.append((idx, c))
            elif j == 6: # S7
                # (x01-x11)(y10+y11)
                for idx, c in x[1]: la.append((idx, c))
                for idx, c in x[3]: la.append((idx, -c))
                for idx, c in y[2]: lb.append((idx, c))
                for idx, c in y[3]: lb.append((idx, c))
            
            for idx, c in la:
                U[idx][r] += c
            for idx, c in lb:
                V[idx][r] += c
            
            # Update W
            # z_coeffs[j] gives coeffs for Z00, Z01, Z10, Z11
            # outer_coeffs[k] gives coeffs for C00, C01, C10, C11
            # We need to combine them.
            # Contribution to C_block_b from M_k is outer_coeffs[k][b]
            # Contribution to Z_block_a from S_j is z_coeffs[j][a]
            # Total coeff for C_block_b entry from S_j is sum over a?
            # No, Z_a is a 2x2 matrix. C_b is a 2x2 matrix.
            # C_b = sum_k outer_k_b M_k.
            # M_k = Z^{(k)}.
            # So C_b = sum_k outer_k_b Z^{(k)}.
            # Z^{(k)}_a = sum_j z_coeff_j_a S_{k,j}.
            # So C_b = sum_k outer_k_b sum_j z_coeff_j_a S_{k,j}.
            # For fixed k, j, the scalar S_{k,j} contributes to C_b with coefficient outer_k_b * z_coeff_j_a?
            # No, Z_a is the a-th block of Z.
            # Z = [[Z00, Z01], [Z10, Z11]].
            # C = [[C00, C01], [C10, C11]].
            # C00 = M1_00 + ...
            # M_k is a 2x2 matrix.
            # M_k[0,0] is Z00.
            # So C00[0,0] gets contribution from M_k[0,0] with coeff outer_k_00.
            # M_k[0,0] gets contribution from S_j with coeff z_coeff_j_00.
            # So C00[0,0] gets S_j * outer_k_00 * z_coeff_j_00.
            
            # Indices in W:
            # C00: k=0,1 and i=0,1?
            # W index = k_col * 4 + i_row.
            # C[i, k].
            # C00 block: i in 0,1, k in 0,1.
            # W indices:
            # C[0,0] -> W[0*4+0] = W[0]
            # C[0,1] -> W[1*4+0] = W[4]
            # C[1,0] -> W[0*4+1] = W[1]
            # C[1,1] -> W[1*4+1] = W[5]
            
            # C01 block: i in 0,1, k in 2,3.
            # C[0,2] -> W[2*4+0] = W[8]
            # C[0,3] -> W[3*4+0] = W[12]
            # C[1,2] -> W[2*4+1] = W[9]
            # C[1,3] -> W[3*4+1] = W[13]
            
            # C10 block: i in 2,3, k in 0,1.
            # C[2,0] -> W[0*4+2] = W[2]
            # C[2,1] -> W[1*4+2] = W[6]
            # C[3,0] -> W[0*4+3] = W[3]
            # C[3,1] -> W[1*4+3] = W[7]
            
            # C11 block: i in 2,3, k in 2,3.
            # C[2,2] -> W[2*4+2] = W[10]
            # C[2,3] -> W[3*4+2] = W[14]
            # C[3,2] -> W[2*4+3] = W[11]
            # C[3,3] -> W[3*4+3] = W[15]
            
            # Map block (b_row, b_col) and local (l_row, l_col) to W index
            def w_idx(br, bc, lr, lc):
                i = br * 2 + lr
                k = bc * 2 + lc
                return k * 4 + i
            
            # z_coeffs[j] is [Z00, Z01, Z10, Z11]
            # Z00 is block (0,0)
            # Z01 is block (0,1)
            # Z10 is block (1,0)
            # Z11 is block (1,1)
            
            for a in range(4):
                # a=0: Z00 (br=0, bc=0)
                # a=1: Z01 (br=0, bc=1)
                # a=2: Z10 (br=1, bc=0)
                # a=3: Z11 (br=1, bc=1)
                z_c = z_coeffs[j][a]
                if z_c == 0: continue
                
                # Determine block row/col from a
                if a == 0: br, bc = 0, 0
                elif a == 1: br, bc = 0, 1
                elif a == 2: br, bc = 1, 0
                else: br, bc = 1, 1
                
                # For each local entry in this 2x2 block
                for lr in range(2):
                    for lc in range(2):
                        # This entry of Z_a corresponds to C[br*2+lr, bc*2+lc]
                        # Which is C[i, k] with i=br*2+lr, k=bc*2+lc
                        # W index is k*4 + i
                        w_i = w_idx(br, bc, lr, lc)
                        
                        # This Z_a entry is the (lr, lc) entry of the matrix M_k
                        # M_k is the k-th outer product result.
                        # C_block_b gets M_k with coeff outer_coeffs[k][b]
                        # Here b is the block index in C.
                        # Z_a is the a-th block of M_k.
                        # So M_k's a-th block contributes to C's a-th block?
                        # No. M_k is a 2x2 matrix of scalars?
                        # No, M_k is the result of multiplying two 2x2 matrices X, Y.
                        # So M_k is a 2x2 matrix of scalars.
                        # Wait. X and Y are 2x2 matrices of LINEAR FORMS.
                        # So M_k = X Y is a 2x2 matrix of BILINEAR FORMS.
                        # Each entry of M_k is a sum of products.
                        # Strassen decomposes this 2x2 matrix multiplication into 7 scalar products.
                        # So M_k[0,0] is a linear combination of S1..S7.
                        # M_k[0,1] is a linear combination of S1..S7.
                        # etc.
                        
                        # So, for a fixed k and j, S_{k,j} contributes to M_k[a_row, a_col] with coeff z_coeffs[j][a]?
                        # My z_coeffs definition:
                        # z_coeffs[j] = [coeff for Z00, Z01, Z10, Z11]
                        # Z00 is M_k[0,0].
                        # Z01 is M_k[0,1].
                        # Z10 is M_k[1,0].
                        # Z11 is M_k[1,1].
                        
                        # So S_{k,j} contributes to M_k[br, bc] with coeff z_c.
                        
                        # Now, C_block_b = sum_k outer_k_b M_k.
                        # So C[br*2+lr, bc*2+lc] gets contribution from M_k[br, bc] with coeff outer_coeffs[k][b].
                        # Here b is the block index in C.
                        # The block at (br, bc) in M_k corresponds to the block at (br, bc) in C.
                        # So b = a.
                        
                        outer_c = outer_coeffs[k][a]
                        total_c = z_c * outer_c
                        if total_c != 0:
                            W[w_i][r] += total_c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
