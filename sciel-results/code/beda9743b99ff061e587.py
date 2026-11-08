import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # We use recursive Strassen on 8x8 matrices (padded from 6x6)
    # Rank = 7^3 = 343
    
    def get_strassen_terms(size):
        if size == 1:
            return [({(0,0): 1}, {(0,0): 1}, {(0,0): 1})]
        
        half = size // 2
        sub_terms = get_strassen_terms(half)
        
        # Strassen formulas
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Output coefficients
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_defs = [
            # (U_blocks, V_blocks, W_coeff_dict)
            # U_blocks: {(block_row, block_col): coeff}
            # V_blocks: {(block_row, block_col): coeff}
            # W_coeff_dict: {(out_row, out_col): coeff}
            
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # M2
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # M3
            ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}) # M7
        ]
        
        terms = []
        for u_blocks, v_blocks, w_coeffs in strassen_defs:
            for (ur, uc), u_coeff in u_blocks.items():
                for (vr, vc), v_coeff in v_blocks.items():
                    # This pair (ur,uc) and (vr,vc) defines a sub-multiplication
                    # A_sub = A[ur*half : (ur+1)*half, uc*half : (uc+1)*half]
                    # B_sub = B[vr*half : (vr+1)*half, vc*half : (vc+1)*half]
                    
                    for sub_u, sub_v, sub_w in sub_terms:
                        new_u = {}
                        new_v = {}
                        new_w = {}
                        
                        # Expand U
                        for (li, lj), val in sub_u.items():
                            gi = ur * half + li
                            gj = uc * half + lj
                            new_u[(gi, gj)] = new_u.get((gi, gj), 0) + u_coeff * val
                        
                        # Expand V
                        for (li, lj), val in sub_v.items():
                            gi = vr * half + li
                            gj = vc * half + lj
                            new_v[(gi, gj)] = new_v.get((gi, gj), 0) + v_coeff * val
                        
                        # Expand W
                        # The output of the sub-multiplication is a matrix of size half x half
                        # It contributes to C at positions determined by w_coeffs
                        # w_coeffs maps (out_block_r, out_block_c) to coeff
                        # The local output indices are (lk, li) ? No, W is (k, i)
                        # In the tensor definition, W[k*n+i].
                        # Sub-W is (lk, li) where lk, li are local indices in 0..half-1
                        # Global k = out_block_r * half + lk
                        # Global i = out_block_c * half + li
                        # Wait, the standard Strassen output mapping:
                        # C00 gets M1, M4, -M5, M7
                        # The sub-multiplication result is added to C at the block positions.
                        
                        for (wr, wc), w_coeff in w_coeffs.items():
                            for (lk, li), val in sub_w.items():
                                gi = wr * half + li
                                gj = wc * half + lk
                                # Note: W index is (k, i). In matrix C, C[i, k]?
                                # The task says: "output factor follows the official trace convention W[k*n+i,r]"
                                # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k]
                                # So W index corresponds to (k, i) ?
                                # Let's check the seed program:
                                # W[k * n + i][col_idx]
                                # And U[r * m + c] where r=i, c=j.
                                # V[r * p + c] where r=j, c=k.
                                # So U[i, j], V[j, k], W[k, i].
                                # This matches C[i, k] = sum A[i, j] V[j, k] W[k, i].
                                # So W indices are (k, i).
                                
                                # In my Strassen derivation:
                                # C is the result matrix.
                                # C[wr, wc] is the block.
                                # So global row i = wr * half + li
                                # global col k = wc * half + lk
                                # So W index is (k, i) = (wc * half + lk, wr * half + li)
                                
                                gk = wc * half + lk
                                gi = wr * half + li
                                new_w[(gk, gi)] = new_w.get((gk, gi), 0) + w_coeff * val

                        terms.append((new_u, new_v, new_w))
        return terms

    terms = get_strassen_terms(8)
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            if i < n and j < m:
                U[i * m + j][r_idx] = val
        for (j, k), val in v_dict.items():
            if j < m and k < p:
                V[j * p + k][r_idx] = val
        for (k, i), val in w_dict.items():
            if k < p and i < n:
                W[k * n + i][r_idx] = val
                
    return {"U": U, "V": V, "W": W}
