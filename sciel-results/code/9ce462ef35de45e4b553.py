"""Matrix multiplication 4,4,8: Strassen-based decomposition (rank 98)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses Strassen at two levels:
    - Split the 4x8 output into two 4x4 blocks
    - Each 4x4x4 multiplication uses Strassen: 7 products of 2x2x2
    - Each 2x2x2 product uses Strassen: 7 scalar multiplications
    - Total rank = 2 * 7 * 7 = 98
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build the decomposition programmatically
    # For each of the 2 output blocks (left 4x4, right 4x4):
    #   For each of 7 Strassen 4x4x4 terms:
    #     For each of 7 Strassen 2x2x2 terms:
    #       Create one rank-1 term
    
    rank = 2 * 7 * 7  # 98
    
    # U: shape (n*m, rank) = (16, 98)
    # V: shape (m*p, rank) = (32, 98)  
    # W: shape (p*n, rank) = (32, 98)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 decomposition coefficients
    # For multiplying P (2x2) * Q (2x2) = R (2x2):
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11)Q00
    # M3 = P00(Q10-Q11)
    # M4 = P11(Q01-Q00)
    # M5 = (P00+P01)Q11
    # M6 = (P10-P00)(Q00+Q10)
    # M7 = (P01-P11)(Q01+Q11)
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    
    # For each of the 7 M terms, define:
    # - p_coeffs: coefficients for P entries [P00, P01, P10, P11]
    # - q_coeffs: coefficients for Q entries [Q00, Q01, Q10, Q11]
    # - r_coeffs: coefficients for R entries [R00, R01, R10, R11]
    
    # P indices: 0=P00, 1=P01, 2=P10, 3=P11
    # Q indices: 0=Q00, 1=Q01, 2=Q10, 3=Q11
    # R indices: 0=R00, 1=R01, 2=R10, 3=R11
    
    strassen_2x2x2 = [
        # (p_coeffs, q_coeffs, r_coeffs)
        # M1 = (P00+P11)(Q00+Q11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # contributes to R00 and R11
        # M2 = (P10+P11)Q00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # contributes to R10
        # M3 = P00(Q10-Q11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # contributes to R01
        # M4 = P11(Q01-Q00)
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),  # contributes to R00 and R10
        # M5 = (P00+P01)Q11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),  # contributes to R01
        # M6 = (P10-P00)(Q00+Q10)
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),  # contributes to R11
        # M7 = (P01-P11)(Q01+Q11)
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),  # contributes to R00
    ]
    
    # Strassen 4x4x4 decomposition
    # Split A (4x4) into 2x2 blocks: A00, A01, A10, A11
    # Split B (4x4) into 2x2 blocks: B00, B01, B10, B11
    # 
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B10)
    # S7 = (A01-A11)(B01+B11)
    #
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S6
    
    # For each S term, we need to specify:
    # - Which A blocks and with what coefficients
    # - Which B blocks and with what coefficients
    # - Which C blocks and with what coefficients
    
    # A block indices: 0=A00 (rows 0-1, cols 0-1), 1=A01 (rows 0-1, cols 2-3), 
    #                  2=A10 (rows 2-3, cols 0-1), 3=A11 (rows 2-3, cols 2-3)
    # B block indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C block indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_4x4x4 = [
        # (a_block_coeffs, b_block_coeffs, c_block_coeffs)
        # S1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # contributes to C00 and C11
        # S2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # contributes to C10
        # S3 = A00(B10-B11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # contributes to C01
        # S4 = A11(B01-B00)
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),  # contributes to C00 and C10
        # S5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),  # contributes to C01
        # S6 = (A10-A00)(B00+B10)
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),  # contributes to C11
        # S7 = (A01-A11)(B01+B11)
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),  # contributes to C00
    ]
    
    col = 0
    for block in range(2):  # Left (0) and right (1) 4x4 output blocks
        # Output column offset: block 0 -> columns 0-3, block 1 -> columns 4-7
        out_col_offset = block * 4
        
        for s in range(7):  # Strassen 4x4x4 term
            a_coeffs, b_coeffs, c_coeffs = strassen_4x4x4[s]
            
            for t in range(7):  # Strassen 2x2x2 term
                p_coeffs, q_coeffs, r_coeffs = strassen_2x2x2[t]
                
                # For each non-zero coefficient in the product, add a rank-1 term
                # We need to expand the product of linear forms
                
                # For each A block and B block combination
                for a_blk in range(4):
                    if a_coeffs[a_blk] == 0:
                        continue
                    a_coeff = a_coeffs[a_blk]
                    
                    for b_blk in range(4):
                        if b_coeffs[b_blk] == 0:
                            continue
                        b_coeff = b_coeffs[b_blk]
                        
                        # For each 2x2x2 term, expand the product
                        for pi in range(4):
                            if p_coeffs[pi] == 0:
                                continue
                            p_coeff = p_coeffs[pi]
                            
                            for qi in range(4):
                                if q_coeffs[qi] == 0:
                                    continue
                                q_coeff = q_coeffs[qi]
                                
                                # Now we have a scalar multiplication:
                                # A_entry * B_entry -> contributes to C entries
                                
                                # Determine which A entry this is
                                # A block a_blk has local 2x2 indices
                                # a_blk 0: rows 0-1, cols 0-1
                                # a_blk 1: rows 0-1, cols 2-3
                                # a_blk 2: rows 2-3, cols 0-1
                                # a_blk 3: rows 2-3, cols 2-3
                                a_row_block = a_blk // 2  # 0 for blocks 0,1; 1 for blocks 2,3
                                a_col_block = a_blk % 2   # 0 for blocks 0,2; 1 for blocks 1,3
                                
                                # pi: 0=P00, 1=P01, 2=P10, 3=P11
                                pi_row = pi // 2
                                pi_col = pi % 2
                                
                                a_global_row = a_row_block * 2 + pi_row
                                a_global_col = a_col_block * 2 + pi_col
                                
                                # Determine which B entry this is
                                # B block b_blk has local 2x2 indices
                                b_row_block = b_blk // 2
                                b_col_block = b_blk % 2
                                
                                qi_row = qi // 2
                                qi_col = qi % 2
                                
                                b_global_row = b_row_block * 2 + qi_row
                                b_global_col = b_col_block * 2 + qi_col
                                
                                # The product is:
                                # A[a_global_row, a_global_col] * B[b_global_row, b_global_col]
                                # with coefficient a_coeff * p_coeff * b_coeff * q_coeff
                                
                                # This contributes to C entries via r_coeffs
                                # The 2x2x2 result has entries that map to C blocks
                                
                                # For each C block this term contributes to
                                for c_blk in range(4):
                                    if c_coeffs[c_blk] == 0:
                                        continue
                                    c_coeff = c_coeffs[c_blk]
                                    
                                    # The 2x2x2 result entry ri maps to a specific C entry
                                    # ri: 0=R00, 1=R01, 2=R10, 3=R11
                                    ri_row = ri // 2 if False else (r_coeffs.index(1) if False else 0)
                                    
                                    # Actually, we need to be more careful.
                                    # The r_coeffs tell us which R entries this term contributes to.
                                    # But each R entry maps to a specific C entry within the C block.
                                    
                                    # Let me restructure: for each of the 4 R entries,
                                    # check if this term contributes to it.
                                    
                                    for ri in range(4):
                                        if r_coeffs[ri] == 0:
                                            continue
                                        r_coeff = r_coeffs[ri]
                                        
                                        # ri maps to local position in C block c_blk
                                        ri_row = ri // 2
                                        ri_col = ri % 2
                                        
                                        # C block c_blk has global position:
                                        c_row_block = c_blk // 2
                                        c_col_block = c_blk % 2
                                        
                                        c_global_row = c_row_block * 2 + ri_row
                                        c_global_col = c_col_block * 2 + ri_col
                                        
                                        # Add this rank-1 term
                                        total_coeff = a_coeff * p_coeff * b_coeff * q_coeff * c_coeff * r_coeff
                                        
                                        if total_coeff != 0:
                                            # U index: a_global_row * m + a_global_col
                                            u_idx = a_global_row * m + a_global_col
                                            # V index: b_global_row * p + b_global_col
                                            v_idx = b_global_row * p + b_global_col
                                            # W index: (out_col_offset + c_global_col) * n + c_global_row
                                            w_idx = (out_col_offset + c_global_col) * n + c_global_row
                                            
                                            U[u_idx, col] += total_coeff
                                            V[v_idx, col] += total_coeff
                                            W[w_idx, col] += total_coeff
                                            
                        col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
