"""Program for matrix multiplication tensor decomposition of 5x5 by 5x5."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # Split indices 0..4 into 0..3 and 4.
    # Blocks:
    # (4,4,4) -> Rank 49 (Strassen)
    # (4,4,1) -> Rank 16 (Schoolbook)
    # (4,1,4) -> Rank 16 (Schoolbook)
    # (4,1,1) -> Rank 4 (Schoolbook)
    # (1,4,4) -> Rank 16 (Schoolbook)
    # (1,4,1) -> Rank 4 (Schoolbook)
    # (1,1,4) -> Rank 4 (Schoolbook)
    # (1,1,1) -> Rank 1 (Schoolbook)
    # Total Rank: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    total_rank = 110
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col_offset = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook_block(bn, bm, bp, i_offset, j_offset, k_offset):
        nonlocal col_offset
        rank_local = bn * bm * bp
        for li in range(bn):
            gi = i_offset + li
            for lj in range(bm):
                gj = j_offset + lj
                for lk in range(bp):
                    gk = k_offset + lk
                    U[gi * 5 + gj, col_offset] = 1
                    V[gj * 5 + gk, col_offset] = 1
                    W[gk * 5 + gi, col_offset] = 1
                    col_offset += 1
        return rank_local

    # Helper to add a Strassen block for 4x4x4
    def add_strassen_block_4x4x4():
        nonlocal col_offset
        
        # Indices for the 4x4 block are 0..3
        # A is 4x4, B is 4x4, C is 4x4
        # We use Strassen's algorithm on 2x2 blocks of A, B.
        
        # Strassen's 7 products for 2x2 matrix multiplication
        # P1 = A11 + A22
        # Q1 = B11 + B22
        # M1 = P1 * Q1
        #
        # P2 = A21 + A22
        # Q2 = B11
        # M2 = P2 * Q2
        #
        # P3 = A11
        # Q3 = B12 - B22
        # M3 = P3 * Q3
        #
        # P4 = A22
        # Q4 = B21 - B11
        # M4 = P4 * Q4
        #
        # P5 = A11 + A12
        # Q5 = B22
        # M5 = P5 * Q5
        #
        # P6 = A21 - A11
        # Q6 = B11 + B12
        # M6 = P6 * Q6
        #
        # P7 = A12 - A22
        # Q7 = B21 + B22
        # M7 = P7 * Q7
        
        # Result:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to decompose each Mk = Pk * Qk using Strassen's rank-7 decomposition for 2x2.
        # Let Pk be a 2x2 matrix of linear forms in A.
        # Let Qk be a 2x2 matrix of linear forms in B.
        
        # Define the 7 Strassen products for the 2x2 case.
        # For a 2x2 multiplication X * Y = Z:
        # S1 = (X11 + X22)(Y11 + Y22)
        # S2 = (X21 + X22)Y11
        # S3 = X11(Y12 - Y22)
        # S4 = X22(Y21 - Y11)
        # S5 = (X11 + X12)Y22
        # S6 = (X21 - X11)(Y11 + Y12)
        # S7 = (X12 - X22)(Y21 + Y22)
        
        # Z11 = S1 + S4 - S5 + S7
        # Z12 = S3 + S5
        # Z21 = S2 + S4
        # Z22 = S1 - S2 + S3 + S6
        
        # For each k in 1..7:
        #   Pk is 2x2, Qk is 2x2.
        #   We compute Mk = Pk * Qk using the 7 S terms.
        #   Each S term is a product of a linear form in A and a linear form in B.
        #   So each S term corresponds to one column in U, V, W.
        
        # We will iterate k from 1 to 7.
        for k in range(1, 8):
            # Define Pk and Qk as 2x2 matrices of dictionaries { (i,j): coeff }
            # Indices i,j are local to the 4x4 block (0..3)
            
            Pk = [ [ {} for _ in range(2) ] for _ in range(2) ]
            Qk = [ [ {} for _ in range(2) ] for _ in range(2) ]
            
            # Helper to add to Pk or Qk
            def add_to_matrix(M, r, c, i, j, coeff):
                if coeff != 0:
                    M[r][c][(i,j)] = M[r][c].get((i,j), 0) + coeff
            
            if k == 1:
                # P1 = A11 + A22
                # A11: (0,0), (0,1), (1,0), (1,1)
                # A22: (2,2), (2,3), (3,2), (3,3)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, c, 1) # A11
                        add_to_matrix(Pk, r, c, 2+r, 2+c, 1) # A22
                # Q1 = B11 + B22
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, r, c, 1) # B11
                        add_to_matrix(Qk, r, c, 2+r, 2+c, 1) # B22
            elif k == 2:
                # P2 = A21 + A22
                # A21: (2,0), (2,1), (3,0), (3,1)
                # A22: (2,2), (2,3), (3,2), (3,3)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, 2+r, c, 1) # A21
                        add_to_matrix(Pk, r, c, 2+r, 2+c, 1) # A22
                # Q2 = B11
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, r, c, 1)
            elif k == 3:
                # P3 = A11
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, c, 1)
                # Q3 = B12 - B22
                # B12: (0,2), (0,3), (1,2), (1,3)
                # B22: (2,2), (2,3), (3,2), (3,3)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, r, 2+c, 1) # B12
                        add_to_matrix(Qk, r, c, 2+r, 2+c, -1) # -B22
            elif k == 4:
                # P4 = A22
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, 2+r, 2+c, 1)
                # Q4 = B21 - B11
                # B21: (2,0), (2,1), (3,0), (3,1)
                # B11: (0,0), (0,1), (1,0), (1,1)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, 2+r, c, 1) # B21
                        add_to_matrix(Qk, r, c, r, c, -1) # -B11
            elif k == 5:
                # P5 = A11 + A12
                # A11: (0,0)..(1,1)
                # A12: (0,2)..(1,3)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, c, 1) # A11
                        add_to_matrix(Pk, r, c, r, 2+c, 1) # A12
                # Q5 = B22
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, 2+r, 2+c, 1)
            elif k == 6:
                # P6 = A21 - A11
                # A21: (2,0)..(3,1)
                # A11: (0,0)..(1,1)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, 2+r, c, 1) # A21
                        add_to_matrix(Pk, r, c, r, c, -1) # -A11
                # Q6 = B11 + B12
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, r, c, 1) # B11
                        add_to_matrix(Qk, r, c, r, 2+c, 1) # B12
            elif k == 7:
                # P7 = A12 - A22
                # A12: (0,2)..(1,3)
                # A22: (2,2)..(3,3)
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, 2+c, 1) # A12
                        add_to_matrix(Pk, r, c, 2+r, 2+c, -1) # -A22
                # Q7 = B21 + B22
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Qk, r, c, 2+r, c, 1) # B21
                        add_to_matrix(Qk, r, c, 2+r, 2+c, 1) # B22
            
            # Now compute Mk = Pk * Qk using Strassen's 7 products for 2x2.
            # We need to generate the 7 S terms.
            # S1 = (Pk11 + Pk22)(Qk11 + Qk22)
            # S2 = (Pk21 + Pk22)Qk11
            # S3 = Pk11(Qk12 - Qk22)
            # S4 = Pk22(Qk21 - Qk11)
            # S5 = (Pk11 + Pk12)Qk22
            # S6 = (Pk21 - Pk11)(Qk11 + Qk12)
            # S7 = (Pk12 - Pk22)(Qk21 + Qk22)
            
            # We will create a list of (U_dict, V_dict) for the 7 S terms.
            S_terms = []
            
            # Helper to combine two dictionaries of linear forms into a product term
            # U_dict: { (i,j): coeff } for A
            # V_dict: { (j,k): coeff } for B
            # This represents one rank-1 term.
            
            def combine_forms(P_form, Q_form):
                # P_form is a dict { (r,c): { (i,j): coeff } } ? No.
                # In the Strassen 2x2 decomposition, the inputs to the scalar multiplications
                # are linear combinations of the entries of Pk and Qk.
                # Pk entries are already linear forms in A.
                # So we need to sum the linear forms.
                
                # Let's represent the linear form for A as a dict { (i,j): coeff }
                # and for B as a dict { (j,k): coeff }
                
                # For S1: (Pk11 + Pk22) and (Qk11 + Qk22)
                # Pk11 is a dict. Pk22 is a dict.
                # Sum them into one dict.
                
                def sum_dicts(d1, d2):
                    res = {}
                    for key, val in d1.items():
                        res[key] = res.get(key, 0) + val
                    for key, val in d2.items():
                        res[key] = res.get(key, 0) + val
                    # Remove zeros
                    return {k: v for k, v in res.items() if v != 0}
                
                def sub_dicts(d1, d2):
                    res = {}
                    for key, val in d1.items():
                        res[key] = res.get(key, 0) + val
                    for key, val in d2.items():
                        res[key] = res.get(key, 0) - val
                    return {k: v for k, v in res.items() if v != 0}
                
                return sum_dicts, sub_dicts

            sum_dicts, sub_dicts = combine_forms(None, None)
            
            # Define the 7 S terms for the 2x2 multiplication Pk * Qk
            # Pk is 2x2, Qk is 2x2.
            # Indices 0,1 for the 2x2 blocks.
            
            # S1
            U1 = sum_dicts(Pk[0][0], Pk[1][1])
            V1 = sum_dicts(Qk[0][0], Qk[1][1])
            S_terms.append((U1, V1))
            
            # S2
            U2 = sum_dicts(Pk[1][0], Pk[1][1])
            V2 = Qk[0][0].copy()
            S_terms.append((U2, V2))
            
            # S3
            U3 = Pk[0][0].copy()
            V3 = sub_dicts(Qk[0][1], Qk[1][1])
            S_terms.append((U3, V3))
            
            # S4
            U4 = Pk[1][1].copy()
            V4 = sub_dicts(Qk[1][0], Qk[0][0])
            S_terms.append((U4, V4))
            
            # S5
            U5 = sum_dicts(Pk[0][0], Pk[0][1])
            V5 = Qk[1][1].copy()
            S_terms.append((U5, V5))
            
            # S6
            U6 = sub_dicts(Pk[1][0], Pk[0][0])
            V6 = sum_dicts(Qk[0][0], Qk[0][1])
            S_terms.append((U6, V6))
            
            # S7
            U7 = sub_dicts(Pk[0][1], Pk[1][1])
            V7 = sum_dicts(Qk[1][0], Qk[1][1])
            S_terms.append((U7, V7))
            
            # Now, for each S term, we have a U dict and V dict.
            # We need to determine the contribution to C.
            # Mk = Pk * Qk is a 2x2 matrix.
            # Mk[r][s] = sum_t Pk[r][t] Qk[t][s]
            # But we used Strassen to compute Mk.
            # The entries of Mk are linear combinations of S1..S7.
            
            # Mk11 = S1 + S4 - S5 + S7
            # Mk12 = S3 + S5
            # Mk21 = S2 + S4
            # Mk22 = S1 - S2 + S3 + S6
            
            # We need to map these back to the global C matrix.
            # The block Mk is one of the 7 Strassen products for the 4x4 multiplication.
            # We need to know which block of C each Mk contributes to.
            
            # From the 4x4 Strassen formulas:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for a given k, and a given entry (r,s) of Mk,
            # we need to find the coefficient in the C block formulas.
            
            # C_block[r][s] is a 2x2 block of C.
            # C11 corresponds to C[0:2, 0:2]
            # C12 corresponds to C[0:2, 2:4]
            # C21 corresponds to C[2:4, 0:2]
            # C22 corresponds to C[2:4, 2:4]
            
            # Let's define the contribution of Mk to C blocks.
            # contrib[k][c_r][c_c][r][s] = coeff
            # where c_r, c_c are block indices (0,1) for C.
            # r, s are local indices (0,1) within the 2x2 block.
            
            contrib = {}
            # C11 = M1 + M4 - M5 + M7
            for r in range(2):
                for s in range(2):
                    contrib[(0,0,r,s)] = {1: 1, 4: 1, 5: -1, 7: 1}
            # C12 = M3 + M5
            for r in range(2):
                for s in range(2):
                    contrib[(0,1,r,s)] = {3: 1, 5: 1}
            # C21 = M2 + M4
            for r in range(2):
                for s in range(2):
                    contrib[(1,0,r,s)] = {2: 1, 4: 1}
            # C22 = M1 - M2 + M3 + M6
            for r in range(2):
                for s in range(2):
                    contrib[(1,1,r,s)] = {1: 1, 2: -1, 3: 1, 6: 1}
            
            # Now, for each of the 7 S terms in Mk:
            # The S term contributes to Mk[r][s] with some coefficient.
            # Then Mk[r][s] contributes to C[br, bc, r, s] with some coefficient.
            
            # Mk11 = S1 + S4 - S5 + S7
            # Mk12 = S3 + S5
            # Mk21 = S2 + S4
            # Mk22 = S1 - S2 + S3 + S6
            
            mk_contrib = {
                (0,0): {1: 1, 4: 1, 5: -1, 7: 1},
                (0,1): {3: 1, 5: 1},
                (1,0): {2: 1, 4: 1},
                (1,1): {1: 1, 2: -1, 3: 1, 6: 1}
            }
            
            # For each S term index s_idx (1..7)
            for s_idx in range(1, 8):
                U_dict = S_terms[s_idx-1][0]
                V_dict = S_terms[s_idx-1][1]
                
                # Check if U_dict and V_dict are empty
                if not U_dict or not V_dict:
                    continue
                
                # Find the contributions of S_s_idx to Mk entries
                # We need to sum over all (mr, ms) in Mk where S_s_idx has a non-zero coeff
                # and all (br, bc) in C blocks where Mk[mr, ms] has a non-zero coeff.
                
                # For each (mr, ms) in mk_contrib:
                for (mr, ms), coeffs in mk_contrib.items():
                    if s_idx not in coeffs:
                        continue
                    mk_coeff = coeffs[s_idx]
                    
                    # For each (br, bc) in contrib:
                    for (br, bc, cr, cs), c_coeffs in contrib.items():
                        if cr != mr or cs != ms:
                            continue
                        if k not in c_coeffs:
                            continue
                        c_coeff = c_coeffs[k]
                        
                        total_coeff = mk_coeff * c_coeff
                        if total_coeff == 0:
                            continue
                        
                        # Global indices for C
                        # C is 4x4 in the block.
                        # Block (br, bc) is at offset 2*br, 2*bc.
                        # Local (cr, cs) is offset cr, cs.
                        gi = 2*br + cr
                        gj = 2*bc + cs
                        
                        # Add to U, V, W
                        for (ai, aj), coeff_a in U_dict.items():
                            U[ai * 5 + aj, col_offset] += coeff_a * total_coeff
                        for (bi, bj), coeff_b in V_dict.items():
                            V[bi * 5 + bj, col_offset] += coeff_b * total_coeff
                        W[gi * 5 + gj, col_offset] += total_coeff
                        
                        col_offset += 1
                        break # Only one column per S term? 
                        # Wait. Each S term is a single scalar multiplication.
                        # It contributes to multiple C entries?
                        # No, the tensor decomposition term is u otimes v otimes w.
                        # w is a vector. Here w has potentially multiple non-zero entries.
                        # So we should NOT increment col_offset inside the loop over C entries.
                        # We should accumulate the W vector for this column first.
                        
                        # Let's restructure.
                        # For a fixed S term, we determine the full W vector.
                        # Then we write U, V, W to the column and increment col_offset.
                        
                        pass # Placeholder, logic needs restructuring

        # The logic above was flawed. Let's restart the S term processing.
        # For each k (1..7):
        #   Compute S_terms (7 pairs of U_dict, V_dict).
        #   For each s_idx (1..7):
        #     Determine the W vector for this term.
        #     The term is S_s_idx.
        #     S_s_idx contributes to Mk[mr, ms] with coeff mk_coeff.
        #     Mk[mr, ms] contributes to C[br, bc, mr, ms] with coeff c_coeff.
        #     So W[global_index] += mk_coeff * c_coeff.
        #     We do this for all valid (mr, ms, br, bc).
        #     Then write U, V, W to the column.
        
        return None # Placeholder

    # Re-implementing add_strassen_block_4x4x4 with correct logic
    
    def add_strassen_block_4x4x4_correct():
        nonlocal col_offset
        
        # Strassen definitions for 4x4
        # k=1..7
        # Pk, Qk definitions as before
        
        # We will store the 7 S terms for each k.
        # Then for each S term, we compute the W vector.
        
        for k in range(1, 8):
            # Build Pk, Qk as before
            Pk = [ [ {} for _ in range(2) ] for _ in range(2) ]
            Qk = [ [ {} for _ in range(2) ] for _ in range(2) ]
            
            def add_to_matrix(M, r, c, i, j, coeff):
                if coeff != 0:
                    M[r][c][(i,j)] = M[r][c].get((i,j), 0) + coeff
            
            if k == 1:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, c, 1)
                        add_to_matrix(Pk, r, c, 2+r, 2+c, 1)
                        add_to_matrix(Qk, r, c, r, c, 1)
                        add_to_matrix(Qk, r, c, 2+r, 2+c, 1)
            elif k == 2:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, 2+r, c, 1)
                        add_to_matrix(Pk, r, c, 2+r, 2+c, 1)
                        add_to_matrix(Qk, r, c, r, c, 1)
            elif k == 3:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, c, 1)
                        add_to_matrix(Qk, r, c, r, 2+c, 1)
                        add_to_matrix(Qk, r, c, 2+r, 2+c, -1)
            elif k == 4:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, 2+r, 2+c, 1)
                        add_to_matrix(Qk, r, c, 2+r, c, 1)
                        add_to_matrix(Qk, r, c, r, c, -1)
            elif k == 5:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, c, 1)
                        add_to_matrix(Pk, r, c, r, 2+c, 1)
                        add_to_matrix(Qk, r, c, 2+r, 2+c, 1)
            elif k == 6:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, 2+r, c, 1)
                        add_to_matrix(Pk, r, c, r, c, -1)
                        add_to_matrix(Qk, r, c, r, c, 1)
                        add_to_matrix(Qk, r, c, r, 2+c, 1)
            elif k == 7:
                for r in range(2):
                    for c in range(2):
                        add_to_matrix(Pk, r, c, r, 2+c, 1)
                        add_to_matrix(Pk, r, c, 2+r, 2+c, -1)
                        add_to_matrix(Qk, r, c, 2+r, c, 1)
                        add_to_matrix(Qk, r, c, 2+r, 2+c, 1)
            
            # Compute 7 S terms for Pk * Qk
            def sum_dicts(d1, d2):
                res = {}
                for key, val in d1.items():
                    res[key] = res.get(key, 0) + val
                for key, val in d2.items():
                    res[key] = res.get(key, 0) + val
                return {k: v for k, v in res.items() if v != 0}
            
            def sub_dicts(d1, d2):
                res = {}
                for key, val in d1.items():
                    res[key] = res.get(key, 0) + val
                for key, val in d2.items():
                    res[key] = res.get(key, 0) - val
                return {k: v for k, v in res.items() if v != 0}
            
            S_terms = []
            S_terms.append((sum_dicts(Pk[0][0], Pk[1][1]), sum_dicts(Qk[0][0], Qk[1][1]))) # S1
            S_terms.append((sum_dicts(Pk[1][0], Pk[1][1]), Qk[0][0].copy())) # S2
            S_terms.append((Pk[0][0].copy(), sub_dicts(Qk[0][1], Qk[1][1]))) # S3
            S_terms.append((Pk[1][1].copy(), sub_dicts(Qk[1][0], Qk[0][0]))) # S4
            S_terms.append((sum_dicts(Pk[0][0], Pk[0][1]), Qk[1][1].copy())) # S5
            S_terms.append((sub_dicts(Pk[1][0], Pk[0][0]), sum_dicts(Qk[0][0], Qk[0][1]))) # S6
            S_terms.append((sub_dicts(Pk[0][1], Pk[1][1]), sum_dicts(Qk[1][0], Qk[1][1]))) # S7
            
            # Contributions of S terms to Mk
            mk_contrib = {
                (0,0): {1: 1, 4: 1, 5: -1, 7: 1},
                (0,1): {3: 1, 5: 1},
                (1,0): {2: 1, 4: 1},
                (1,1): {1: 1, 2: -1, 3: 1, 6: 1}
            }
            
            # Contributions of Mk to C blocks
            c_contrib = {
                (0,0): {1: 1, 4: 1, 5: -1, 7: 1}, # C11
                (0,1): {3: 1, 5: 1}, # C12
                (1,0): {2: 1, 4: 1}, # C21
                (1,1): {1: 1, 2: -1, 3: 1, 6: 1} # C22
            }
            
            # For each S term
            for s_idx in range(1, 8):
                U_dict, V_dict = S_terms[s_idx-1]
                if not U_dict or not V_dict:
                    continue
                
                W_vec = {}
                
                # Iterate over Mk entries
                for (mr, ms), coeffs_m in mk_contrib.items():
                    if s_idx not in coeffs_m:
                        continue
                    mk_coeff = coeffs_m[s_idx]
                    
                    # Iterate over C blocks
                    for (br, bc), coeffs_c in c_contrib.items():
                        if k not in coeffs_c:
                            continue
                        c_coeff = coeffs_c[k]
                        
                        total_coeff = mk_coeff * c_coeff
                        if total_coeff == 0:
                            continue
                        
                        gi = 2*br + mr
                        gj = 2*bc + ms
                        
                        W_vec[(gi, gj)] = W_vec.get((gi, gj), 0) + total_coeff
                
                if not W_vec:
                    continue
                
                # Write to U, V, W
                for (ai, aj), coeff_a in U_dict.items():
                    U[ai * 5 + aj, col_offset] = coeff_a
                for (bi, bj), coeff_b in V_dict.items():
                    V[bi * 5 + bj, col_offset] = coeff_b
                for (gi, gj), coeff_w in W_vec.items():
                    W[gi * 5 + gj, col_offset] = coeff_w
                
                col_offset += 1
        
        return col_offset

    # Execute blocks
    # 4x4x4 at (0,0,0)
    col_offset += add_strassen_block_4x4x4_correct()
    
    # 4x4x1 at (0,0,4)
    col_offset += add_schoolbook_block(4, 4, 1, 0, 0, 4)
    
    # 4x1x4 at (0,4,0)
    col_offset += add_schoolbook_block(4, 1, 4, 0, 4, 0)
    
    # 4x1x1 at (0,4,4)
    col_offset += add_schoolbook_block(4, 1, 1, 0, 4, 4)
    
    # 1x4x4 at (4,0,0)
    col_offset += add_schoolbook_block(1, 4, 4, 4, 0, 0)
    
    # 1x4x1 at (4,0,4)
    col_offset += add_schoolbook_block(1, 4, 1, 4, 0, 4)
    
    # 1x1x4 at (4,4,0)
    col_offset += add_schoolbook_block(1, 1, 4, 4, 4, 0)
    
    # 1x1x1 at (4,4,4)
    col_offset += add_schoolbook_block(1, 1, 1, 4, 4, 4)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
