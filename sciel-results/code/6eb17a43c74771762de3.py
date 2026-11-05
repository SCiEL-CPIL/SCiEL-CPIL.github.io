"""Program for 4x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return 7 (A_mask, B_mask, C_mask) for 2x2 matrix multiplication.
    Each mask is a dict: (row, col) -> coefficient."""
    A00, A01, A10, A11 = (0,0), (0,1), (1,0), (1,1)
    B00, B01, B10, B11 = (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    terms = []
    # M1
    terms.append(( {A00:1, A11:1}, {B00:1, B11:1} ))
    # M2
    terms.append(( {A10:1, A11:1}, {B00:1} ))
    # M3
    terms.append(( {A00:1}, {B01:1, B11:-1} ))
    # M4
    terms.append(( {A11:1}, {B10:1, B00:-1} ))
    # M5
    terms.append(( {A00:1, A01:1}, {B11:1} ))
    # M6
    terms.append(( {A10:1, A00:-1}, {B00:1, B01:1} ))
    # M7
    terms.append(( {A01:1, A11:-1}, {B10:1, B11:1} ))
    
    # C combination: C_ij = sum_q coef[q][ij] * M_q
    C_coefs = []
    # For each term q, C_coefs[q] = {(0,0): c00, (0,1): c01, (1,0): c10, (1,1): c11}
    C_coefs.append({(0,0):1, (0,1):0, (1,0):0, (1,1):1})  # M1
    C_coefs.append({(0,0):0, (0,1):0, (1,0):1, (1,1):1})  # M2
    C_coefs.append({(0,0):0, (0,1):1, (1,0):0, (1,1):-1}) # M3
    C_coefs.append({(0,0):-1,(0,1):0, (1,0):1, (1,1):0})  # M4
    C_coefs.append({(0,0):0, (0,1):1, (1,0):0, (1,1):0})  # M5
    C_coefs.append({(0,0):1, (0,1):0, (1,0):0, (1,1):-1}) # M6
    C_coefs.append({(0,0):-1,(0,1):0, (1,0):0, (1,1):0})  # M7
    
    return terms, C_coefs

def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We decompose the 4x5x5 multiplication into blocks.
    # A is 4x5, B is 5x5, C is 4x5.
    # Split A into rows: top 2 rows (A_top, 2x5) and bottom 2 rows (A_bot, 2x5).
    # Split B into cols: left 2 cols (B_left, 5x2) and right 3 cols (B_right, 5x3).
    # C = A*B: C is 4x5.
    # C_top_left = A_top * B_left (2x2)
    # C_top_right = A_top * B_right (2x3)
    # C_bot_left = A_bot * B_left (2x2)
    # C_bot_right = A_bot * B_right (2x3)
    
    # Use Strassen for the 2x2 sub-problems.
    # For 2x3 sub-problems, use schoolbook (rank 6 each).
    
    U = []  # list of dicts: (row_in_A, col_in_A) -> coeff
    V = []  # list of dicts: (row_in_B, col_in_B) -> coeff
    W = []  # list of dicts: (row_in_C, col_in_C) -> coeff
    
    strassen_terms, strassen_C = _strassen_2x2()
    
    # Add a term: u_dict (A indices), v_dict (B indices), w_dict (C indices)
    def add_term(u_dict, v_dict, w_dict):
        U.append(u_dict)
        V.append(v_dict)
        W.append(w_dict)
    
    # C_top_left = A_top * B_left (2x2)
    # A_top rows: 0,1; B_left cols: 0,1; C_top_left rows: 0,1; cols: 0,1
    for q in range(7):
        A_mask, B_mask = strassen_terms[q]
        C_mask = strassen_C[q]
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for (ar, ac), val in A_mask.items():
            u_dict[(ar, ac)] = val  # A_top row ar (0,1) -> global row ar; col ac (0,1)
        for (br, bc), val in B_mask.items():
            v_dict[(br, bc)] = val  # B_left: row br (0..4), col bc (0,1)
        for (cr, cc), val in C_mask.items():
            # C_top_left (cr, cc) -> global C (cr, cc)
            if val != 0:
                w_dict[(cr, cc)] = w_dict.get((cr, cc), 0) + val
        if w_dict:
            add_term(u_dict, v_dict, w_dict)
    
    # C_bot_left = A_bot * B_left (2x2)
    # A_bot rows: 2,3; B_left cols: 0,1; C_bot_left rows: 2,3; cols: 0,1
    for q in range(7):
        A_mask, B_mask = strassen_terms[q]
        C_mask = strassen_C[q]
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for (ar, ac), val in A_mask.items():
            u_dict[(ar+2, ac)] = val  # A_bot row ar+2 (2,3) -> global row; col ac (0,1)
        for (br, bc), val in B_mask.items():
            v_dict[(br, bc)] = val  # B_left: row br (0..4), col bc (0,1)
        for (cr, cc), val in C_mask.items():
            # C_bot_left (cr, cc) -> global C (cr+2, cc)
            if val != 0:
                w_dict[(cr+2, cc)] = w_dict.get((cr+2, cc), 0) + val
        if w_dict:
            add_term(u_dict, v_dict, w_dict)
    
    # C_top_right = A_top * B_right (2x3) - schoolbook
    # A_top rows: 0,1; B_right cols: 2,3,4; C_top_right rows: 0,1; cols: 2,3,4
    for i in range(2):
        for j in range(3):
            u_dict = {(i, 0): 1}  # Wait, this isn't right. Schoolbook: C[i,j] = sum_k A[i,k] B[k,j]
            # Each rank-1 term: A[i,:] * B[:,j]
            # But A is 2x5, B is 5x3 (for this block)
            # Term for (i, j): A[i, 0:2] * B[0:5, j] where j is 0,1,2 in B_right
            # Actually, let's do it properly.
            # C_top_right[i, j] = sum_{k=0}^{4} A_top[i, k] * B_right[k, j]
            # where j is 0,1,2 corresponding to global cols 2,3,4
            pass
    
    # Let me redo this more carefully.
    # For each sub-product, I'll enumerate rank-1 terms.
    
    U.clear()
    V.clear()
    W.clear()
    
    # Helper: add schoolbook terms for A_sub (na x mb) * B_sub (mb x pc) -> C_sub (na x pc)
    # A_sub rows: ar_start..ar_start+na-1, cols: ac_start..ac_start+mb-1
    # B_sub rows: br_start..br_start+mb-1, cols: bc_start..bc_start+pc-1
    # C_sub rows: cr_start..cr_start+na-1, cols: cc_start..cc_start+pc-1
    def add_schoolbook(ar_start, ac_start, na, mb, br_start, bc_start, pc, cr_start, cc_start):
        for i in range(na):
            for j in range(pc):
                for k in range(mb):
                    u = {(ar_start+i, ac_start+k): 1}
                    v = {(br_start+k, bc_start+j): 1}
                    w = {(cr_start+i, cc_start+j): 1}
                    add_term(u, v, w)
    
    # Helper: add Strassen terms for 2x2 block
    def add_strassen_2x2(ar_start, ac_start, br_start, bc_start, cr_start, cc_start):
        for q in range(7):
            A_mask, B_mask = strassen_terms[q]
            C_mask = strassen_C[q]
            u_dict = {}
            v_dict = {}
            w_dict = {}
            for (ar, ac), val in A_mask.items():
                key = (ar_start+ar, ac_start+ac)
                u_dict[key] = u_dict.get(key, 0) + val
            for (br, bc), val in B_mask.items():
                key = (br_start+br, bc_start+bc)
                v_dict[key] = v_dict.get(key, 0) + val
            for (cr, cc), val in C_mask.items():
                key = (cr_start+cr, cc_start+cc)
                if val != 0:
                    w_dict[key] = w_dict.get(key, 0) + val
            if u_dict and v_dict and w_dict:
                add_term(u_dict, v_dict, w_dict)
    
    # C_top_left = A[0:2, 0:2] * B[0:5, 0:2] -> C[0:2, 0:2]
    # This is 2x5 * 5x2 = 2x2
    # Use Strassen on the 2x2 part, but A is 2x5 and B is 5x2.
    # Strassen applies to 2x2 * 2x2. Here the inner dimension is 5, not 2.
    # So Strassen doesn't directly apply unless we further decompose.
    # 
    # Actually, Strassen works for any n x n * n x n where n is a power of 2,
    # by recursive block decomposition. For 2x5 * 5x2, we can't directly use Strassen
    # because the matrices aren't square.
    #
    # Let me reconsider. The standard approach:
    # 4x5 * 5x5 = 4x5
    # 
    # We can split as:
    # A = [A1; A2] where A1 is 2x5, A2 is 2x5
    # B = [B1 | B2] where B1 is 5x2, B2 is 5x3
    # C = [A1*B1 | A1*B2; A2*B1 | A2*B2]
    #
    # A1*B1: 2x5 * 5x2 = 2x2. Schoolbook rank = 2*5*2 = 20.
    # Can we do better? For 2x5*5x2, the tensor is essentially the contraction
    # of a 2x5 matrix with a 5x2 matrix. This is equivalent to computing
    # 2x2 matrix products with an intermediate 5-dimensional space.
    # The rank is at most 2*5*2 = 20, and at least 4 (since it's a 2x2 result).
    # Actually, this is the "matrix multiplication" tensor with dimensions (2,5,2).
    # The rank of M_{2,5,2}... I'm not sure of the exact rank.
    # 
    # For now, let's just use schoolbook for all blocks and see if we can optimize.
    
    # Schoolbook for all:
    # A1*B1: 2*5*2 = 20
    # A1*B2: 2*5*3 = 30
    # A2*B1: 2*5*2 = 20
    # A2*B2: 2*5*3 = 30
    # Total: 100 (same as original)
    
    # To improve, we need to find a better decomposition.
    # 
    # Idea: Use the fact that 5 = 2+3.
    # A1*B1 = A1[:, 0:2] * B1[0:2, :] + A1[:, 2:5] * B1[2:5, :]
    # Each is 2x2 * 2x2 (first part) and 2x3 * 3x2 (second part).
    # 2x2 * 2x2 can be done with Strassen (rank 7).
    # 2x3 * 3x2: schoolbook rank = 2*3*2 = 12.
    # So A1*B1: 7 + 12 = 19 (saving 1)
    # 
    # A1*B2 = A1[:, 0:2] * B2[0:2, :] + A1[:, 2:5] * B2[2:5, :]
    # 2x2 * 2x3: schoolbook = 12
    # 2x3 * 3x3: schoolbook = 18
    # Total: 30 (no saving)
    # 
    # Hmm, the savings are minimal. Let me think differently.
    #
    # Actually, for 2x2 * 2x2, Strassen gives rank 7 vs schoolbook 8. Saving of 1.
    # For the 4x4 * 4x4 sub-problem within the 4x5 * 5x5, we could get more savings.
    #
    # Let me try: split A into 4x2 + 4x3 (by columns), B into 2x5 + 3x5 (by rows).
    # A*B = A[:, 0:2]*B[0:2, :] + A[:, 2:5]*B[2:5, :]
    # A[:, 0:2] is 4x2, B[0:2, :] is 2x5: 4x2*2x5 = 4x5, schoolbook rank = 4*2*5 = 40
    # A[:, 2:5] is 4x3, B[2:5, :] is 3x5: 4x3*3x5 = 4x5, schoolbook rank = 4*3*5 = 60
    # Total: 100. No improvement.
    #
    # The key insight from the observations: Strassen for 4x4x4 reduces rank from 64 to 28.
    # So if we can isolate a 4x4 * 4x4 sub-product, we save 36.
    #
    # A is 4x5, B is 5x5.
    # Let A = [A_left | A_right] where A_left is 4x4, A_right is 4x1.
    # Let B = [B_top; B_bottom] where B_top is 4x5, B_bottom is 1x5.
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top: 4x4 * 4x5 = 4x5. Schoolbook rank = 4*4*5 = 80.
    #   Using Strassen-like: split into 4x4*4x4 + 4x4*4x1
    #   4x4*4x4: Strassen rank 28 (for the 4x4x4 tensor)
    #   4x4*4x1: schoolbook rank = 4*4*1 = 16
    #   Total: 28 + 16 = 44 (saving 36)
    # A_right * B_bottom: 4x1 * 1x5 = 4x5. Schoolbook rank = 4*1*5 = 20.
    # Total: 44 + 20 = 64.
    #
    # This is a saving of 36 from 100!
    #
    # Now I need to implement Strassen for 4x4*4x4.
    # Strassen recursively: 4x4 = 2x(2x2), so use Strassen at the 2x2 level,
    # which calls 2x2 Strassen (7 terms), each of which is 2x2*2x2.
    # So 4x4*4x4 Strassen has 7^2 = 49 terms? No, that's not right.
    #
    # Standard Strassen for n x n where n is a power of 2:
    # T(1) = 1, T(n) = 7 * T(n/2)
    # T(2) = 7, T(4) = 7*7 = 49.
    #
    # But the observation says rank 28 for 4x4x4. That's less than 49.
    # This might be using a different algorithm or a specific decomposition.
    #
    # Let me check: the known rank of M_{4,4,4} is at most 26 (from recent results),
    # and Strassen gives 49. So 28 might be from a specific construction.
    #
    # For now, let me implement recursive Strassen and see what rank we get.
    # T(4) = 49 for Strassen.
    #
    # Wait, the observation says "Strassen decomposition for the 4x4x4 sub-tensor reduces
    # rank from 64 to 28". 64 is schoolbook (4*4*4). 28 is claimed for Strassen.
    # But standard Strassen gives 49. Maybe they're using a different method.
    #
    # Let me just implement schoolbook for now and verify correctness, then optimize.
    
    # Let me use the decomposition:
    # A = [A_left | A_right], A_left 4x4, A_right 4x1
    # B = [B_top; B_bottom], B_top 4x5, B_bottom 1x5
    # C = A_left*B_top + A_right*B_bottom
    #
    # A_left*B_top: split B_top = [B_top_left | B_top_right], B_top_left 4x4, B_top_right 4x1
    # A_left*B_top_left: 4x4*4x4, schoolbook rank 64
    # A_left*B_top_right: 4x4*4x1, schoolbook rank 16
    # A_right*B_bottom: 4x1*1x5, schoolbook rank 20
    # Total: 64+16+20 = 100. Still 100!
    #
    # The issue is that splitting doesn't help unless we use a sub-linear algorithm for
    # the 4x4*4x4 part. With Strassen (rank 49), total = 49+16+20 = 85.
    #
    # Let me implement Strassen for 4x4*4x4.
    
    def _strassen_recursive(A_size, B_size):
        """Return list of (A_sub_indices, B_sub_indices, C_sub_indices, coeff)
        for Strassen decomposition of A_size x B_size matrix multiplication.
        A is A_size x A_size, B is B_size x B_size, C is A_size x B_size.
        Only works when A_size == B_size and is a power of 2."""
        if A_size != B_size:
            return None
        if A_size == 1:
            return [({(0,0):1}, {(0,0):1}, {(0,0):1})]
        if A_size % 2 != 0:
            return None
        half = A_size // 2
        sub_terms, sub_C = _strassen_2x2()
        
        # 4x4: split into 2x(2x2)
        # A = [[A00, A01], [A10, A11]] each 2x2
        # B = [[B00, B01], [B10, B11]] each 2x2
        # C = A*B, 4x4
        # Strassen: 7 products of 2x2 matrices
        # For each 2x2 product, recurse to get 7 terms of 1x1
        
        # Actually, let me just build it up.
        # For 2x2: 7 terms, each is 1x1*1x1->1x1
        # For 4x4: 7 terms, each is 2x2*2x2->2x2, and each 2x2*2x2 has 7 terms
        # So 49 total terms, each 1x1*1x1->1x1
        
        # Let me build the 4x4 Strassen terms.
        # The 7 "big" terms for 4x4:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # where A00, A01, etc. are 2x2 blocks.
        
        # For each M_q, we need to decompose the 2x2*2x2 product.
        # Using the 2x2 Strassen terms.
        
        big_terms = []
        for q in range(7):
            A_mask, B_mask = sub_terms[q]
            C_mask = sub_C[q]
            # A_mask and B_mask are 2x2 masks
            # We need to expand each 2x2 product using 2x2 Strassen
            # But wait, A_mask is a 2x2 matrix of coefficients, and we're multiplying
            # two 2x2 "matrices" where each entry is itself a 2x2 block.
            # This is getting complex. Let me just do schoolbook for 4x4*4x4.
            
            pass
        
        return None
    
    # Given the complexity, let me just use schoolbook but with the block decomposition
    # that allows using Strassen for 2x2 sub-problems.
    
    # Split A (4x5) into 2 blocks of 2 rows: A1 (2x5), A2 (2x5)
    # Split B (5x5) into 2 blocks of 2 cols + 1 col: B1 (5x2), B2 (5x2), B3 (5x1)
    # C = [A1*B1 | A1*B2 | A1*B3; A2*B1 | A2*B2 | A2*B3]
    #
    # A1*B1: 2x5*5x2 = 2x2. Split A1 = [A1a | A1b] (2x2, 2x3), B1 = [B1a; B1b] (2x2, 3x2)
    # A1*B1 = A1a*B1a + A1b*B1b
    # A1a*B1a: 2x2*2x2 = 2x2. Strassen rank 7.
    # A1b*B1b: 2x3*3x2 = 2x2. Schoolbook rank 12.
    # Total for A1*B1: 7 + 12 = 19.
    #
    # Similarly A2*B1: 19.
    #
    # A1*B2: 2x5*5x2 = 2x2. Same as above: 19.
    # A2*B2: 19.
    #
    # A1*B3: 2x5*5x1 = 2x1. Schoolbook rank 10.
    # A2*B3: 10.
    #
    # Total: 19*4 + 10*2 = 76 + 20 = 96.
    # Saving of 4 from 100.
    
    # Let me implement this.
    U.clear()
    V.clear()
    W.clear()
    
    # Strassen 2x2 terms
    s_terms, s_C = _strassen_2x2()
    
    def add_strassen_2x2_block(ar_off, ac_off, br_off, bc_off, cr_off, cc_off):
        """Add Strassen terms for a 2x2*2x2 -> 2x2 block.
        A block: rows ar_off..ar_off+1, cols ac_off..ac_off+1
        B block: rows br_off..br_off+1, cols bc_off..bc_off+1
        C block: rows cr_off..cr_off+1, cols cc_off..cc_off+1
        """
        for q in range(7):
            A_mask, B_mask = s_terms[q]
            C_mask = s_C[q]
            u_dict = {}
            v_dict = {}
            w_dict = {}
            for (ar, ac), val in A_mask.items():
                key = (ar_off+ar, ac_off+ac)
                u_dict[key] = u_dict.get(key, 0) + val
            for (br, bc), val in B_mask.items():
                key = (br_off+br, bc_off+bc)
                v_dict[key] = v_dict.get(key, 0) + val
            for (cr, cc), val in C_mask.items():
                key = (cr_off+cr, cc_off+cc)
                if val != 0:
                    w_dict[key] = w_dict.get(key, 0) + val
            if u_dict and v_dict and w_dict:
                add_term(u_dict, v_dict, w_dict)
    
    def add_schoolbook_block(ar_off, ac_off, na, mb, br_off, bc_off, pc, cr_off, cc_off):
        """Add schoolbook terms for a na x mb * mb x pc -> na x pc block.
        A: rows ar_off..ar_off+na-1, cols ac_off..ac_off+mb-1
        B: rows br_off..br_off+mb-1, cols bc_off..bc_off+pc-1
        C: rows cr_off..cr_off+na-1, cols cc_off..cc_off+pc-1
        """
        for i in range(na):
            for j in range(pc):
                for k in range(mb):
                    u = {(ar_off+i, ac_off+k): 1}
                    v = {(br_off+k, bc_off+j): 1}
                    w = {(cr_off+i, cc_off+j): 1}
                    add_term(u, v, w)
    
    # A1*B1: 2x5*5x2 -> 2x2
    # A1: rows 0-1, cols 0-4
    # B1: rows 0-4, cols 0-1
    # C: rows 0-1, cols 0-1
    # Split: A1a = A[0:2, 0:2], A1b = A[0:2, 2:5]
    #        B1a = B[0:2, 0:2], B1b = B[2:5, 0:2]
    # A1*B1 = A1a*B1a + A1b*B1b
    add_strassen_2x2_block(0, 0, 0, 0, 0, 0)  # A1a*B1a: Strassen
    add_schoolbook_block(0, 2, 2, 3, 2, 0, 2, 0, 0)  # A1b*B1b: schoolbook
    
    # A2*B1: 2x5*5x2 -> 2x2
    # A2: rows 2-3, cols 0-4
    # B1: rows 0-4, cols 0-1
    # C: rows 2-3, cols 0-1
    # A2a = A[2:4, 0:2], A2b = A[2:4, 2:5]
    # B1a = B[0:2, 0:2], B1b = B[2:5, 0:2]
    add_strassen_2x2_block(2, 0, 0, 0, 2, 0)  # A2a*B1a: Strassen
    add_schoolbook_block(2, 2, 2, 3, 2, 0, 2, 2, 0)  # A2b*B1b: schoolbook
    
    # A1*B2: 2x5*5x2 -> 2x2
    # A1: rows 0-1, cols 0-4
    # B2: rows 0-4, cols 2-3
    # C: rows 0-1, cols 2-3
    # A1a = A[0:2, 0:2], A1b = A[0:2, 2:5]
    # B2a = B[0:2, 2:4], B2b = B[2:5, 2:4]
    add_strassen_2x2_block(0, 0, 0, 2, 0, 2)  # A1a*B2a: Strassen
    add_schoolbook_block(0, 2, 2, 3, 2, 2, 2, 0, 2)  # A1b*B2b: schoolbook
    
    # A2*B2: 2x5*5x2 -> 2x2
    # A2: rows 2-3, cols 0-4
    # B2: rows 0-4, cols 2-3
    # C: rows 2-3, cols 2-3
    add_strassen_2x2_block(2, 0, 0, 2, 2, 2)  # A2a*B2a: Strassen
    add_schoolbook_block(2, 2, 2, 3, 2, 2, 2, 2, 2)  # A2b*B2b: schoolbook
    
    # A1*B3: 2x5*5x1 -> 2x1
    # A1: rows 0-1, cols 0-4
    # B3: rows 0-4, col 4
    # C: rows 0-1, col 4
    add_schoolbook_block(0, 0, 2, 5, 0, 4, 1, 0, 4)
    
    # A2*B3: 2x5*5x1 -> 2x1
    # A2: rows 2-3, cols 0-4
    # B3: rows 0-4, col 4
    # C: rows 2-3, col 4
    add_schoolbook_block(2, 0, 2, 5, 0, 4, 1, 2, 4)
    
    # Convert to matrices
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U[c].items():
            U_mat[i * m + j][c] = val
        for (j, k), val in V[c].items():
            V_mat[j * p + k][c] = val
        for (k, i), val in W[c].items():
            W_mat[k * n + i][c] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
