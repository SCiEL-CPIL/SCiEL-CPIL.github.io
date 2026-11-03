import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def _build_strassen_2x2():
    """
    Returns the 7 terms of Strassen's 2x2x2 algorithm.
    Each term is (x_terms, y_terms, c_terms) where x_terms, y_terms, c_terms
    are dicts mapping (i,j) -> coeff.
    """
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)*B00 -> contributes to C10
    # M3 = A00*(B01-B11) -> contributes to C01
    # M4 = A11*(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)*B11 -> contributes to C01
    # M6 = (A10-A00)*(B00+B01) -> contributes to C10
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00, C11
    
    terms = []
    
    # M1
    x = {(0,0): 1, (1,1): 1}
    y = {(0,0): 1, (1,1): 1}
    c = {(0,0): 1, (1,1): 1}
    terms.append((x, y, c))
    
    # M2
    x = {(1,0): 1, (1,1): 1}
    y = {(0,0): 1}
    c = {(1,0): 1}
    terms.append((x, y, c))
    
    # M3
    x = {(0,0): 1}
    y = {(0,1): 1, (1,1): -1}
    c = {(0,1): 1}
    terms.append((x, y, c))
    
    # M4
    x = {(1,1): 1}
    y = {(1,0): 1, (0,0): -1}
    c = {(0,0): 1, (1,0): 1}
    terms.append((x, y, c))
    
    # M5
    x = {(0,0): 1, (0,1): 1}
    y = {(1,1): 1}
    c = {(0,1): 1}
    terms.append((x, y, c))
    
    # M6
    x = {(1,0): 1, (0,0): -1}
    y = {(0,0): 1, (0,1): 1}
    c = {(1,0): 1}
    terms.append((x, y, c))
    
    # M7
    x = {(0,1): 1, (1,1): -1}
    y = {(1,0): 1, (1,1): 1}
    c = {(0,0): 1, (1,1): 1}
    terms.append((x, y, c))
    
    return terms

def _embed_strassen_2x2_to_4x4(strassen_terms):
    """
    Embed the 2x2x2 Strassen terms into the 4x4x4 problem using block structure.
    Returns a list of 49 terms for the full 4x4x4 multiplication.
    """
    # We split 4x4x4 into four 2x2x2 subproblems? No, that's not how Strassen 4x4x4 works.
    # Strassen 4x4x4 rank is 49. This is obtained by applying Strassen's 2x2 to blocks of size 2x2.
    # The 4x4 matrices are partitioned into four 2x2 blocks.
    # Let A, B, C be 4x4 partitioned into 2x2 blocks of size 2x2.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2.
    
    # We apply Strassen's 7 multiplication steps to the 2x2 block matrices.
    # Each step involves a 2x2x2 tensor product of the block matrices.
    # Since each block is 2x2, each multiplication Mij * Mjk requires rank 7 (Strassen 2x2x2).
    # Total rank = 7 * 7 = 49.
    
    full_terms = []
    for i in range(7):
        xi, yi, ci = strassen_terms[i]
        for j in range(7):
            xj, yj, cj = strassen_terms[j]
            
            full_x = {}
            full_y = {}
            full_c = {}
            
            for (bi, bj), ci_coeff in xi.items():
                for (li, lj), cj_coeff in xj.items():
                    # bi, bj are block indices (0 or 1)
                    # li, lj are indices within the 2x2 block
                    # Full row index: 2*bi + li
                    # Full col index: 2*bj + lj
                    fi = 2 * bi + li
                    fj = 2 * bj + lj
                    coeff = ci_coeff * cj_coeff
                    full_x[(fi, fj)] = full_x.get((fi, fj), 0) + coeff
            
            for (bi, bj), ci_coeff in yi.items():
                for (li, lj), cj_coeff in yj.items():
                    fi = 2 * bi + li
                    fj = 2 * bj + lj
                    coeff = ci_coeff * cj_coeff
                    full_y[(fi, fj)] = full_y.get((fi, fj), 0) + coeff
            
            for (bi, bj), ci_coeff in ci.items():
                for (li, lj), cj_coeff in cj.items():
                    fi = 2 * bi + li
                    fj = 2 * bj + lj
                    coeff = ci_coeff * cj_coeff
                    full_c[(fi, fj)] = full_c.get((fi, fj), 0) + coeff
            
            # Remove zeros
            full_x = {k: v for k, v in full_x.items() if v != 0}
            full_y = {k: v for k, v in full_y.items() if v != 0}
            full_c = {k: v for k, v in full_c.items() if v != 0}
            
            full_terms.append((full_x, full_y, full_c))
    
    return full_terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Precompute Strassen terms
    strassen_2x2 = _build_strassen_2x2()
    strassen_4x4 = _embed_strassen_2x2_to_4x4(strassen_2x2)
    
    # Total rank estimate: 49 (for 4x4x4) + 44 (for 4x4x3 using 4x 11-rank 2x2x3)
    # Actually, for 4x4x3, we can use a hybrid.
    # Let's stick with the known good structure:
    # Part 1: 4x4x4 for first 4 columns of B (and corresponding rows of C). Rank 49.
    # Part 2: 4x4x3 for last 3 columns of B. Rank 4*11 = 44? Or maybe lower?
    # The current best is 93 (49 + 44).
    
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 ---
    for x_terms, y_terms, c_terms in strassen_4x4:
        r = current_rank
        current_rank += 1
        
        u_coeffs = [0.0] * (n * m)
        v_coeffs = [0.0] * (m * p)
        w_coeffs = [0.0] * (p * n)
        
        for (i, j), coeff in x_terms.items():
            u_idx = i * m + j
            u_coeffs[u_idx] = coeff
            
        for (i, j), coeff in y_terms.items():
            # Only columns 0-3 are used in this part
            if j < 4:
                v_idx = i * p + j
                v_coeffs[v_idx] = coeff
            # If j >= 4, we ignore because V matrix is m x p, and we are handling the 4x4x4 part
            # Wait, V is m x p. In the 4x4x4 part, we multiply A(4x4) by B_top(4x4).
            # B_top uses columns 0-3 of B. So v_coeffs should only touch j < 4.
        
        for (i, j), coeff in c_terms.items():
            # C indices: i is row, j is col. W index is k*n + i? No, W[k*n+i, r].
            # j is the column of C (which corresponds to the column of B).
            # So W[j*n + i, r] = coeff.
            w_idx = j * n + i
            w_coeffs[w_idx] = coeff
            
        for i in range(n * m):
            U[i][r] = u_coeffs[i]
        for j in range(m * p):
            V[j][r] = v_coeffs[j]
        for k in range(p * n):
            W[k][r] = w_coeffs[k]
            
    # --- Part 2: 4x4x3 ---
    # We handle the last 3 columns of B (indices 4, 5, 6).
    # We split A into four 2x2 blocks: A00, A01, A10, A11.
    # B columns 4-6 form a 4x3 matrix.
    # We can split B into two parts: B_45 (4x2) and B_6 (4x1).
    # B_45 can be handled by two 2x2x2 Strassen blocks?
    # Let's use the 2x2x3 decomposition for each of the 4 block interactions.
    
    # A00 * B_top_45 -> C_top_45 (2x2) using Strassen 2x2x2 (rank 7)
    # A00 * B_top_6 -> C_top_6 (2x1) using Schoolbook 2x2x1 (rank 4? or 2? Actually 2*1=2 scalar mults)
    # A01 * B_bot_45 -> C_top_45 (2x2)
    # A01 * B_bot_6 -> C_top_6 (2x1)
    # A10 * B_top_45 -> C_bot_45 (2x2)
    # A10 * B_top_6 -> C_bot_6 (2x1)
    # A11 * B_bot_45 -> C_bot_45 (2x2)
    # A11 * B_bot_6 -> C_bot_6 (2x1)
    
    # Rank for 2x2x2 is 7.
    # Rank for 2x2x1 is 4 (schoolbook).
    # Total per block pair: 7 + 4 = 11.
    # 4 block pairs: 4 * 11 = 44.
    
    # Let's implement a helper to add a 2x2x3 contribution.
    def add_2x2x3(a_row, a_col, b_row, c_row):
        nonlocal current_rank
        
        # A block is at rows 2*a_row..2*a_row+1, cols 2*a_col..2*a_col+1
        # B block is at rows 2*b_row..2*b_row+1, cols 4..6
        # C block is at rows 2*c_row..2*c_row+1, cols 4..6
        
        # Part 2a: 2x2x2 for cols 4,5
        for x_sub, y_sub, c_sub in strassen_2x2:
            r = current_rank
            current_rank += 1
            
            u_coeffs = [0.0] * (n * m)
            v_coeffs = [0.0] * (m * p)
            w_coeffs = [0.0] * (p * n)
            
            for (li, lj), coeff in x_sub.items():
                i = 2 * a_row + li
                j = 2 * a_col + lj
                u_idx = i * m + j
                u_coeffs[u_idx] = coeff
                
            for (li, lj), coeff in y_sub.items():
                i = 2 * b_row + li
                j = 4 + lj  # Cols 4 and 5
                v_idx = i * p + j
                v_coeffs[v_idx] = coeff
                
            for (li, lj), coeff in c_sub.items():
                i = 2 * c_row + li
                j = 4 + lj  # Cols 4 and 5
                w_idx = j * n + i
                w_coeffs[w_idx] = coeff
                
            for i in range(n * m):
                U[i][r] = u_coeffs[i]
            for j in range(m * p):
                V[j][r] = v_coeffs[j]
            for k in range(p * n):
                W[k][r] = w_coeffs[k]
                
        # Part 2b: 2x2x1 for col 6
        # C_col6 = A * B_col6
        # C_0,6 = A_0,0 * B_0,6 + A_0,1 * B_1,6
        # C_1,6 = A_1,0 * B_0,6 + A_1,1 * B_1,6
        
        schoolbook_2x2x1 = [
            ( {(0,0): 1}, {(0,0): 1}, {(0,0): 1} ), # A00 B0 -> C0
            ( {(0,1): 1}, {(1,0): 1}, {(0,0): 1} ), # A01 B1 -> C0
            ( {(1,0): 1}, {(0,0): 1}, {(1,0): 1} ), # A10 B0 -> C1
            ( {(1,1): 1}, {(1,0): 1}, {(1,0): 1} ), # A11 B1 -> C1
        ]
        
        for x_sub, y_sub, c_sub in schoolbook_2x2x1:
            r = current_rank
            current_rank += 1
            
            u_coeffs = [0.0] * (n * m)
            v_coeffs = [0.0] * (m * p)
            w_coeffs = [0.0] * (p * n)
            
            for (li, lj), coeff in x_sub.items():
                i = 2 * a_row + li
                j = 2 * a_col + lj
                u_idx = i * m + j
                u_coeffs[u_idx] = coeff
                
            for (li, lj), coeff in y_sub.items():
                i = 2 * b_row + li
                j = 6  # Col 6
                v_idx = i * p + j
                v_coeffs[v_idx] = coeff
                
            for (li, lj), coeff in c_sub.items():
                i = 2 * c_row + li
                j = 6  # Col 6
                w_idx = j * n + i
                w_coeffs[w_idx] = coeff
                
            for i in range(n * m):
                U[i][r] = u_coeffs[i]
            for j in range(m * p):
                V[j][r] = v_coeffs[j]
            for k in range(p * n):
                W[k][r] = w_coeffs[k]

    # Call add_2x2x3 for the 4 blocks
    add_2x2x3(0, 0, 0, 0) # A00, B_top, C_top
    add_2x2x3(0, 1, 1, 0) # A01, B_bot, C_top
    add_2x2x3(1, 0, 0, 1) # A10, B_top, C_bot
    add_2x2x3(1, 1, 1, 1) # A11, B_bot, C_bot

    # Trim
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
