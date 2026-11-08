"""
Matrix multiplication 4x4x5 tensor decomposition.
Uses Strassen's algorithm for the 4x4x4 sub-problem (rank 49)
and schoolbook for the remaining 4x4x1 part (rank 16), total rank 65.
"""
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for a rank-65 exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
    
    The decomposition uses:
    - Strassen's 7-term algorithm for the 4x4x4 part (49 rank terms)
    - Schoolbook for the 4x4x1 part (16 rank terms)
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 5
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 65)
    V = [[0] * rank for _ in range(m * p)]  # (20, 65)
    W = [[0] * rank for _ in range(p * n)]  # (20, 65)
    
    col = 0
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4x4, split into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # Each 2x2x2 Strassen term has 7 rank-1 terms.
    # For the 4x4x4 case, we have 7 Strassen products, each requiring 7 rank terms
    # for the inner 2x2x2 multiplication, giving 49 terms total.
    
    # Define the 7 Strassen bilinear forms for 2x2x2
    # Each form is a list of (coeff_A, coeff_B, coeff_C) for each of the 7 rank terms
    # For a 2x2x2 matrix multiplication using Strassen:
    # The 7 products M1..M7 are bilinear forms.
    # Each M_k is a 2x2 matrix product of linear combinations.
    # To decompose each M_k into rank-1 terms, we use the standard 2x2x2 schoolbook
    # on the linear combinations, but that would give 4 terms each = 28 terms.
    # 
    # Actually, Strassen's algorithm for 2x2x2 uses 7 scalar multiplications.
    # Each scalar multiplication corresponds to one rank term in the tensor decomposition.
    # But wait - for 2x2x2, the tensor rank is 7 (Strassen).
    # The 7 rank terms correspond to the 7 Strassen products.
    #
    # Let me define the 7 rank terms for 2x2x2 explicitly.
    # For 2x2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # C = AB means:
    # i = ae + bg
    # j = af + bh
    # k = ce + dg
    # l = cf + dh
    #
    # Strassen's 7 products:
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M2 = (c+d)e = ce + de
    # M3 = a(f-h) = af - ah
    # M4 = d(g-e) = dg - de
    # M5 = (a+b)h = ah + bh
    # M6 = (c-a)(e+f) = ce + cf - ae - af
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    #
    # C entries:
    # i = M1 + M4 - M5 + M7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
    #   = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh = ae + bg ✓
    # j = M3 + M5 = (af-ah) + (ah+bh) = af + bh ✓
    # k = M2 + M4 = (ce+de) + (dg-de) = ce + dg ✓
    # l = M1 - M2 + M3 + M6 = (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cf-ae-af)
    #   = ae+ah+de+dh -ce-de +af-ah +ce+cf-ae-af = dh + cf ✓
    
    # The 7 rank terms for 2x2x2:
    # Each term r has U_r (2x2), V_r (2x2), W_r (2x2) such that
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = C[i,k]... 
    # Wait, I need to be careful with the index conventions.
    
    # For 2x2x2 with trace convention W[k*2+i]:
    # U is 4x1 (flattened 2x2 A), V is 4x1 (flattened 2x2 B), W is 4x1 (flattened 2x2 C)
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Term 1: M1 = (a+d)(e+h)
    # a is A[0,0], d is A[1,1], e is B[0,0], h is B[1,1]
    # U: [1,0,0,1] (a+d), V: [1,0,0,1] (e+h), W: contributes to all C entries
    # M1 contributes: +1 to C[0,0], -1 to C[1,1]... wait let me recheck.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # So M1 contributes: +1 to C[0,0], +1 to C[1,1]
    # M2 contributes: +1 to C[1,0], -1 to C[1,1]
    # M3 contributes: +1 to C[0,1], +1 to C[1,1]
    # M4 contributes: +1 to C[0,0], +1 to C[1,0]
    # M5 contributes: -1 to C[0,0], +1 to C[0,1]
    # M6 contributes: +1 to C[1,1]
    # M7 contributes: +1 to C[0,0]
    
    # Let me define the 7 terms more carefully.
    # For each Strassen product M_k, we have:
    # M_k = (linear combo of A) * (linear combo of B)
    # The result M_k is a scalar that gets added to various C entries.
    
    # Actually, I realize the issue. In Strassen's algorithm, each M_k is a scalar
    # (for 1x1x1 case) or a matrix (for larger blocks). For the base 2x2x2 case,
    # the 7 products are scalars, and they combine to give the 4 entries of C.
    
    # For the tensor decomposition of 2x2x2 with rank 7:
    # We need 7 rank-1 tensors whose sum equals the matmul tensor.
    
    # Let me just use the known explicit decomposition.
    # The 2x2x2 matmul tensor T has T[(i,j),(j',k),(k',i')] = 1 if j=j', k=k', i=i'
    # i.e., T is nonzero only when the indices align properly.
    
    # I'll use a different approach: directly construct the 4x4x4 decomposition
    # using block Strassen, and handle the index mapping carefully.
    
    # For 4x4x4, split into 2x2 blocks.
    # A block (bi, bj) where bi in {0,1}, bj in {0,1}, each block is 2x2.
    # A[bi*2+i, bj*2+j] for i,j in {0,1}
    
    # Strassen products for 4x4x4:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # Each M_k is a 2x2 matrix product. To decompose M_k into rank terms,
    # we use the 2x2x2 decomposition (rank 7) for each.
    
    # For a 2x2x2 multiplication X*Y=Z where X, Y, Z are 2x2:
    # The 7 rank terms are:
    
    # I'll define helper to add a 2x2x2 Strassen decomposition at given offsets.
    
    def add_strassen_2x2x2(U_base, V_base, W_base, U_off, V_off, W_off, sign=1):
        """
        Add 7 rank terms for 2x2x2 Strassen decomposition.
        U_base, V_base, W_base: the linear combination coefficients (2x2 lists)
        U_off, V_off, W_off: offset in the global matrices
        sign: multiply all coefficients by this
        """
        # The 7 Strassen terms for 2x2x2:
        # Each term has U (2x2), V (2x2), W (2x2)
        # U_flat[i*2+j], V_flat[j*2+k], W_flat[k*2+i]
        
        terms = [
            # Term 1: (x00+x11)(y00+y11)
            {'U': [[1,0],[0,1]], 'V': [[1,0],[0,1]], 'W': [[1,0],[0,1]]},
            # Term 2: (x10+x11)y00
            {'U': [[0,0],[1,1]], 'V': [[1,0],[0,0]], 'W': [[0,0],[1,-1]]},
            # Term 3: x00(y01-y11)
            {'U': [[1,0],[0,0]], 'V': [[0,1],[0,-1]], 'W': [[0,1],[0,1]]},
            # Term 4: x11(y10-y00)
            {'U': [[0,0],[0,1]], 'V': [[-1,0],[1,0]], 'W': [[1,0],[1,0]]},
            # Term 5: (x00+x01)y11
            {'U': [[1,1],[0,0]], 'V': [[0,0],[0,1]], 'W': [[-1,0],[0,1]]},
            # Term 6: (x10-x00)(y00+y01)
            {'U': [[-1,0],[1,0]], 'V': [[1,1],[0,0]], 'W': [[0,0],[0,1]]},
            # Term 7: (x01-x11)(y10+y11)
            {'U': [[0,1],[0,-1]], 'V': [[0,0],[1,1]], 'W': [[1,0],[0,0]]},
        ]
        
        for term in terms:
            for i in range(2):
                for j in range(2):
                    u_coeff = term['U'][i][j] * sign
                    if u_coeff != 0:
                        U_base[i*2+j][U_off + i*2 + j] = u_coeff
            
            for j in range(2):
                for k in range(2):
                    v_coeff = term['V'][j][k] * sign
                    if v_coeff != 0:
                        V_base[j*2+k][V_off + j*2 + k] = v_coeff
            
            for k in range(2):
                for i in range(2):
                    w_coeff = term['W'][k][i] * sign
                    if w_coeff != 0:
                        W_base[k*2+i][W_off + k*2 + i] = w_coeff
    
    # Now build the 4x4x4 part using block Strassen.
    # 7 Strassen products, each is a 2x2x2 multiplication.
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is at rows 0-1, cols 0-1; A11 is at rows 2-3, cols 2-3
    # B00 is at rows 0-1, cols 0-1; B11 is at rows 2-3, cols 2-3
    # Result contributes to C00 (rows 0-1, cols 0-1) and C11 (rows 2-3, cols 2-3)
    
    # For M1, the A part is A00+A11, which is a 2x2 matrix where:
    # (A00+A11)[i,j] = A[i,j] + A[i+2,j+2]
    # Similarly for B.
    
    # The result M1 is a 2x2 matrix, and it gets added to both C00 and C11.
    
    # Let me define the linear combinations for each Strassen product.
    
    # M1: A_part = A00 + A11, B_part = B00 + B11
    # M2: A_part = A10 + A11, B_part = B00
    # M3: A_part = A00, B_part = B01 - B11
    # M4: A_part = A11, B_part = B10 - B00
    # M5: A_part = A00 + A01, B_part = B11
    # M6: A_part = A10 - A00, B_part = B00 + B01
    # M7: A_part = A01 - A11, B_part = B10 + B11
    
    # C contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M_k, we need to:
    # 1. Create the linear combination of A blocks (2x2 matrix of coefficients)
    # 2. Create the linear combination of B blocks (2x2 matrix of coefficients)
    # 3. For each of the 7 inner Strassen terms, add rank terms
    #    with W coefficients for all C blocks that M_k contributes to.
    
    # Let me define the A and B linear combinations as 2x2 matrices of coefficients
    # where each entry indicates which global A/B entry is used.
    
    # Actually, let me think about this differently.
    # For M1 = (A00+A11)(B00+B11):
    # The "A matrix" for this product has entries:
    # A_mat[i,j] = A[i,j] + A[i+2,j+2] for i,j in {0,1}
    # In terms of global A indices: A[i,j] is at index i*4+j, A[i+2,j+2] is at (i+2)*4+(j+2)
    
    # For the tensor decomposition, each rank term r has:
    # U_r: 16-dim vector (coefficients for A)
    # V_r: 20-dim vector (coefficients for B)  
    # W_r: 20-dim vector (coefficients for C)
    
    # For M1, the A part uses A[i,j] and A[i+2,j+2].
    # So U_r will have 1 at positions i*4+j and (i+2)*4+(j+2) for the appropriate i,j.
    
    # Let me just directly construct all 49 terms for the 4x4x4 part.
    
    # I'll define the 7 Strassen products with their A and B block combinations,
    # and the C blocks they contribute to.
    
    strassen_products = [
        # (name, A_blocks, B_blocks, C_blocks_with_signs)
        # A_blocks: dict from local (i,j) to list of (global_row, global_col, coeff)
        # B_blocks: dict from local (j,k) to list of (global_row, global_col, coeff)
        # C_blocks: list of (block_row, block_col, sign)
    ]
    
    # Let me use a simpler representation.
    # For each Strassen product, I'll specify:
    # - A_combo: 2x2 list where A_combo[i][j] = list of (global_i, global_j, coeff)
    # - B_combo: 2x2 list where B_combo[j][k] = list of (global_j, global_k, coeff)
    # - C_contrib: list of (block_i, block_j, sign)
    
    # M1 = (A00+A11)(B00+B11)
    # A00: rows 0-1, cols 0-1. A11: rows 2-3, cols 2-3.
    A1 = [
        [[(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)]],
        [[(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]]
    ]
    B1 = [
        [[(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)]],
        [[(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]]
    ]
    C1 = [(0,0,1), (1,1,1)]  # C00 +1, C11 +1
    
    # M2 = (A10+A11)B00
    # A10: rows 2-3, cols 0-1. A11: rows 2-3, cols 2-3.
    A2 = [
        [[(2,0,1), (2,2,1)], [(2,1,1), (2,3,1)]],
        [[(3,0,1), (3,2,1)], [(3,1,1), (3,3,1)]]
    ]
    B2 = [
        [[(0,0,1)], [(0,1,1)]],
        [[(1,0,1)], [(1,1,1)]]
    ]
    C2 = [(1,0,1), (1,1,-1)]  # C10 +1, C11 -1
    
    # M3 = A00(B01-B11)
    A3 = [
        [[(0,0,1)], [(0,1,1)]],
        [[(1,0,1)], [(1,1,1)]]
    ]
    B3 = [
        [[(0,0,-1), (2,0,1)], [(0,1,-1), (2,1,1)]],
        [[(1,0,-1), (3,0,1)], [(1,1,-1), (3,1,1)]]
    ]
    C3 = [(0,1,1), (1,1,1)]  # C01 +1, C11 +1
    
    # M4 = A11(B10-B00)
    A4 = [
        [[(2,2,1)], [(2,3,1)]],
        [[(3,2,1)], [(3,3,1)]]
    ]
    B4 = [
        [[(0,0,-1), (2,0,1)], [(0,1,-1), (2,1,1)]],
        [[(1,0,-1), (3,0,1)], [(1,1,-1), (3,1,1)]]
    ]
    C4 = [(0,0,1), (1,0,1)]  # C00 +1, C10 +1
    
    # M5 = (A00+A01)B11
    A5 = [
        [[(0,0,1), (0,2,1)], [(0,1,1), (0,3,1)]],
        [[(1,0,1), (1,2,1)], [(1,1,1), (1,3,1)]]
    ]
    B5 = [
        [[(2,2,1)], [(2,3,1)]],
        [[(3,2,1)], [(3,3,1)]]
    ]
    C5 = [(0,0,-1), (0,1,1)]  # C00 -1, C01 +1
    
    # M6 = (A10-A00)(B00+B01)
    A6 = [
        [[(0,0,-1), (2,0,1)], [(0,1,-1), (2,1,1)]],
        [[(1,0,-1), (3,0,1)], [(1,1,-1), (3,1,1)]]
    ]
    B6 = [
        [[(0,0,1), (0,2,1)], [(0,1,1), (0,3,1)]],
        [[(1,0,1), (1,2,1)], [(1,1,1), (1,3,1)]]
    ]
    C6 = [(1,1,1)]  # C11 +1
    
    # M7 = (A01-A11)(B10+B11)
    A7 = [
        [[(0,2,1), (2,2,-1)], [(0,3,1), (2,3,-1)]],
        [[(1,2,1), (3,2,-1)], [(1,3,1), (3,3,-1)]]
    ]
    B7 = [
        [[(2,0,1), (2,2,1)], [(2,1,1), (2,3,1)]],
        [[(3,0,1), (3,2,1)], [(3,1,1), (3,3,1)]]
    ]
    C7 = [(0,0,1)]  # C00 +1
    
    # Now for each Strassen product, decompose the 2x2x2 multiplication into 7 rank terms.
    # The 2x2x2 Strassen terms (for X*Y=Z):
    
    inner_terms = [
        # (U_local, V_local, W_local) where each is 2x2
        # Term 1: (x00+x11)(y00+y11)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # Term 2: (x10+x11)y00
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),
        # Term 3: x00(y01-y11)
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,1]]),
        # Term 4: x11(y10-y00)
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,0],[1,0]]),
        # Term 5: (x00+x01)y11
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]),
        # Term 6: (x10-x00)(y00+y01)
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # Term 7: (x01-x11)(y10+y11)
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    all_products = [
        (A1, B1, C1),
        (A2, B2, C2),
        (A3, B3, C3),
        (A4, B4, C4),
        (A5, B5, C5),
        (A6, B6, C6),
        (A7, B7, C7),
    ]
    
    for A_combo, B_combo, C_contrib in all_products:
        for U_loc, V_loc, W_loc in inner_terms:
            # For this rank term, add contributions to U, V, W
            # U: for each local (i,j), add U_loc[i][j] * A_combo[i][j] coefficients
            for i in range(2):
                for j in range(2):
                    u_coeff = U_loc[i][j]
                    if u_coeff != 0:
                        for (gi, gj, ac) in A_combo[i][j]:
                            U[gi * 4 + gj][col] += u_coeff * ac
            
            # V: for each local (j,k), add V_loc[j][k] * B_combo[j][k] coefficients
            for j in range(2):
                for k in range(2):
                    v_coeff = V_loc[j][k]
                    if v_coeff != 0:
                        for (gj, gk, bc) in B_combo[j][k]:
                            V[gj * 5 + gk][col] += v_coeff * bc
            
            # W: for each C block that this product contributes to
            # W_loc[k][i] is the coefficient for C_block[k][i]
            # Global C index: (block_row*2 + k)*4 + (block_col*2 + i)
            # But we use trace convention: W[k*n+i] where k is col, i is row
            # Wait, C is 4x5, but for the 4x4x4 part, C is 4x4.
            # C[row, col] where row in 0..3, col in 0..3
            # Trace convention: W[col*4 + row]
            for (block_row, block_col, sign) in C_contrib:
                for k in range(2):
                    for i in range(2):
                        w_coeff = W_loc[k][i] * sign
                        if w_coeff != 0:
                            global_row = block_row * 2 + i
                            global_col = block_col * 2 + k
                            # Trace convention: W[col*4 + row]
                            W[global_col * 4 + global_row][col] += w_coeff
            
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    # Now add the 4x4x1 part: C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_j A[i,j] * B[j, 4]
    # Trace convention: W[4*4 + i] = W[16 + i]
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
