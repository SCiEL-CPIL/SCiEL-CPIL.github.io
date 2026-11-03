import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen term
    # A_block: 2x2 matrix indices in U (i, j)
    # B_block: 2x2 matrix indices in V (j, k)
    # C_block: 2x2 matrix indices in W (k, i) -> W[k*n + i]
    # Actually, we need to map the local 2x2 indices to global indices.
    
    def add_strassen_2x2x2(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_k_offset, c_i_offset):
        nonlocal col
        # Local indices 0,1 for 2x2 blocks
        # a_i: 0,1 -> global i = a_i_offset + a_i
        # a_j: 0,1 -> global j = a_j_offset + a_j
        # b_j: 0,1 -> global j = b_j_offset + b_j
        # b_k: 0,1 -> global k = b_k_offset + b_k
        # c_k: 0,1 -> global k = c_k_offset + c_k
        # c_i: 0,1 -> global i = c_i_offset + c_i
        
        # Strassen 7 products
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We iterate over the 7 scalar terms of the 2x2x2 Strassen decomposition
        # Each scalar term corresponds to a rank-1 tensor.
        # The 2x2x2 Strassen decomposition has 7 terms.
        # Let's list the coefficients for U, V, W for each of the 7 terms.
        
        # Term 1: M1
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C12=1, C21=1, C22=1 (with signs)
        # Actually, we need to add 7 separate rank-1 terms for the 7 Strassen products?
        # No, the 2x2x2 multiplication itself has rank 7.
        # So we add 7 terms to the global decomposition.
        
        # Let's define the 7 terms of 2x2x2 Strassen explicitly.
        # t1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
        # The standard Strassen 7 terms for 2x2x2:
        # 1. (A11+A22)(B11+B22) -> contributes to C11, C12, C21, C22
        #    Actually, it's better to use the explicit 7 rank-1 tensors.
        
        # Let's use the known 7 tensors for 2x2x2.
        # They are:
        # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -- No, this is not correct.
        
        # Let's derive or look up the 7 vectors.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # i = ae+bg, j = af+bh, k = ce+dg, l = cf+dh
        
        # Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # i = m1 + m4 - m5 + m7
        # j = m3 + m5
        # k = m2 + m4
        # l = m1 - m2 + m3 + m6
        
        # We need to express this as sum of 7 rank-1 tensors.
        # Each tensor is U_t, V_t, W_t of size 4.
        
        # t1 (for m1):
        # U: a=1, d=1 -> [1, 0, 0, 1]
        # V: e=1, h=1 -> [1, 0, 0, 1] (Note: V is indexed by j*p+k? No, local 2x2)
        # Local V indices: e=0, f=1, g=2, h=3.
        # So V: [1, 0, 0, 1]
        # W: i=1, j=1, k=0, l=1? No.
        # i = m1 + ...
        # The contribution of m1 to i is 1, to j is 0, to k is 0, to l is 1.
        # So W: [1, 0, 0, 1]
        
        # t2 (for m2):
        # U: c=1, d=1 -> [0, 0, 1, 1]
        # V: e=1 -> [1, 0, 0, 0]
        # W: k=1, l=-1 -> [0, 0, 1, -1]
        
        # t3 (for m3):
        # U: a=1 -> [1, 0, 0, 0]
        # V: f=1, h=-1 -> [0, 1, 0, -1]
        # W: j=1, l=1 -> [0, 1, 0, 1]
        
        # t4 (for m4):
        # U: d=1 -> [0, 0, 0, 1]
        # V: g=1, e=-1 -> [-1, 0, 1, 0]
        # W: i=1, k=1 -> [1, 0, 1, 0]
        
        # t5 (for m5):
        # U: a=1, b=1 -> [1, 1, 0, 0]
        # V: h=1 -> [0, 0, 0, 1]
        # W: i=-1, j=1 -> [-1, 1, 0, 0]
        
        # t6 (for m6):
        # U: c=1, a=-1 -> [-1, 0, 1, 0]
        # V: e=1, f=1 -> [1, 1, 0, 0]
        # W: l=1 -> [0, 0, 0, 1]
        
        # t7 (for m7):
        # U: b=1, d=-1 -> [0, 1, 0, -1]
        # V: g=1, h=1 -> [0, 0, 1, 1]
        # W: i=1, l=1 -> [1, 0, 0, 1]
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
        ]
        
        for u_t, v_t, w_t in terms:
            # Map local indices to global
            for li in range(2):
                for lj in range(2):
                    val = u_t[li * 2 + lj]
                    if val != 0:
                        gi = a_i_offset + li
                        gj = a_j_offset + lj
                        U[gi * m + gj][col] += val
                        
            for lj in range(2):
                for lk in range(2):
                    val = v_t[lj * 2 + lk]
                    if val != 0:
                        gj = b_j_offset + lj
                        gk = b_k_offset + lk
                        V[gj * p + gk][col] += val
                        
            for lk in range(2):
                for li in range(2):
                    val = w_t[lk * 2 + li]
                    if val != 0:
                        gk = c_k_offset + lk
                        gi = c_i_offset + li
                        W[gk * n + gi][col] += val
                        
            col += 1

    # Part 1: 4x4x4 (Rank 49)
    # A is 4x4, B is 4x4 (cols 0-3)
    # Split into 2x2 blocks
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    
    # Strassen 7 products
    # M1 = (A11+A22)(B11+B22)
    # We need to add the 2x2x2 Strassen for the sum?
    # No, M1 is a 2x2x2 multiplication of (A11+A22) and (B11+B22).
    # We can't just call add_strassen_2x2x2 because the inputs are sums.
    # We need to expand the sums.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 independent 2x2x2 multiplications.
    # Each has rank 7. Total 28 for M1?
    # No, Strassen reduces the number of multiplications from 8 to 7 for 2x2x2.
    # But here we are computing a product of sums.
    # Wait, the Strassen algorithm for 4x4x4 uses 7 products of 2x2x2 matrices.
    # Each product is a 2x2x2 multiplication.
    # So we just call add_strassen_2x2x2 for each of the 7 products?
    # No, the inputs to the 2x2x2 multiplication are linear combinations of the A and B blocks.
    # My helper `add_strassen_2x2x2` assumes the inputs are single blocks.
    # I need a helper that can handle linear combinations.
    
    # Let's modify the approach.
    # We will manually construct the 49 terms.
    # Or, we can create a helper that takes linear combinations.
    
    # Let's stick to the explicit construction for safety and speed.
    
    # We will create a list of operations.
    # Each operation is (U_coeffs, V_coeffs, W_coeffs)
    # But it's easier to just fill the matrices.
    
    # Let's re-implement add_strassen_2x2x2 to accept linear combinations?
    # No, let's just use the fact that the 4x4x4 Strassen decomposition is composed of 7 2x2x2 Strassen decompositions.
    # But the inputs are sums.
    # Example: M1 = (A11+A22)(B11+B22).
    # This is NOT a single 2x2x2 multiplication in the tensor sense if we want to use the rank-7 decomposition.
    # The rank-7 decomposition is for the multiplication tensor T_{2,2,2}.
    # The product (A11+A22)(B11+B22) is a linear combination of 4 products.
    # We can compute it using the rank-7 decomposition of the sum?
    # No, the tensor rank is not linear.
    # However, we can just use the schoolbook for the 2x2x2 if the inputs are sums?
    # No, that would increase rank.
    
    # Correct approach:
    # The 4x4x4 Strassen algorithm computes 7 products M1..M7.
    # Each Mj is a 2x2 matrix.
    # Mj = X_j Y_j where X_j, Y_j are 2x2 matrices formed from linear combinations of A and B blocks.
    # We can compute Mj using the rank-7 decomposition of 2x2x2.
    # But X_j and Y_j are not the original A, B blocks.
    # We need to express the contribution of X_j Y_j to the final C.
    # This is getting complicated to implement generically.
    
    # Alternative: Just use the known 49 terms.
    # Or, simpler: Use the 2x2x2 Strassen for each of the 7 products, but handle the linear combinations in the U, V vectors.
    
    # Let's define a function to add a term where U, V, W are linear combinations.
    # But the rank-7 decomposition has fixed U, V, W vectors.
    # We can scale them.
    
    # Let's just use the schoolbook for 4x4x4? Rank 64.
    # 64 + 28 + 16 = 108. Worse than 112? No, 108 < 112.
    # But 93 is better.
    
    # Let's try to implement the 4x4x4 Strassen properly.
    # We will create 7 "sub-tensors" corresponding to M1..M7.
    # For each Mj, we know the linear combinations for X_j and Y_j.
    # We apply the 7 rank-1 tensors of 2x2x2 Strassen.
    # For each rank-1 tensor (u_t, v_t, w_t), we scale u_t by the coefficients of X_j, v_t by Y_j, and w_t by 1.
    # And add to U, V, W.
    
    # This is feasible.
    
    # Let's define the 7 Strassen products for 4x4x4.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate j=1..7.
    # For each j, we have X_j, Y_j.
    # X_j is a 2x2 matrix of coefficients for A blocks.
    # Y_j is a 2x2 matrix of coefficients for B blocks.
    # We apply the 7 rank-1 tensors of 2x2x2.
    
    # Let's define the 7 rank-1 tensors for 2x2x2 again.
    # t1..t7 as before.
    
    # For M1:
    # X1 = A11 + A22. Coeffs: A11=1, A22=1.
    # Y1 = B11 + B22. Coeffs: B11=1, B22=1.
    # For each t in t1..t7:
    #   U_add = sum_{a} coeff_X[a] * t.u[a]
    #   V_add = sum_{b} coeff_Y[b] * t.v[b]
    #   W_add = t.w
    #   Add to global U, V, W.
    
    # This works!
    
    # Let's implement this logic.
    
    def add_strassen_product(x_coeffs, y_coeffs, c_k_offset, c_i_offset, w_sign=1):
        nonlocal col
        # x_coeffs: dict { (i, j): coeff } for A blocks
        # y_coeffs: dict { (j, k): coeff } for B blocks
        # c_k_offset, c_i_offset: for W
        # w_sign: sign for W (for C22 = M1 - M2 + ...)
        
        # The 7 terms of 2x2x2 Strassen
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
        ]
        
        for u_t, v_t, w_t in terms:
            # Compute U_add
            for li in range(2):
                for lj in range(2):
                    val = u_t[li * 2 + lj]
                    if val != 0:
                        # Find coeff in x_coeffs for (li, lj)
                        coeff = x_coeffs.get((li, lj), 0)
                        if coeff != 0:
                            gi = li # local i
                            gj = lj # local j
                            # Map to global
                            # This function is called for a specific 2x2 block location?
                            # No, x_coeffs are relative to the 2x2 block structure of A.
                            # We need to know the global offsets.
                            pass
            
            # This is getting messy with offsets.
            # Let's pass global offsets to the helper.
            
    # Let's restart the helper with global offsets.
    
    def add_strassen_product_global(x_coeffs, y_coeffs, a_i_off, a_j_off, b_j_off, b_k_off, c_k_off, c_i_off, w_sign=1):
        nonlocal col
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
        ]
        
        for u_t, v_t, w_t in terms:
            for li in range(2):
                for lj in range(2):
                    val = u_t[li * 2 + lj]
                    if val != 0:
                        coeff = x_coeffs.get((li, lj), 0)
                        if coeff != 0:
                            gi = a_i_off + li
                            gj = a_j_off + lj
                            U[gi * m + gj][col] += coeff * val
                            
            for lj in range(2):
                for lk in range(2):
                    val = v_t[lj * 2 + lk]
                    if val != 0:
                        coeff = y_coeffs.get((lj, lk), 0)
                        if coeff != 0:
                            gj = b_j_off + lj
                            gk = b_k_off + lk
                            V[gj * p + gk][col] += coeff * val
                            
            for lk in range(2):
                for li in range(2):
                    val = w_t[lk * 2 + li]
                    if val != 0:
                        gk = c_k_off + lk
                        gi = c_i_off + li
                        W[gk * n + gi][col] += w_sign * val
                        
            col += 1

    # Part 1: 4x4x4
    # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2)
    # B blocks: B11(0,0), B12(0,2), B21(2,0), B22(2,2)
    # C blocks: C11(0,0), C12(0,2), C21(2,0), C22(2,2)
    
    # M1 = (A11+A22)(B11+B22)
    x1 = {(0,0): 1, (1,1): 1} # A11, A22
    y1 = {(0,0): 1, (1,1): 1} # B11, B22
    # C11 contribution: +1, C22 contribution: +1
    # We call add_strassen_product_global for C11 and C22?
    # No, the W vector is added to the global W.
    # The W vector for the 2x2x2 product is for a 2x2 C block.
    # So we need to specify which C block this product contributes to.
    # But M1 contributes to C11 and C22.
    # So we need to call the function twice? Or modify W?
    # The function adds to W at c_k_off, c_i_off.
    # So we call it for C11 (c_k_off=0, c_i_off=0) with w_sign=1.
    # And for C22 (c_k_off=2, c_i_off=2) with w_sign=1.
    # But this would add 14 terms (7 for C11, 7 for C22).
    # But M1 is ONE product.
    # The rank-1 decomposition of M1 has 7 terms.
    # Each term contributes to C11 and C22.
    # So we should add the W vector to both C11 and C22 in the SAME term.
    # My function only adds to one C block.
    # I need to modify the function to accept a list of (c_k_off, c_i_off, sign).
    
    def add_strassen_product_multi(x_coeffs, y_coeffs, a_i_off, a_j_off, b_j_off, b_k_off, c_targets, w_signs):
        nonlocal col
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
        ]
        
        for u_t, v_t, w_t in terms:
            for li in range(2):
                for lj in range(2):
                    val = u_t[li * 2 + lj]
                    if val != 0:
                        coeff = x_coeffs.get((li, lj), 0)
                        if coeff != 0:
                            gi = a_i_off + li
                            gj = a_j_off + lj
                            U[gi * m + gj][col] += coeff * val
                            
            for lj in range(2):
                for lk in range(2):
                    val = v_t[lj * 2 + lk]
                    if val != 0:
                        coeff = y_coeffs.get((lj, lk), 0)
                        if coeff != 0:
                            gj = b_j_off + lj
                            gk = b_k_off + lk
                            V[gj * p + gk][col] += coeff * val
                            
            for lk in range(2):
                for li in range(2):
                    val = w_t[lk * 2 + li]
                    if val != 0:
                        for idx, (c_k_off, c_i_off, w_sign) in enumerate(zip(c_targets[0], c_targets[1], w_signs)):
                            gk = c_k_off + lk
                            gi = c_i_off + li
                            W[gk * n + gi][col] += w_sign * val
                            
            col += 1

    # M1 = (A11+A22)(B11+B22)
    # C11: +1, C22: +1
    add_strassen_product_multi({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 0, 0, 0, 0, ([0, 2], [0, 2]), [1, 1])
    
    # M2 = (A21+A22)B11
    # C21: +1, C22: -1
    add_strassen_product_multi({(1,0): 1, (1,1): 1}, {(0,0): 1}, 2, 0, 0, 0, ([2, 2], [0, 2]), [1, -1])
    
    # M3 = A11(B12-B22)
    # C12: +1, C22: +1
    add_strassen_product_multi({(0,0): 1}, {(0,1): 1, (1,1): -1}, 0, 0, 0, 2, ([0, 2], [2, 2]), [1, 1])
    
    # M4 = A22(B21-B11)
    # C11: +1, C21: +1
    add_strassen_product_multi({(1,1): 1}, {(1,0): 1, (0,0): -1}, 2, 2, 0, 0, ([0, 2], [0, 0]), [1, 1])
    
    # M5 = (A11+A12)B22
    # C11: -1, C12: +1
    add_strassen_product_multi({(0,0): 1, (0,1): 1}, {(1,1): 1}, 0, 0, 2, 2, ([0, 0], [0, 2]), [-1, 1])
    
    # M6 = (A21-A11)(B11+B12)
    # C22: +1
    add_strassen_product_multi({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 2, 0, 0, 0, ([2], [2]), [1])
    
    # M7 = (A12-A22)(B21+B22)
    # C11: +1
    add_strassen_product_multi({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 0, 2, 2, 0, ([0], [0]), [1])
    
    # Part 2: 4x4x2 (Rank 28)
    # C_L (cols 4,5) = A11 B_L + A12 B_R
    # C_R (cols 4,5) = A21 B_L + A22 B_R
    # B_L: rows 0,1; cols 4,5.
    # B_R: rows 2,3; cols 4,5.
    
    # P1 = A11 B_L
    add_strassen_product_multi({(0,0): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}, 0, 0, 0, 4, ([4, 5], [0, 1]), [1, 1])
    # Wait, B_L is 2x2.
    # y_coeffs for B_L:
    # B_L[0,0] is B row 0, col 4.
    # B_L[0,1] is B row 0, col 5.
    # B_L[1,0] is B row 1, col 4.
    # B_L[1,1] is B row 1, col 5.
    # So y_coeffs = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}?
    # No, B_L is the block itself.
    # So y_coeffs should be identity for the block.
    # {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    
    # Let's correct P1.
    # A11 is block (0,0).
    # B_L is block (0, 4) in global?
    # My helper takes b_j_off, b_k_off.
    # b_j_off=0, b_k_off=4.
    # y_coeffs should be identity for the 2x2 block.
    # {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    
    # I already called it with wrong coeffs? No, I haven't called it yet.
    
    # P1 = A11 B_L
    add_strassen_product_multi({(0,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 0, 0, 0, 4, ([4, 5], [0, 1]), [1, 1])
    
    # P2 = A12 B_R
    # A12 is block (0,2).
    # B_R is block (2, 4).
    add_strassen_product_multi({(0,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 0, 2, 2, 4, ([4, 5], [0, 1]), [1, 1])
    
    # P3 = A21 B_L
    # A21 is block (2,0).
    # B_L is block (0, 4).
    add_strassen_product_multi({(0,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 2, 0, 0, 4, ([4, 5], [2, 3]), [1, 1])
    
    # P4 = A22 B_R
    # A22 is block (2,2).
    # B_R is block (2, 4).
    add_strassen_product_multi({(0,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 2, 2, 2, 4, ([4, 5], [2, 3]), [1, 1])
    
    # Part 3: 4x4x1 (Rank 16)
    # Schoolbook for col 6.
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
