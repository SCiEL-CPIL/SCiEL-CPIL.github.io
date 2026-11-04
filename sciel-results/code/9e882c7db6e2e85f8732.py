import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Helper to create a rank-1 term
    # u is a vector of length n*m, v is m*p, w is p*n
    # We represent the decomposition as a list of (u, v, w) tuples
    # where u, v, w are sparse vectors (dicts or lists)
    
    def make_term(u_dict, v_dict, w_dict, n, m, p):
        # u_dict: {index: value}
        # v_dict: {index: value}
        # w_dict: {index: value}
        pass

    # Strassen 2x2 Decomposition
    # Indices for 2x2:
    # U: i*m+j -> 0,1,2,3
    # V: j*p+k -> 0,1,2,3
    # W: k*n+i -> 0,1,2,3
    
    # We need to define the 7 terms for 2x2.
    # Each term is (u, v, w)
    # u, v, w are vectors of length 4.
    
    strassen_terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0) idx 0, A22: (1,1) idx 3
    # B11: (0,0) idx 0, B22: (1,1) idx 3
    # u: A11+A22 -> [1,0,0,1]
    # v: B11+B22 -> [1,0,0,1]
    # w: C11+=M1, C12+=M1, C21+=M1, C22+=M1?
    # Wait, Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to decompose each M_k into 7 rank-1 terms.
    # This is recursive.
    
    # Let's implement a generic function to build the tensor decomposition
    # for n x n using Strassen if n is even, or schoolbook if n=1.
    
    def build_decomp(n):
        if n == 1:
            # Rank 1
            # U[0]=1, V[0]=1, W[0]=1
            return [({0: 1}, {0: 1}, {0: 1})]
        
        if n % 2 != 0:
            # Fallback to schoolbook for odd n > 1?
            # Or use the 4+1 split for n=5?
            # For now, let's assume we call this with n=4 and n=1.
            pass

        # n is even
        half = n // 2
        # Get decomposition for half x half
        sub_decomp = build_decomp(half)
        
        # Strassen 7 products
        # We need to map the sub-decomp terms to the full n x n indices.
        
        # Helper to map sub-indices to full-indices
        # Sub U index: i_sub * half + j_sub
        # Full U index: i_full * n + j_full
        # i_full = i_sub if block is top, else i_sub + half
        # j_full = j_sub if block is left, else j_sub + half
        
        def map_u(i_block, j_block, idx_sub):
            i_sub, j_sub = divmod(idx_sub, half)
            i_full = i_sub + i_block * half
            j_full = j_sub + j_block * half
            return i_full * n + j_full

        def map_v(i_block, j_block, idx_sub):
            # V index: j * p + k
            # In square case p=n
            # j_sub, k_sub
            i_sub, j_sub = divmod(idx_sub, half)
            j_full = i_sub + i_block * half
            k_full = j_sub + j_block * half
            return j_full * n + k_full

        def map_w(i_block, j_block, idx_sub):
            # W index: k * n + i
            # k_sub, i_sub
            i_sub, j_sub = divmod(idx_sub, half)
            k_full = i_sub + i_block * half
            i_full = j_sub + j_block * half
            return k_full * n + i_full

        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        # u: A11+A22. Blocks (0,0) and (1,1)
        # v: B11+B22. Blocks (0,0) and (1,1)
        # w: C11, C12, C21, C22 all get +M1?
        # No, the result of M1 is added to specific blocks of C.
        # C11 += M1, C12 += M1, C21 += M1, C22 += M1?
        # Let's check Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C11, C12, C21, C22?
        # Wait, M1 is a matrix.
        # C11 gets M1. C12 gets M1?
        # Yes, the formula says C11 = ... + M1.
        # So the block (0,0) of C gets M1.
        # Block (0,1) of C gets M1?
        # C12 = M3 + M5. M1 is not there.
        # So M1 only contributes to C11?
        # Let's re-read Strassen.
        # M1 = (A11+A22)(B11+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C11 and C22.
        # M2 contributes to C21 and C22 (with -).
        # M3 contributes to C12 and C22.
        # M4 contributes to C11 and C21.
        # M5 contributes to C12 (with +) and C11 (with -).
        # M6 contributes to C22.
        # M7 contributes to C11.
        
        # For each M_k, we have a decomposition into sub-terms.
        # Each sub-term (u, v, w) from build_decomp(half)
        # u is the input for the "left" matrix part.
        # v is the input for the "right" matrix part.
        # w is the output for the "result" matrix part.
        
        # For M1:
        # u: A11 + A22.
        # We need to form a vector u_full such that it picks up A11 and A22.
        # v: B11 + B22.
        # w: C11 and C22.
        
        # This means for each sub-term (u_sub, v_sub, w_sub):
        # u_full will have non-zeros at map_u(0,0, idx) and map_u(1,1, idx)
        # v_full will have non-zeros at map_v(0,0, idx) and map_v(1,1, idx)
        # w_full will have non-zeros at map_w(0,0, idx) and map_w(1,1, idx)
        
        # We can define the coefficients for u, v, w for each M_k.
        
        def add_term(u_coeffs, v_coeffs, w_coeffs, sub_terms):
            # u_coeffs: dict {block_idx: coeff}
            # block_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for u_sub, v_sub, w_sub in sub_terms:
                u_full = {}
                v_full = {}
                w_full = {}
                
                for idx, val in u_sub.items():
                    for b_idx, coeff in u_coeffs.items():
                        if coeff != 0:
                            i_b, j_b = divmod(b_idx, 2)
                            idx_full = map_u(i_b, j_b, idx)
                            u_full[idx_full] = u_full.get(idx_full, 0) + val * coeff
                
                for idx, val in v_sub.items():
                    for b_idx, coeff in v_coeffs.items():
                        if coeff != 0:
                            i_b, j_b = divmod(b_idx, 2)
                            idx_full = map_v(i_b, j_b, idx)
                            v_full[idx_full] = v_full.get(idx_full, 0) + val * coeff
                
                for idx, val in w_sub.items():
                    for b_idx, coeff in w_coeffs.items():
                        if coeff != 0:
                            i_b, j_b = divmod(b_idx, 2)
                            idx_full = map_w(i_b, j_b, idx)
                            w_full[idx_full] = w_full.get(idx_full, 0) + val * coeff
                
                if u_full and v_full and w_full:
                    terms.append((u_full, v_full, w_full))
        
        # M1: (A11+A22)(B11+B22)
        # u: A11 (block 0) + A22 (block 3)
        # v: B11 (block 0) + B22 (block 3)
        # w: C11 (block 0) + C22 (block 3)
        add_term({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}, sub_decomp)
        
        # M2: (A21+A22)B11
        # u: A21 (block 2) + A22 (block 3)
        # v: B11 (block 0)
        # w: C21 (block 2) + C22 (block 3, with -1)
        add_term({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}, sub_decomp)
        
        # M3: A11(B12-B22)
        # u: A11 (block 0)
        # v: B12 (block 1) - B22 (block 3)
        # w: C12 (block 1) + C22 (block 3)
        add_term({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}, sub_decomp)
        
        # M4: A22(B21-B11)
        # u: A22 (block 3)
        # v: B21 (block 2) - B11 (block 0)
        # w: C11 (block 0) + C21 (block 2)
        add_term({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}, sub_decomp)
        
        # M5: (A11+A12)B22
        # u: A11 (block 0) + A12 (block 1)
        # v: B22 (block 3)
        # w: C12 (block 1) - C11 (block 0, with -1)
        add_term({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}, sub_decomp)
        
        # M6: (A21-A11)(B11+B12)
        # u: A21 (block 2) - A11 (block 0)
        # v: B11 (block 0) + B12 (block 1)
        # w: C22 (block 3)
        add_term({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}, sub_decomp)
        
        # M7: (A12-A22)(B21+B22)
        # u: A12 (block 1) - A22 (block 3)
        # v: B21 (block 2) + B22 (block 3)
        # w: C11 (block 0)
        add_term({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}, sub_decomp)
        
        return terms

    # Build 4x4 decomposition
    decomp_4 = build_decomp(4)
    
    # Now build 5x5 using 4+1 split
    # We need to map 4x4 indices to 5x5 indices.
    # And handle the 1x1 and cross terms.
    
    n = 5
    m = 5
    p = 5
    
    final_terms = []
    
    # Helper to map 4x4 indices to 5x5
    # 4x4 indices are in 0..3
    # 5x5 indices are in 0..4
    # We place the 4x4 block at (0,0) to (3,3)
    
    def map_4_to_5_u(idx_4):
        i, j = divmod(idx_4, 4)
        return i * 5 + j
        
    def map_4_to_5_v(idx_4):
        j, k = divmod(idx_4, 4)
        return j * 5 + k
        
    def map_4_to_5_w(idx_4):
        k, i = divmod(idx_4, 4)
        return k * 5 + i

    # Add 4x4 terms
    for u, v, w in decomp_4:
        u_5 = {map_4_to_5_u(idx): val for idx, val in u.items()}
        v_5 = {map_4_to_5_v(idx): val for idx, val in v.items()}
        w_5 = {map_4_to_5_w(idx): val for idx, val in w.items()}
        final_terms.append((u_5, v_5, w_5))
    
    # Add cross terms
    # Schoolbook for the rest
    # A41 B14, A44 B41, A41 B11, A14 B44, A11 B14, A14 B41, A11 B11
    
    # Indices for 4x1 (A41): i in 0..3, j=4. U index: i*5+4
    # Indices for 1x4 (B14): i=4, j in 0..3. V index: 4*5+j
    # Indices for 4x4 (A44): i in 0..3, j in 0..3. U index: i*5+j
    
    # Term: A41 B14
    # u: A41 (i,4)
    # v: B14 (4,j)
    # w: C44? No, C44 is 4x4 block?
    # C = A B.
    # C_{ij} = sum A_{ik} B_{kj}
    # A41 is rows 0-3, col 4.
    # B14 is row 4, cols 0-3.
    # Product contributes to C_{0-3, 0-3}.
    # So w is in the 4x4 block.
    
    # We can just iterate i, j, k for the remaining parts.
    # But we already have the 4x4 part covered by decomp_4.
    # decomp_4 covers A44 B44.
    # We need:
    # 1. A41 B14 -> C44 (4x4 block)
    # 2. A44 B41 -> C41 (4x1 block)
    # 3. A41 B11 -> C41 (4x1 block)
    # 4. A14 B44 -> C14 (1x4 block)
    # 5. A11 B14 -> C14 (1x4 block)
    # 6. A14 B41 -> C11 (1x1 block)
    # 7. A11 B11 -> C11 (1x1 block)
    
    # Let's add them one by one.
    
    # 1. A41 B14
    for i in range(4):
        for j in range(4):
            u = {i*5+4: 1}
            v = {4*5+j: 1}
            w = {j*5+i: 1} # C_{ij}
            final_terms.append((u, v, w))
            
    # 2. A44 B41
    for i in range(4):
        for k in range(4):
            u = {i*5+k: 1}
            v = {4*5+4: 1} # B_{4,4} is B41? No.
            # B41 is row 4, col 4? No.
            # B is 5x5.
            # B41 means Row 4, Col 1?
            # In my notation:
            # A41: Rows 0-3, Col 4.
            # B14: Row 4, Cols 0-3.
            # B41: Row 4, Col 4? No, B41 is Row 4, Col 1?
            # Let's stick to indices.
            # A_{i,k} where i<4, k<4.
            # B_{k,4} where k<4.
            # C_{i,4}.
            pass
    
    # Let's use a generic loop for the remaining 110 - 49 = 61 terms?
    # No, the count was 110.
    # 49 from A44 B44.
    # 16 from A41 B14.
    # 16 from A44 B41.
    # 4 from A41 B11.
    # 16 from A14 B44.
    # 4 from A11 B14.
    # 4 from A14 B41.
    # 1 from A11 B11.
    # Total 110.
    
    # 2. A44 B41 (4x4 * 4x1 -> 4x1)
    # A_{i,k}, B_{k,4}, C_{i,4}
    for i in range(4):
        for k in range(4):
            u = {i*5+k: 1}
            v = {k*5+4: 1}
            w = {4*5+i: 1}
            final_terms.append((u, v, w))
            
    # 3. A41 B11 (4x1 * 1x1 -> 4x1)
    # A_{i,4}, B_{4,4}, C_{i,4}
    for i in range(4):
        u = {i*5+4: 1}
        v = {4*5+4: 1}
        w = {4*5+i: 1}
        final_terms.append((u, v, w))
        
    # 4. A14 B44 (1x4 * 4x4 -> 1x4)
    # A_{4,k}, B_{k,j}, C_{4,j}
    for k in range(4):
        for j in range(4):
            u = {4*5+k: 1}
            v = {k*5+j: 1}
            w = {j*5+4: 1}
            final_terms.append((u, v, w))
            
    # 5. A11 B14 (1x1 * 1x4 -> 1x4)
    # A_{4,4}, B_{4,j}, C_{4,j}
    for j in range(4):
        u = {4*5+4: 1}
        v = {4*5+j: 1}
        w = {j*5+4: 1}
        final_terms.append((u, v, w))
        
    # 6. A14 B41 (1x4 * 4x1 -> 1x1)
    # A_{4,k}, B_{k,4}, C_{4,4}
    for k in range(4):
        u = {4*5+k: 1}
        v = {k*5+4: 1}
        w = {4*5+4: 1}
        final_terms.append((u, v, w))
        
    # 7. A11 B11 (1x1 * 1x1 -> 1x1)
    u = {4*5+4: 1}
    v = {4*5+4: 1}
    w = {4*5+4: 1}
    final_terms.append((u, v, w))
    
    # Construct U, V, W matrices
    rank = len(final_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(final_terms):
        for idx, val in u.items():
            U[idx][col] = val
        for idx, val in v.items():
            V[idx][col] = val
        for idx, val in w.items():
            W[idx][col] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
