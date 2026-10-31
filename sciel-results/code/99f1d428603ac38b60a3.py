"""Rewritten program for Matrix Multiplication 4x4x4 using Strassen's Algorithm (Rank 49)."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 (outer Strassen) * 7 (inner Strassen) = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping helpers
    # A is 4x4. Flattened index a = i * 4 + j (row-major)
    # B is 4x4. Flattened index b = j * 4 + k (row-major)
    # C is 4x4. Flattened index c = k * 4 + i (col-major as per seed W[k*n+i])
    
    # Define block indices
    # Block (bi, bj) of size 2x2 corresponds to:
    # A: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Flattened indices for A block (bi, bj):
    # (2*bi)*4 + 2*bj, (2*bi)*4 + 2*bj + 1, (2*bi+1)*4 + 2*bj, (2*bi+1)*4 + 2*bj + 1
    
    def get_a_indices(bi, bj):
        r0 = 2 * bi
        c0 = 2 * bj
        return [r0 * 4 + c0, r0 * 4 + c0 + 1, (r0 + 1) * 4 + c0, (r0 + 1) * 4 + c0 + 1]
        
    def get_b_indices(bi, bj):
        # B block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # Wait, B is multiplied as A * B.
        # A is n x m (4x4). B is m x p (4x4).
        # Block B_{ij} is rows 2*i to 2*i+1, cols 2*j to 2*j+1.
        # Flattened index in B is row * 4 + col.
        r0 = 2 * bi
        c0 = 2 * bj
        return [r0 * 4 + c0, r0 * 4 + c0 + 1, (r0 + 1) * 4 + c0, (r0 + 1) * 4 + c0 + 1]
        
    def get_c_indices(bi, bj):
        # C block (bi, bj) is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
        # Flattened index in W is col * 4 + row (col-major).
        # So index = c * 4 + r.
        r0 = 2 * bi
        c0 = 2 * bj
        return [c0 * 4 + r0, c0 * 4 + (r0 + 1), (c0 + 1) * 4 + r0, (c0 + 1) * 4 + (r0 + 1)]

    # Inner Strassen Linear Forms for 2x2 matrices
    # l=0: m1 = (x00+x11)(y00+y11)
    # l=1: m2 = (x10+x11)y00
    # l=2: m3 = x00(y10-y11)
    # l=3: m4 = x11(y01-y00)
    # l=4: m5 = (x00+x10)y11
    # l=5: m6 = (x01-x00)(y00+y10)
    # l=6: m7 = (x10-x11)(y11+y01)
    
    # Coefficients for U (X matrix) for each l and each local index (00, 01, 10, 11)
    # Local index 0: 00, 1: 01, 2: 10, 3: 11
    u_inner_coeffs = [
        [1, 0, 0, 1],  # l=0
        [0, 0, 1, 1],  # l=1
        [1, 0, 0, 0],  # l=2
        [0, 0, 0, 1],  # l=3
        [1, 0, 1, 0],  # l=4
        [-1, 1, 0, 0], # l=5
        [0, 0, 1, -1]  # l=6
    ]
    
    # Coefficients for V (Y matrix) for each l and each local index
    v_inner_coeffs = [
        [1, 0, 0, 1],  # l=0
        [1, 0, 0, 0],  # l=1
        [0, 0, 1, -1], # l=2
        [-1, 1, 0, 0], # l=3
        [0, 0, 0, 1],  # l=4
        [1, 0, 1, 0],  # l=5
        [0, 1, 0, 1]   # l=6
    ]
    
    # Coefficients for W (Z matrix output) for each l and each local index (00, 01, 10, 11)
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    w_inner_coeffs = [
        [1, 0, 0, 1],  # l=0 (m1)
        [0, 0, 1, -1], # l=1 (m2)
        [0, 1, 0, 1],  # l=2 (m3)
        [1, 0, 1, 0],  # l=3 (m4)
        [-1, 1, 0, 0], # l=4 (m5)
        [0, 0, 0, 1],  # l=5 (m6)
        [1, 0, 0, 0]   # l=6 (m7)
    ]
    
    # Outer Strassen Linear Forms
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B10-B11)
    # k=3: M4 = A11(B01-B00)
    # k=4: M5 = (A00+A10)B11
    # k=5: M6 = (A01-A00)(B00+B10)
    # k=6: M7 = (A10-A11)(B11+B01)
    
    # Coefficients for U (A blocks) for each k and each block (00, 01, 10, 11)
    u_outer_coeffs = [
        [1, 0, 0, 1],  # k=0
        [0, 0, 1, 1],  # k=1
        [1, 0, 0, 0],  # k=2
        [0, 0, 0, 1],  # k=3
        [1, 0, 1, 0],  # k=4
        [-1, 1, 0, 0], # k=5
        [0, 0, 1, -1]  # k=6
    ]
    
    # Coefficients for V (B blocks) for each k and each block
    v_outer_coeffs = [
        [1, 0, 0, 1],  # k=0
        [1, 0, 0, 0],  # k=1
        [0, 0, 1, -1], # k=2
        [-1, 1, 0, 0], # k=3
        [0, 0, 0, 1],  # k=4
        [1, 0, 1, 0],  # k=5
        [0, 1, 0, 1]   # k=6
    ]
    
    # Coefficients for W (C blocks) for each k and each block (00, 01, 10, 11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_outer_coeffs = [
        [1, 0, 0, 1],  # k=0 (M1)
        [0, 0, 1, -1], # k=1 (M2)
        [0, 1, 0, 1],  # k=2 (M3)
        [1, 0, 1, 0],  # k=3 (M4)
        [-1, 1, 0, 0], # k=4 (M5)
        [0, 0, 0, 1],  # k=5 (M6)
        [1, 0, 0, 0]   # k=6 (M7)
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            # For this rank component (k, l):
            # U gets contribution from Outer k and Inner l
            # V gets contribution from Outer k and Inner l
            # W gets contribution from Outer k and Inner l
            
            # Iterate over the 4 blocks of A/B/C
            for block in range(4):
                # Determine which blocks of A and B are involved in Outer step k
                # u_outer_coeffs[k][block] is the coefficient for A_block in X_k
                coeff_u_block = u_outer_coeffs[k][block]
                coeff_v_block = v_outer_coeffs[k][block]
                coeff_w_block = w_outer_coeffs[k][block]
                
                if coeff_u_block == 0 and coeff_v_block == 0 and coeff_w_block == 0:
                    continue
                
                # Map global block index to local 2x2 indices
                # block 0: (0,0), block 1: (0,1), block 2: (1,0), block 3: (1,1)
                bi = block // 2
                bj = block % 2
                
                # Get flattened indices for this block in A, B, C
                a_indices = get_a_indices(bi, bj)
                b_indices = get_b_indices(bi, bj)
                c_indices = get_c_indices(bi, bj)
                
                # For each local entry in the 2x2 block, apply inner coefficients
                for local_idx in range(4):
                    # Get specific flattened index
                    idx_a = a_indices[local_idx]
                    idx_b = b_indices[local_idx]
                    idx_c = c_indices[local_idx]
                    
                    # Get inner coefficients for this local entry
                    c_u_inner = u_inner_coeffs[l][local_idx]
                    c_v_inner = v_inner_coeffs[l][local_idx]
                    c_w_inner = w_inner_coeffs[l][local_idx]
                    
                    # Combine outer and inner coefficients
                    final_u_coeff = coeff_u_block * c_u_inner
                    final_v_coeff = coeff_v_block * c_v_inner
                    final_w_coeff = coeff_w_block * c_w_inner
                    
                    if final_u_coeff != 0:
                        U[idx_a][col] = final_u_coeff
                    if final_v_coeff != 0:
                        V[idx_b][col] = final_v_coeff
                    if final_w_coeff != 0:
                        W[idx_c][col] = final_w_coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
