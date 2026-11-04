import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 9
    
    # We'll build U, V, W as lists of lists
    # U: (16, rank), V: (36, rank), W: (36, rank)
    
    terms = []  # Each term is (U_vec, V_vec, W_vec)
    
    # Helper: 2x2 Strassen terms for indices (i0,j0) in A, (j0,k0) in B, (k0,i0) in C
    # i0, j0, k0 are the base indices (0 or 2 for 4x4)
    
    def strassen_2x2(i0, j0, k0, scale_u=None, scale_v=None, scale_w=None):
        """Generate 7 Strassen terms for a 2x2*2x2 block at positions (i0,j0)-(i0+1,j0+1) in A,
        (j0,k0)-(j0+1,k0+1) in B, (k0,i0)-(k0+1,i0+1) in C.
        scale_u, scale_v, scale_w are optional scaling factors for the U, V, W vectors."""
        if scale_u is None: scale_u = 1
        if scale_v is None: scale_v = 1
        if scale_w is None: scale_w = 1
        
        # 2x2 Strassen U vectors (for A indices)
        # A local: [[a00, a01], [a10, a11]] at positions (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        u_terms = [
            [1, 0, 0, 1],   # P1: a00+a11
            [0, 0, 1, 1],   # P2: a10+a11
            [1, 0, 0, 0],   # P3: a00
            [0, 0, 0, 1],   # P4: a11
            [1, 1, 0, 0],   # P5: a00+a01
            [-1, 0, 1, 0],  # P6: a10-a00
            [0, 1, 0, -1],  # P7: a01-a11
        ]
        
        # 2x2 Strassen V vectors (for B indices)
        v_terms = [
            [1, 0, 0, 1],   # P1: b00+b11
            [1, 0, 0, 0],   # P2: b00
            [0, 1, 0, -1],  # P3: b01-b11
            [-1, 0, 1, 0],  # P4: b10-b00
            [0, 0, 0, 1],   # P5: b11
            [1, 1, 0, 0],   # P6: b00+b01
            [0, 0, 1, 1],   # P7: b10+b11
        ]
        
        # 2x2 Strassen W vectors (for C indices)
        w_terms = [
            [1, 0, 0, 1],   # P1: C00+C11
            [0, 0, 1, -1],  # P2: C10-C11
            [0, 1, 0, 1],   # P3: C01+C11
            [1, 0, 1, 0],   # P4: C00+C10
            [-1, 1, 0, 0],  # P5: -C00+C01
            [0, 0, 0, 1],   # P6: C11
            [1, 0, 0, 0],   # P7: C00
        ]
        
        result = []
        for t in range(7):
            # Build full U vector (16 entries)
            u_full = [0] * 16
            # A local index (a,b) maps to global (i0+a, j0+b)
            a_local = [0, 1, 2, 3]  # (0,0), (0,1), (1,0), (1,1)
            for li, val in enumerate(u_terms[t]):
                if val != 0:
                    a, b = divmod(li, 2)
                    gi, gj = i0 + a, j0 + b
                    u_full[gi * 4 + gj] = val * scale_u
            
            # Build full V vector (36 entries)
            v_full = [0] * 36
            for li, val in enumerate(v_terms[t]):
                if val != 0:
                    a, b = divmod(li, 2)
                    gj, gk = j0 + a, k0 + b
                    v_full[gj * 9 + gk] = val * scale_v
            
            # Build full W vector (36 entries)
            w_full = [0] * 36
            for li, val in enumerate(w_terms[t]):
                if val != 0:
                    a, b = divmod(li, 2)
                    gk, gi = k0 + a, i0 + b
                    w_full[gk * 4 + gi] = val * scale_w
            
            result.append((u_full, v_full, w_full))
        
        return result
    
    # Strassen 4x4*4x4 for B1 (columns 0-3)
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 is at (0,0)-(1,1) and (2,2)-(3,3)
    # This is not a simple 2x2 block, it's a sum of two 2x2 blocks
    
    # I need a different approach. Let me define the 4x4 Strassen terms directly.
    
    # For 4x4*4x4 Strassen:
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x2)
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 has entries:
    #   (0,0): A[0,0]+A[2,2], (0,1): A[0,1]+A[2,3]
    #   (1,0): A[1,0]+A[3,2], (1,1): A[1,1]+A[3,3]
    # B11+B22 has entries:
    #   (0,0): B[0,0]+B[2,2], (0,1): B[0,1]+B[2,3]
    #   (1,0): B[1,0]+B[3,2], (1,1): B[1,1]+B[3,3]
    
    # This is a 2x2*2x2 product where the "A" matrix is A11+A22 and "B" matrix is B11+B22.
    # But A11+A22 is not a contiguous 2x2 block in A, so I need to handle it specially.
    
    # Let me define a more general function that takes explicit U and V vectors.
    
    def strassen_2x2_explicit(u_base, v_base, w_base, i0, j0, k0):
        """Generate 7 Strassen terms given explicit 4-entry vectors for U, V, W bases.
        u_base[li] is the coefficient for A[i0+a, j0+b] where li = a*2+b
        v_base[li] is the coefficient for B[j0+a, k0+b]
        w_base[li] is the coefficient for C[k0+a, i0+b]"""
        u_terms = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ]
        v_terms = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        w_terms = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        
        result = []
        for t in range(7):
            u_full = [0] * 16
            for li, val in enumerate(u_terms[t]):
                if val != 0:
                    a, b = divmod(li, 2)
                    u_full[(i0+a) * 4 + (j0+b)] = val * u_base[li]
            
            v_full = [0] * 36
            for li, val in enumerate(v_terms[t]):
                if val != 0:
                    a, b = divmod(li, 2)
                    v_full[(j0+a) * 9 + (k0+b)] = val * v_base[li]
            
            w_full = [0] * 36
            for li, val in enumerate(w_terms[t]):
                if val != 0:
                    a, b = divmod(li, 2)
                    w_full[(k0+a) * 4 + (i0+b)] = val * w_base[li]
            
            result.append((u_full, v_full, w_full))
        
        return result
    
    # For M1 = (A11+A22)(B11+B22):
    # A11+A22: u_base for the "2x2" A matrix
    # The 2x2 A matrix has entries at (0,0), (0,1), (1,0), (1,1) corresponding to:
    # (0,0): A[0,0]+A[2,2] → u_base[0] = 1 (for A[0,0]) + 1 (for A[2,2])
    # But this isn't a single position, it's a sum.
    
    # I think I need to handle this differently. The Strassen 4x4*4x4 is not simply
    # 7 applications of 2x2 Strassen. It's a specific 49-term decomposition.
    
    # Let me look up the explicit 49-term decomposition for 4x4*4x4.
    
    # Actually, I recall that the 4x4*4x4 rank-49 decomposition can be constructed
    # by viewing it as a 2x2 block matrix multiplication and using Strassen's 7 products,
    # where each product is a 2x2*2x2 multiplication (rank 7).
    
    # The key is that the "A" and "B" matrices in each Strassen product are linear
    # combinations of the 2x2 blocks, which are themselves 2x2 matrices.
    
    # So for M1 = (A11+A22)(B11+B22):
    # The "A" matrix is A11+A22, which has entries:
    #   [0,0]: A[0,0]+A[2,2]
    #   [0,1]: A[0,1]+A[2,3]
    #   [1,0]: A[1,0]+A[3,2]
    #   [1,1]: A[1,1]+A[3,3]
    
    # This is a 2x2 matrix, and we apply the 2x2 Strassen to it.
    # The U vector for each Strassen term will have non-zero entries at the positions
    # corresponding to the A entries involved.
    
    # For the 2x2 Strassen term P1 of M1:
    # U has 1 at local (0,0) and (1,1) of the "A11+A22" matrix
    # Which corresponds to global A positions: (0,0), (2,2) for local (0,0)
    # and (0,1), (2,3) for local (0,1)... wait, no.
    
    # The "A11+A22" 2x2 matrix has:
    # local (0,0) = A[0,0] + A[2,2]
    # local (0,1) = A[0,1] + A[2,3]
    # local (1,0) = A[1,0] + A[3,2]
    # local (1,1) = A[1,1] + A[3,3]
    
    # So for P1 of M1, U has 1 at local (0,0) and (1,1):
    # U[A[0,0]] += 1, U[A[2,2]] += 1, U[A[1,1]] += 1, U[A[3,3]] += 1
    
    # This means the U vector for this term has 1 at positions 0, 3, 3, 11 (in 4x4 indexing)
    # Wait, A[0,0] is at index 0*4+0=0, A[2,2] is at 2*4+2=10, A[1,1] is at 1*4+1=5, A[3,3] is at 3*4+3=15
    
    # So U = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    
    # Similarly for V and W.
    
    # Let me define a function that, given a 2x2 "virtual" matrix defined by a mapping from
    # local (a,b) to a list of (global_i, global_j) positions with coefficients,
    # generates the U vector.
    
    def make_u_from_mapping(mapping):
        """mapping: dict from local (a,b) to list of (gi, gj, coeff)"""
        u = [0] * 16
        for (a, b), entries in mapping.items():
            for gi, gj, coeff in entries:
                u[gi * 4 + gj] += coeff
        return u
    
    def make_v_from_mapping(mapping):
        """mapping: dict from local (a,b) to list of (gj, gk, coeff)"""
        v = [0] * 36
        for (a, b), entries in mapping.items():
            for gj, gk, coeff in entries:
                v[gj * 9 + gk] += coeff
        return v
    
    def make_w_from_mapping(mapping):
        """mapping: dict from local (a,b) to list of (gk, gi, coeff)"""
        w = [0] * 36
        for (a, b), entries in mapping.items():
            for gk, gi, coeff in entries:
                w[gk * 4 + gi] += coeff
        return w
    
    # Define the 2x2 Strassen U, V, W patterns
    strassen_u = [
        [(0,0), (1,1)],      # P1
        [(1,0), (1,1)],      # P2
        [(0,0)],             # P3
        [(1,1)],             # P4
        [(0,0), (0,1)],      # P5
        [(1,0), (0,0)],      # P6 (with -1 for (0,0))
        [(0,1), (1,1)],      # P7 (with -1 for (1,1))
    ]
    strassen_u_coeffs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    
    strassen_v = [
        [(0,0), (1,1)],      # P1
        [(0,0)],             # P2
        [(0,1), (1,1)],      # P3
        [(1,0), (0,0)],      # P4
        [(1,1)],             # P5
        [(0,0), (0,1)],      # P6
        [(1,0), (1,1)],      # P7
    ]
    strassen_v_coeffs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    
    strassen_w = [
        [(0,0), (1,1)],      # P1
        [(1,0), (1,1)],      # P2
        [(0,1), (1,1)],      # P3
        [(0,0), (1,0)],      # P4
        [(0,0), (0,1)],      # P5
        [(1,1)],             # P6
        [(0,0)],             # P7
    ]
    strassen_w_coeffs = [
        [1, 1],
        [1, -1],
        [1, 1],
        [1, 1],
        [-1, 1],
        [1],
        [1],
    ]
    
    # Now define the 7 Strassen products for 4x4*4x4:
    # Each product is defined by a mapping for the "A" matrix, "B" matrix, and "C" matrix.
    
    # M1 = (A11+A22)(B11+B22)
    # A mapping: local (a,b) → A entries
    M1_A = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)],
    }
    M1_B = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)],
    }
    M1_C = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)],
    }
    
    # M2 = (A21+A22)B11
    M2_A = {
        (0,0): [(2,0,1), (2,2,1)],
        (0,1): [(2,1,1), (2,3,1)],
        (1,0): [(3,0,1), (3,2,1)],
        (1,1): [(3,1,1), (3,3,1)],
    }
    M2_B = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    M2_C = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    
    # M3 = A11(B12-B22)
    M3_A = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    M3_B = {
        (0,0): [(0,2,1), (2,2,-1)],
        (0,1): [(0,3,1), (2,3,-1)],
        (1,0): [(1,2,1), (3,2,-1)],
        (1,1): [(1,3,1), (3,3,-1)],
    }
    M3_C = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    
    # M4 = A22(B21-B11)
    M4_A = {
        (0,0): [(2,2,1)],
        (0,1): [(2,3,1)],
        (1,0): [(3,2,1)],
        (1,1): [(3,3,1)],
    }
    M4_B = {
        (0,0): [(2,0,1), (0,0,-1)],
        (0,1): [(2,1,1), (0,1,-1)],
        (1,0): [(3,0,1), (1,0,-1)],
        (1,1): [(3,1,1), (1,1,-1)],
    }
    M4_C = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    
    # M5 = (A11+A12)B22
    M5_A = {
        (0,0): [(0,0,1), (0,2,1)],
        (0,1): [(0,1,1), (0,3,1)],
        (1,0): [(1,0,1), (1,2,1)],
        (1,1): [(1,1,1), (1,3,1)],
    }
    M5_B = {
        (0,0): [(2,2,1)],
        (0,1): [(2,3,1)],
        (1,0): [(3,2,1)],
        (1,1): [(3,3,1)],
    }
    M5_C = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    
    # M6 = (A21-A11)(B11+B12)
    M6_A = {
        (0,0): [(2,0,1), (0,0,-1)],
        (0,1): [(2,1,1), (0,1,-1)],
        (1,0): [(3,0,1), (1,0,-1)],
        (1,1): [(3,1,1), (1,1,-1)],
    }
    M6_B = {
        (0,0): [(0,0,1), (0,2,1)],
        (0,1): [(0,1,1), (0,3,1)],
        (1,0): [(1,0,1), (1,2,1)],
        (1,1): [(1,1,1), (1,3,1)],
    }
    M6_C = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    
    # M7 = (A12-A22)(B21+B22)
    M7_A = {
        (0,0): [(0,2,1), (2,2,-1)],
        (0,1): [(0,3,1), (2,3,-1)],
        (1,0): [(1,2,1), (3,2,-1)],
        (1,1): [(1,3,1), (3,3,-1)],
    }
    M7_B = {
        (0,0): [(2,0,1), (2,2,1)],
        (0,1): [(2,1,1), (2,3,1)],
        (1,0): [(3,0,1), (3,2,1)],
        (1,1): [(3,1,1), (3,3,1)],
    }
    M7_C = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)],
    }
    
    products = [
        (M1_A, M1_B, M1_C),
        (M2_A, M2_B, M2_C),
        (M3_A, M3_B, M3_C),
        (M4_A, M4_B, M4_C),
        (M5_A, M5_B, M5_C),
        (M6_A, M6_B, M6_C),
        (M7_A, M7_B, M7_C),
    ]
    
    # Generate all 49 terms for 4x4*4x4
    def generate_4x4_strassen(col_offset):
        """Generate 49 terms for 4x4*4x4 where B columns are col_offset to col_offset+3."""
        terms = []
        for A_map, B_map, C_map in products:
            for t in range(7):
                u_full = [0] * 16
                for (a, b), entries in A_map.items():
                    for gi, gj, coeff in entries:
                        # Apply Strassen U pattern
                        pass
                
                # I need to apply the Strassen U, V, W patterns to the mappings.
                # For each Strassen term t, the U vector is determined by which local
                # positions are selected by strassen_u[t], and for each selected position,
                # we add the corresponding global entries from A_map.
                
                u_full = [0] * 16
                for (a, b), coeff in zip(strassen_u[t], strassen_u_coeffs[t]):
                    if coeff != 0:
                        for gi, gj, c in A_map[(a, b)]:
                            u_full[gi * 4 + gj] += coeff * c
                
                v_full = [0] * 36
                for (a, b), coeff in zip(strassen_v[t], strassen_v_coeffs[t]):
                    if coeff != 0:
                        for gj, gk, c in B_map[(a, b)]:
                            v_full[gj * 9 + (gk + col_offset)] += coeff * c
                
                w_full = [0] * 36
                for (a, b), coeff in zip(strassen_w[t], strassen_w_coeffs[t]):
                    if coeff != 0:
                        for gk, gi, c in C_map[(a, b)]:
                            w_full[(gk + col_offset) * 4 + gi] += coeff * c
                
                terms.append((u_full, v_full, w_full))
        
        return terms
    
    # Generate terms for C1 = A*B1 (B1 is columns 0-3)
    terms_c1 = generate_4x4_strassen(0)
    
    # Generate terms for C2 = A*B2 (B2 is columns 4-7)
    terms_c2 = generate_4x4_strassen(4)
    
    # Generate terms for C3 = A*B3 (B3 is column 8) - schoolbook
    terms_c3 = []
    for i in range(4):
        for j in range(4):
            u_full = [0] * 16
            u_full[i * 4 + j] = 1
            v_full = [0] * 36
            v_full[j * 9 + 8] = 1
            w_full = [0] * 36
            w_full[8 * 4 + i] = 1
            terms_c3.append((u_full, v_full, w_full))
    
    all_terms = terms_c1 + terms_c2 + terms_c3
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for i in range(16):
            if u[i] != 0:
                U[i][c] = u[i]
        for i in range(36):
            if v[i] != 0:
                V[i][c] = v[i]
            if w[i] != 0:
                W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}
