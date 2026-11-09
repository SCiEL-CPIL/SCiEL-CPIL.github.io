import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1. Total 65.
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0] * rank for _ in range(p * n)] # 20 x 65
    
    col = 0
    
    # Helper to add a term for 2x2 Strassen
    # u_offsets: list of (i, j) top-lefts of 2x2 A blocks
    # v_offsets: list of (j, k) top-lefts of 2x2 B blocks
    # w_offset: (k, i) top-left of 2x2 C block
    def add_2x2_strassen(u_offsets, v_offsets, w_offset):
        nonlocal col
        
        # The 7 Strassen terms for 2x2 multiplication
        # Each term is defined by U, V, W vectors of length 4 (for 2x2 blocks)
        # Indices in 2x2 block: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1 = (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1] (A00, A01, A10, A11) -> indices 0, 3
        # V: [1, 0, 0, 1] -> indices 0, 3
        # W: [1, 0, 0, 1] -> C00, C11
        # Wait, W maps to C.
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the 7 terms explicitly.
        # Term 1: (A00+A11)(B00+B11)
        # Term 2: (A10+A11)B00
        # Term 3: A00(B01-B11)
        # Term 4: A11(B10-B00)
        # Term 5: (A00+A01)B11
        # Term 6: (A10-A00)(B00+B01)
        # Term 7: (A01-A11)(B10+B11)
        
        # We will store the coefficients for U, V, W for each term.
        # U and V are 4-dimensional (for the 2x2 block entries).
        # W is 4-dimensional (for the 2x2 C block entries).
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # M2 -> C10, C11? 
            # Let's check M2 contribution.
            # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
            # So M2 contributes +1 to C10 and -1 to C11.
            # My W vector for M2 should be [0, 0, 1, -1]?
            # Let's re-verify Strassen formulas.
            # C00 = M1 - M4 - M6 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for each M, we define its U, V, and the W coefficients (which can be negative).
            # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (Contributes +1 to C00, +1 to C11)
            # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1] (Contributes +1 to C10, -1 to C11)
            # M3: U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,1] (Contributes +1 to C01, +1 to C11)
            # M4: U=[0,0,0,1], V=[0,0,1,-1], W=[-1,0,1,0] (Contributes -1 to C00, +1 to C10)
            # M5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0] (Contributes +1 to C01) -- Wait, C01 = M3 + M5.
            #    Let's check M5. (A00+A01)B11.
            #    Result is added to C01?
            #    Formula: C01 = M3 + M5.
            #    So W for M5 should be [0,1,0,0].
            # M6: U=[0,0,1,-1], V=[1,1,0,0], W=[0,0,0,1] (Contributes +1 to C11) -- Wait, C11 has +M6.
            #    C00 has -M6.
            #    So W for M6 should be [-1,0,0,1]?
            #    Let's check C00 = M1 - M4 - M6 + M7.
            #    So M6 contributes -1 to C00 and +1 to C11.
            #    W for M6: [-1, 0, 0, 1].
            # M7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0] (Contributes +1 to C00)
            #    C00 has +M7.
            #    So W for M7: [1, 0, 0, 0].
        ]
        
        # Corrected W vectors:
        # M1: [1, 0, 0, 1]
        # M2: [0, 0, 1, -1]
        # M3: [0, 1, 0, 1]
        # M4: [-1, 0, 1, 0]
        # M5: [0, 1, 0, 0]
        # M6: [-1, 0, 0, 1]
        # M7: [1, 0, 0, 0]
        
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]), # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [-1, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
        ]
        
        # Map 2x2 block indices to local 0..3 indices
        # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        for u_coeff, v_coeff, w_coeff in strassen_terms:
            # Add this term to the global U, V, W
            # U part:
            for idx_a, coeff_a in enumerate(u_coeff):
                if coeff_a != 0:
                    # idx_a is 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1) in the 2x2 block
                    local_i = idx_a // 2
                    local_j = idx_a % 2
                    for u_off in u_offsets:
                        global_i = u_off[0] + local_i
                        global_j = u_off[1] + local_j
                        u_idx = global_i * m + global_j
                        U[u_idx][col] += coeff_a
            
            # V part:
            for idx_b, coeff_b in enumerate(v_coeff):
                if coeff_b != 0:
                    local_i = idx_b // 2
                    local_j = idx_b % 2
                    for v_off in v_offsets:
                        global_i = v_off[0] + local_i
                        global_j = v_off[1] + local_j
                        v_idx = global_i * p + global_j
                        V[v_idx][col] += coeff_b
                        
            # W part:
            for idx_c, coeff_c in enumerate(w_coeff):
                if coeff_c != 0:
                    local_i = idx_c // 2
                    local_j = idx_c % 2
                    # w_offset is (k, i) top-left of C block
                    global_k = w_offset[0] + local_i
                    global_i = w_offset[1] + local_j
                    w_idx = global_k * n + global_i
                    W[w_idx][col] += coeff_c
            
            col += 1
            
    # Apply top-level Strassen for 4x4x4
    # A blocks:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B blocks:
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C blocks:
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    add_2x2_strassen([(0,0), (2,2)], [(0,0), (2,2)], (0,0))
    # Wait, the result of M1 is added to C00 and C11.
    # My `add_2x2_strassen` adds the result to the C block specified by `w_offset`.
    # But M1 contributes to TWO C blocks.
    # The Strassen formula says:
    # C00 = M1 - M4 - M6 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 is added to C00 and C11.
    # My `add_2x2_strassen` computes the product and adds it to ONE C block.
    # So I need to call it twice for M1?
    # No, the decomposition is linear.
    # The term for M1 is $u \otimes v \otimes w$.
    # If I call `add_2x2_strassen` with `w_offset=(0,0)`, it adds terms with W vectors pointing to C00.
    # If I call it again with `w_offset=(2,2)`, it adds terms with W vectors pointing to C11.
    # But the U and V vectors are the same.
    # So I would be adding $u \otimes v \otimes w_{C00}$ and $u \otimes v \otimes w_{C11}$.
    # This is correct!
    # However, I must ensure that the U and V vectors are identical in both calls.
    # In my implementation, `u_offsets` and `v_offsets` determine U and V.
    # So for M1, I should call:
    # `add_2x2_strassen([(0,0), (2,2)], [(0,0), (2,2)], (0,0))`
    # `add_2x2_strassen([(0,0), (2,2)], [(0,0), (2,2)], (2,2))`
    # This will add 7 terms for C00 part and 7 terms for C11 part.
    # Total 14 terms for M1?
    # No, the rank of M1 is 7.
    # The contribution to C is $M1_{C00} + M1_{C11}$.
    # This is not a direct sum of two independent rank-7 tensors in the same way.
    # Actually, $M1$ is a single tensor of rank 7 (the $2 \times 2$ multiplication tensor).
    # Its output is a $2 \times 2$ matrix.
    # In the global problem, this $2 \times 2$ matrix is placed in C00 AND added to C11.
    # So the W vector for each of the 7 rank-1 components of M1 should have non-zero entries in BOTH C00 and C11.
    # My current `add_2x2_strassen` puts the W entries in only ONE block.
    # So I need to modify `add_2x2_strassen` to accept a list of W offsets, or modify the loop to add to multiple W offsets.
    
    # Let's refactor `add_2x2_strassen` to take `w_offsets` (list).
    
    # Redefine add_2x2_strassen
    def add_2x2_strassen_v2(u_offsets, v_offsets, w_offsets):
        nonlocal col
        
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]), # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [-1, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
        ]
        
        for u_coeff, v_coeff, w_coeff in strassen_terms:
            # U part
            for idx_a, coeff_a in enumerate(u_coeff):
                if coeff_a != 0:
                    local_i = idx_a // 2
                    local_j = idx_a % 2
                    for u_off in u_offsets:
                        global_i = u_off[0] + local_i
                        global_j = u_off[1] + local_j
                        u_idx = global_i * m + global_j
                        U[u_idx][col] += coeff_a
            
            # V part
            for idx_b, coeff_b in enumerate(v_coeff):
                if coeff_b != 0:
                    local_i = idx_b // 2
                    local_j = idx_b % 2
                    for v_off in v_offsets:
                        global_i = v_off[0] + local_i
                        global_j = v_off[1] + local_j
                        v_idx = global_i * p + global_j
                        V[v_idx][col] += coeff_b
                        
            # W part
            for idx_c, coeff_c in enumerate(w_coeff):
                if coeff_c != 0:
                    local_i = idx_c // 2
                    local_j = idx_c % 2
                    for w_off in w_offsets:
                        global_k = w_off[0] + local_i
                        global_i = w_off[1] + local_j
                        w_idx = global_k * n + global_i
                        W[w_idx][col] += coeff_c
            
            col += 1

    # Now apply top-level Strassen
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    add_2x2_strassen_v2([(0,0), (2,2)], [(0,0), (2,2)], [(0,0), (2,2)])
    
    # M2 = (A10+A11)B00 -> C10, C11 (with -1 for C11)
    # Wait, M2 contributes +1 to C10 and -1 to C11.
    # My `add_2x2_strassen_v2` adds the W vector as is.
    # The W vector for M2 in the local 2x2 decomposition is [0,0,1,-1].
    # This means it adds +1 to local C10 and -1 to local C11.
    # So if I pass `w_offsets=[(2,0), (2,2)]`, it will add:
    # +1 to global C10 (which is local C10 of block (2,0))
    # -1 to global C11 (which is local C11 of block (2,2))
    # This is exactly what we want!
    add_2x2_strassen_v2([(2,0), (2,2)], [(0,0)], [(2,0), (2,2)])
    
    # M3 = A00(B01-B11) -> C01, C11
    # Local W for M3 is [0,1,0,1] (+1 to C01, +1 to C11)
    add_2x2_strassen_v2([(0,0)], [(0,2), (2,2)], [(0,2), (2,2)])
    
    # M4 = A11(B10-B00) -> C00, C10
    # Local W for M4 is [-1,0,1,0] (-1 to C00, +1 to C10)
    add_2x2_strassen_v2([(2,2)], [(2,0), (0,0)], [(0,0), (2,0)])
    
    # M5 = (A00+A01)B11 -> C01
    # Local W for M5 is [0,1,0,0] (+1 to C01)
    add_2x2_strassen_v2([(0,0), (0,2)], [(2,2)], [(0,2)])
    
    # M6 = (A10-A00)(B00+B01) -> C00, C11
    # Local W for M6 is [-1,0,0,1] (-1 to C00, +1 to C11)
    add_2x2_strassen_v2([(2,0), (0,0)], [(0,0), (0,2)], [(0,0), (2,2)])
    
    # M7 = (A01-A11)(B10+B11) -> C00
    # Local W for M7 is [1,0,0,0] (+1 to C00)
    add_2x2_strassen_v2([(0,2), (2,2)], [(2,0), (2,2)], [(0,0)])
    
    # Now add the 4x4x1 part (rank 16)
    # C_{4, i} = sum_{j=0}^3 A_{ij} B_{j, 4}
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

