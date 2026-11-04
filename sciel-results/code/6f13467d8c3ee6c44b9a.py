import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We are targeting rank 88
    # 2 blocks of (Strassen(2,2,2) + Schoolbook(2,3,2) + Schoolbook(2,2,3) + Laderman(2,3,3))
    # Ranks: 7 + 12 + 12 + 13 = 44 per block. Total 88.
    
    U = [[0] * 88 for _ in range(n * m)]
    V = [[0] * 88 for _ in range(m * p)]
    W = [[0] * 88 for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col_offset] += val
        V[v_idx][col_offset] += val
        W[w_idx][col_offset] += val

    # We need to process 2 blocks of rows (0,1) and (2,3)
    for block in range(2):
        i_base = block * 2
        # Sub-block 1: A[i_base:i_base+2, 0:2] * B[0:2, 0:2]
        # This is 2x2x2. Use Strassen.
        # A indices: (i_base, 0), (i_base, 1), (i_base+1, 0), (i_base+1, 1)
        # B indices: (0, 0), (0, 1), (1, 0), (1, 1)
        # C indices: (0, i_base), (0, i_base+1), (1, i_base), (1, i_base+1)
        
        # Strassen M1
        add_term(i_base * m + 0, 0 * p + 0, 0 * n + i_base, 1)
        add_term(i_base * m + 0, 0 * p + 0, 1 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 1, 0 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 1, 1 * n + i_base, 1)
        col_offset += 1
        
        # Strassen M2
        add_term(i_base * m + 1, 0 * p + 0, 0 * n + i_base, 1)
        add_term(i_base * m + 1, 0 * p + 0, 1 * n + i_base, 1)
        add_term(i_base+1 * m + 1, 0 * p + 0, 0 * n + i_base, 1)
        add_term(i_base+1 * m + 1, 0 * p + 0, 1 * n + i_base, 1)
        col_offset += 1
        
        # Strassen M3
        add_term(i_base * m + 0, 0 * p + 1, 0 * n + i_base, 1)
        add_term(i_base * m + 0, 0 * p + 1, 1 * n + i_base, 1)
        add_term(i_base * m + 0, 1 * p + 1, 0 * n + i_base, -1)
        add_term(i_base * m + 0, 1 * p + 1, 1 * n + i_base, -1)
        col_offset += 1
        
        # Strassen M4
        add_term(i_base+1 * m + 1, 1 * p + 0, 0 * n + i_base, 1)
        add_term(i_base+1 * m + 1, 1 * p + 0, 1 * n + i_base, 1)
        add_term(i_base+1 * m + 1, 0 * p + 0, 0 * n + i_base, -1)
        add_term(i_base+1 * m + 1, 0 * p + 0, 1 * n + i_base, -1)
        col_offset += 1
        
        # Strassen M5
        add_term(i_base * m + 0, 1 * p + 1, 0 * n + i_base, 1)
        add_term(i_base * m + 0, 1 * p + 1, 1 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 1, 0 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 1, 1 * n + i_base, 1)
        col_offset += 1
        
        # Strassen M6
        add_term(i_base+1 * m + 0, 0 * p + 0, 0 * n + i_base, 1)
        add_term(i_base+1 * m + 0, 0 * p + 0, 1 * n + i_base, 1)
        add_term(i_base+1 * m + 0, 0 * p + 1, 0 * n + i_base, 1)
        add_term(i_base+1 * m + 0, 0 * p + 1, 1 * n + i_base, 1)
        add_term(i_base * m + 0, 0 * p + 0, 0 * n + i_base, -1)
        add_term(i_base * m + 0, 0 * p + 0, 1 * n + i_base, -1)
        add_term(i_base * m + 0, 0 * p + 1, 0 * n + i_base, -1)
        add_term(i_base * m + 0, 0 * p + 1, 1 * n + i_base, -1)
        col_offset += 1
        
        # Strassen M7
        add_term(i_base * m + 1, 1 * p + 0, 0 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 0, 1 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 1, 0 * n + i_base, 1)
        add_term(i_base * m + 1, 1 * p + 1, 1 * n + i_base, 1)
        add_term(i_base+1 * m + 1, 1 * p + 0, 0 * n + i_base, -1)
        add_term(i_base+1 * m + 1, 1 * p + 0, 1 * n + i_base, -1)
        add_term(i_base+1 * m + 1, 1 * p + 1, 0 * n + i_base, -1)
        add_term(i_base+1 * m + 1, 1 * p + 1, 1 * n + i_base, -1)
        col_offset += 1

        # Sub-block 2: A[i_base:i_base+2, 2:5] * B[2:5, 0:2]
        # This is 2x3x2. Schoolbook.
        # A cols 2,3,4. B rows 2,3,4. C cols 0,1.
        for j in range(2, 5):
            for k in range(0, 2):
                for i in range(2):
                    add_term((i_base + i) * m + j, j * p + k, k * n + (i_base + i), 1)
                    col_offset += 1

        # Sub-block 3: A[i_base:i_base+2, 0:2] * B[0:2, 2:5]
        # This is 2x2x3. Schoolbook.
        # A cols 0,1. B rows 0,1. C cols 2,3,4.
        for j in range(0, 2):
            for k in range(2, 5):
                for i in range(2):
                    add_term((i_base + i) * m + j, j * p + k, k * n + (i_base + i), 1)
                    col_offset += 1

        # Sub-block 4: A[i_base:i_base+2, 2:5] * B[2:5, 2:5]
        # This is 2x3x3. Padded Laderman.
        # We need Laderman factors for 3x3x3.
        # A is 2x3 (pad to 3x3 with 0 row). B is 3x3. C is 2x3 (pad to 3x3 with 0 row).
        # We only care about C[0:2, 0:3].
        
        # Laderman factors for 3x3x3
        # M1 = (A00+A22)(B00+B22)
        # M2 = (A10+A22)B00
        # M3 = A00(B01-B21)
        # M4 = A02(B20-B00)
        # M5 = (A00+A02)B22
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A02-A22)(B20+B22)
        # M8 = (A11+A21)B01
        # M9 = A11(B10-B00)
        # M10 = A21(B12-B10)
        # M11 = (A11+A12)B21
        # M12 = (A21-A11)(B11+B12)
        # M13 = (A12-A21)(B21+B22)
        
        # Output:
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C02 = M8+M11
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        # C12 = M9+M12
        # C20 = M10+M13
        # C21 = M7-M10+M13
        # C22 = M1-M7+M8+M12
        
        # We map A rows 0,1 to global A rows i_base, i_base+1.
        # A col 0,1,2 in sub-block -> global cols 2,3,4.
        # B rows 0,1,2 in sub-block -> global rows 2,3,4.
        # B cols 0,1,2 in sub-block -> global cols 2,3,4.
        # C rows 0,1 in sub-block -> global C rows i_base, i_base+1.
        # C cols 0,1,2 in sub-block -> global C cols 2,3,4.
        
        # We will add terms for M1..M13.
        # Only terms that contribute to C[0:2, 0:3] are needed?
        # No, the decomposition is for the full tensor.
        # But we can just add all terms, and the W matrix will have entries for C[2, :] which we don't care about?
        # Wait, W has shape (p*n, rank) = (20, rank).
        # C[2, :] corresponds to k=2, i in [0,4).
        # In our sub-block, C row 2 is the padded row.
        # If we add terms for C[2, :], they will go into W[2*4 + i].
        # These entries are not part of the required output C[0:2, 0:3] for this sub-block?
        # The required output for this sub-block is C[i_base:i_base+2, 2:5].
        # So we only need W entries for k in [2,5) and i in [i_base, i_base+2).
        # The padded C row 2 (local) corresponds to global C row i_base+2? No.
        # The sub-block C is 2x3. We pad it to 3x3.
        # The 3rd row of the padded C is zero.
        # The decomposition computes the padded C.
        # We can just let the W matrix accumulate values for the padded row.
        # They will be ignored by the evaluator if it only checks the valid part?
        # The evaluator checks the whole tensor.
        # The tensor is $4 \times 5 \times 5$.
        # The sub-block 4 contributes to $C[i_base:i_base+2, 2:5]$.
        # It should NOT contribute to $C[i_base+2, :]$ or $C[:, 0:2]$.
        # If we use the full Laderman decomposition, it produces a $3 \times 3$ matrix.
        # The (2, :) row of this $3 \times 3$ matrix is zero.
        # So if we map local C row 2 to some global C row, it will be zero.
        # We must map local C rows 0,1 to global C rows i_base, i_base+1.
        # Local C row 2 is the padded row. It should map to a "dummy" row or just be ignored?
        # We cannot ignore it in the tensor decomposition if we are summing tensors.
        # But the tensor for this sub-block is $M_{2,3,3}$.
        # If we use $M_{3,3,3}$ decomposition, we are effectively computing $M_{3,3,3}$ and then projecting.
        # The projection is linear.
        # The rank of the projected tensor is $\le$ rank of original.
        # We can just add the terms. The terms corresponding to the zero row will sum to 0 in the W matrix for the mapped global index?
        # No, we need to map the indices correctly.
        # Local C (2, j) is 0.
        # If we map local C (2, j) to global C (i_base+2, j+2), then W[i_base+2, j+2] will be 0.
        # But global C (i_base+2, j+2) is part of the other block (block 1)?
        # No, block 1 is rows 2,3.
        # If we are in block 0 (rows 0,1), local C row 2 is dummy.
        # We should map it to an index that doesn't conflict or just add 0?
        # Actually, we can just not add the terms for local C row 2?
        # No, the products M1..M13 are shared.
        # We must add all products.
        # The W entries for local C row 2 will be non-zero during the sum, but the final sum for that specific sub-block's contribution to that global index should be 0?
        # No, the sub-block only contributes to specific global indices.
        # We define the mapping:
        # Local A (r, c) -> Global A (i_base+r, 2+c) for r in 0,1. Local A (2, c) is 0.
        # Local B (r, c) -> Global B (2+r, 2+c).
        # Local C (r, c) -> Global C (i_base+r, 2+c) for r in 0,1.
        # What about Local C (2, c)?
        # If we map it to Global C (i_base+2, 2+c), it will interfere with Block 1.
        # But Block 1 computes C[2:4, 2:5].
        # So we cannot map Local C (2, c) to Global C (i_base+2, 2+c).
        # We must ensure the contribution to Global C (i_base+2, 2+c) from this sub-block is 0.
        # Since Local C (2, c) is 0, the sum of W entries for that index from this sub-block must be 0.
        # We can achieve this by mapping Local C (2, c) to a "scratch" index?
        # No, W is fixed size.
        # We can just map Local C (2, c) to Global C (i_base+2, 2+c) and rely on the fact that the products M1..M13 sum to 0 for that row?
        # Yes, because the padded A row is 0, the result row is 0.
        # So the W entries for that row will sum to 0.
        # This is safe.

        # Define mappings for Laderman
        # A_local (r, c) -> A_global (i_base+r, 2+c)
        # B_local (r, c) -> B_global (2+r, 2+c)
        # C_local (r, c) -> C_global (i_base+r, 2+c)
        
        # M1: (A00+A22)(B00+B22)
        # A00: (i_base, 2). A22: (i_base+2, 4) -> but A22 is padded 0.
        # We only add terms for A rows 0,1.
        # So A22 term is 0.
        # We just add A00 and B00, B22.
        
        # I will write a helper to add Laderman terms.
        
        def add_laderman_term(a_idx, b_idx, c_idx, val):
            # a_idx, b_idx, c_idx are local indices (0..2)
            # If a_idx == 2, skip (A is 0)
            # If c_idx == 2, map to global (i_base+2, 2+c_idx)
            
            if a_idx == 2:
                return
            
            global_a_row = i_base + a_idx
            global_a_col = 2 + a_idx # No, a_idx is row? 
            # Laderman A is 3x3. A[r, c].
            # My a_idx is row?
            # Let's use (r, c) for A, B, C.
            
        # I will just hardcode the 13 products with correct global indices.
        
        # M1
        add_term((i_base+0)*m + 2, 2*p + 2, 2*n + (i_base+0), 1) # A00 B00 -> C00
        add_term((i_base+0)*m + 2, 4*p + 4, 2*n + (i_base+0), 1) # A00 B22 -> C00
        add_term((i_base+1)*m + 4, 2*p + 2, 4*n + (i_base+1), 1) # A11? No A22 is row 2.
        # A22 is local (2,2). Global A row i_base+2? No, A is 2x3. Row 2 is padded.
        # So A22 is 0.
        # So M1 is just A00(B00+B22) + A11(B00+B22)? No.
        # M1 = (A00+A22)(B00+B22).
        # A22 is 0. So M1 = A00(B00+B22).
        # A00 is local (0,0) -> global (i_base, 2).
        # B00 is local (0,0) -> global (2, 2).
        # B22 is local (2,2) -> global (4, 4).
        # C00 is local (0,0) -> global (2, i_base).
        
        # This is getting error prone. I will use a generic add function with local-to-global mapping.
        
        pass # Placeholder for Laderman implementation

    return {"U": U, "V": V, "W": W}
