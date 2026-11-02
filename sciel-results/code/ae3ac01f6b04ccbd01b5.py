import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if u_val != 0:
            U[u_idx][col] += u_val
        if v_val != 0:
            V[v_idx][col] += v_val
        if w_val != 0:
            W[w_idx][col] += w_val
        col += 1
        
    # 1. Add terms for 4x4x4x4 part (49 terms)
    # Strassen for 4x4x4x4
    # 7 steps, each step is a 2x2x2x2 product (rank 7)
    
    # Define the 7 Strassen steps for 4x4
    # a_coeffs: {block: coeff}
    # b_coeffs: {block: coeff}
    steps = [
        ({11: 1, 22: 1}, {11: 1, 22: 1}),
        ({21: 1, 22: 1}, {11: 1}),
        ({11: 1}, {12: 1, 22: -1}),
        ({22: 1}, {21: 1, 11: -1}),
        ({11: 1, 12: 1}, {22: 1}),
        ({21: 1, 11: -1}, {11: 1, 12: 1}),
        ({12: 1, 22: -1}, {21: 1, 22: 1}),
    ]
    
    # For each step, we have A_s and B_s which are 2x2 matrices.
    # We apply Strassen to A_s * B_s.
    # The 7 terms for 2x2x2x2 are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> c00+=1, c11+=1
    # 2. u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0) -> c10+=1
    # 3. u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0) -> c01+=1
    # 4. u=(0,0,0,1), v=(0,0,1,-1), w=(0,0,1,0) -> c10+=1
    # 5. u=(1,1,0,0), v=(0,0,0,1), w=(0,1,0,0) -> c01+=1
    # 6. u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1) -> c11+=1
    # 7. u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0) -> c00+=1
    
    # Wait, the w vectors are for the local 2x2 C block.
    # We need to map the local C indices to global C indices.
    
    # Local C indices: 00, 01, 10, 11
    # Global C indices depend on the block.
    # For step k, the result C_s is added to the global C.
    # The mapping of C_s to global C is the same for all steps?
    # No, the Strassen algorithm for 4x4 combines the 7 results P1..P7 to form C.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So for each Pk, we need to add its contribution to the global C.
    # Pk is a 2x2 matrix.
    # Pk[0,0] contributes to C11[0,0] (which is C[0,0]), C12[0,0] (which is C[2,0]), etc?
    # No, Pk is the result of the Strassen step.
    # The Strassen step computes a 2x2 matrix.
    # This 2x2 matrix is then combined to form the 4x4 C.
    
    # So for each step k, we generate 7 terms for the 2x2 product.
    # Each term has a w vector (2x2).
    # This w vector is then scaled and added to the global C according to the combination rule.
    
    # Example: Step 1 (P1).
    # P1 = (A11+A22)(B11+B22).
    # The 7 terms for P1 have w vectors w1..w7 (2x2).
    # P1 = sum_{r=1}^7 (u_r * v_r) * w_r
    # C11 += P1, C22 += P1.
    # So for each r, we add (u_r * v_r) * w_r to C11 and C22.
    # C11[0,0] is C[0,0]. C11[0,1] is C[1,0]? No.
    # C11 is the top-left 2x2 block of C.
    # C11[i,j] = C[i,j].
    # So w_r[i,j] contributes to C[i,j].
    
    # So for Step 1, for each r=1..7:
    # add_term(u_r, v_r, w_r) to C11 (i.e., C[0:2, 0:2])
    # add_term(u_r, v_r, w_r) to C22 (i.e., C[2:4, 2:4])
    
    # This seems correct.
    
    # Let's implement this.
    
    # The 7 terms for 2x2x2x2:
    # r=1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
    # r=2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
    # r=3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)
    # r=4: u=(0,0,0,1), v=(0,0,1,-1), w=(0,0,1,0)
    # r=5: u=(1,1,0,0), v=(0,0,0,1), w=(0,1,0,0)
    # r=6: u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1)
    # r=7: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0)
    
    # We need to map the local A/B indices to global A/B indices.
    # Local A indices: 00, 01, 10, 11
    # Global A indices depend on the block.
    # For a given step, A_s is formed from A blocks.
    # A_s[i,j] = sum_{block} coeff * A_block[i,j]
    # So u_r is a linear combination of A_s elements.
    # u_r = sum_{ij} u_r[i,j] A_s[i,j]
    # = sum_{ij} u_r[i,j] sum_{block} coeff_{block} A_block[i,j]
    # = sum_{block, ij} (u_r[i,j] * coeff_{block}) A_block[i,j]
    
    # So for each term r, we have a set of (global_A_idx, coeff) and (global_B_idx, coeff).
    # We then add the term to U, V, W.
    
    # This is efficient.
    
    return {"U": U, "V": V, "W": W}
