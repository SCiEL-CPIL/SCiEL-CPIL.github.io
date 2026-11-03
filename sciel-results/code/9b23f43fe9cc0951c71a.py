"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0

    # Structural decomposition:
    # Split output k into [0,4) and [4,6)
    # Part 1: M_{4,4,4} using Strassen recursion (rank 49)
    # Part 2: M_{4,4,2} using 2x2x2 Strassen blocks (rank 28)
    # Total rank: 77
    
    # We'll build the decomposition column by column in the rank dimension.
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)  
    # W: (p*n, rank) = (24, rank)
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    terms = []  # list of (a_dict, b_dict, c_dict)
    
    # Helper: add a 2x2x2 Strassen block at given offsets
    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # C[i0,k0]+=M1, C[i0,k1]+=M1, C[i1,k0]+=M1, C[i1,k1]+=M1
        terms.append((
            {(i0, j0): 1, (i1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {(i0, k0): 1, (i0, k1): 1, (i1, k0): 1, (i1, k1): 1}
        ))
        # M2 = (A[i0,j1]+A[i1,j1]) * B[j0,k0]
        # C[i0,k0]-=M2, C[i1,k1]+=M2
        terms.append((
            {(i0, j1): 1, (i1, j1): 1},
            {(j0, k0): 1},
            {(i0, k0): -1, (i1, k1): 1}
        ))
        # M3 = A[i0,j0] * (B[j1,k1]-B[j0,k0])
        # C[i0,k0]+=M3, C[i1,k0]-=M3
        terms.append((
            {(i0, j0): 1},
            {(j1, k1): 1, (j0, k0): -1},
            {(i0, k0): 1, (i1, k0): -1}
        ))
        # M4 = A[i1,j0] * (B[j1,k0]-B[j0,k0])
        # C[i1,k0]+=M4
        terms.append((
            {(i1, j0): 1},
            {(j1, k0): 1, (j0, k0): -1},
            {(i1, k0): 1}
        ))
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # C[i0,k1]-=M5, C[i1,k1]+=M5
        terms.append((
            {(i0, j0): 1, (i0, j1): 1},
            {(j1, k1): 1},
            {(i0, k1): -1, (i1, k1): 1}
        ))
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # C[i0,k0]+=M6
        terms.append((
            {(i1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {(i0, k0): 1}
        ))
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # C[i0,k1]+=M7, C[i1,k1]-=M7
        terms.append((
            {(i0, j1): 1, (i1, j1): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {(i0, k1): 1, (i1, k1): -1}
        ))
    
    # Part 1: M_{4,4,4} using Strassen recursion (rank 49)
    # Split into 2x2 blocks, apply Strassen at the block level.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # Strassen:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B11-B00)
    #   M4 = A10*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M2 + M3 + M6
    # C01 = M1 - M5 + M7
    # C10 = M1 - M3 + M4
    # C11 = M2 + M5 - M7
    # Each M_t is a 2x2x2 product, which itself has rank 7.
    # So each M_t contributes 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # For each Strassen block-level term M_t, we need to expand it into 7 rank-1 terms.
    # M_t = P * Q where P is a 2x2 matrix (linear combo of A blocks) and Q is a 2x2 matrix (linear combo of B blocks).
    # The product P*Q is a 2x2x2 tensor that we decompose using Strassen.
    
    # Block indices: A[i_g, j_g] is A[2*i_g:2*i_g+2, 2*j_g:2*j_g+2]
    #                B[j_g, k_g] is B[2*j_g:2*j_g+2, 2*k_g:2*k_g+2]
    #                C[i_g, k_g] is C[2*i_g:2*i_g+2, 2*k_g:2*k_g+2]
    
    # For the 4x4x4 part (k in [0,4)):
    # Block-level Strassen terms, each expanded into 7 rank-1 terms via 2x2x2 Strassen.
    
    # Define the 7 block-level Strassen combinations for 4x4x4:
    # Each is: (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    # a_block_coeffs: dict (i_g, j_g) -> coeff
    # b_block_coeffs: dict (j_g, k_g) -> coeff
    # c_block_coeffs: dict (i_g, k_g) -> coeff
    
    block_strassen_4x4x4 = [
        # M1 = (A00+A11)*(B00+B11), contributes +1 to all C blocks
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        ),
        # M2 = (A10+A11)*B00, C00-=M2, C11+=M2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): -1, (1,1): 1}
        ),
        # M3 = A00*(B11-B00), C00+=M3, C10-=M3
        (
            {(0,0): 1},
            {(1,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): -1}
        ),
        # M4 = A10*(B10-B00), C10+=M4
        (
            {(1,0): 1},
            {(1,0): 1, (0,0): -1},
            {(1,0): 1}
        ),
        # M5 = (A00+A01)*B11, C01-=M5, C11+=M5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,1): -1, (1,1): 1}
        ),
        # M6 = (A10-A00)*(B00+B01), C00+=M6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(0,0): 1}
        ),
        # M7 = (A01-A11)*(B10+B11), C01+=M7, C11-=M7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,1): 1, (1,1): -1}
        ),
    ]
    
    # For each block-level term, expand into 7 rank-1 terms.
    # A block-level term has:
    #   A_part = sum_{(i_g,j_g)} coeff_a * A[i_g, j_g]  (a 2x2 matrix)
    #   B_part = sum_{(j_g,k_g)} coeff_b * B[j_g, k_g]  (a 2x2 matrix)
    #   C_part = sum_{(i_g,k_g)} coeff_c * C[i_g, k_g]  (a 2x2 matrix)
    # And we need: A_part * B_part contributes coeff_c to C[i_g, k_g] for each (i_g, k_g).
    
    # Actually, the block-level term says:
    #   (sum_{(i_g,j_g)} coeff_a[(i_g,j_g)] * A[i_g,j_g]) * (sum_{(j_g,k_g)} coeff_b[(j_g,k_g)] * B[j_g,k_g])
    #   = sum_{(i_g,k_g)} coeff_c[(i_g,k_g)] * C[i_g,k_g]
    
    # Each A[i_g,j_g] is a 2x2 matrix with entries A[2*i_g+a, 2*j_g+b] for a,b in {0,1}.
    # Each B[j_g,k_g] is a 2x2 matrix with entries B[2*j_g+a, 2*k_g+b].
    # Each C[i_g,k_g] is a 2x2 matrix with entries C[2*i_g+a, 2*k_g+b].
    
    # So the full expression is:
    #   sum_{(i_g,j_g),(j_g',k_g)} coeff_a[(i_g,j_g)] * coeff_b[(j_g',k_g)] * A[i_g,j_g] * B[j_g',k_g]
    #   = sum_{(i_g,k_g)} coeff_c[(i_g,k_g)] * C[i_g,k_g]
    
    # Note: j_g and j_g' are the same index (the middle dimension), so we sum over j_g:
    #   sum_{j_g} [sum_{i_g} coeff_a[(i_g,j_g)] * A[i_g,j_g]] * [sum_{k_g} coeff_b[(j_g,k_g)] * B[j_g,k_g]]
    #   Wait, that's not quite right. Let me re-derive.
    
    # The block-level Strassen says:
    #   C[i_g, k_g] = sum_{j_g} A[i_g, j_g] * B[j_g, k_g]  (for the 4x4x4 case)
    # And the Strassen algorithm computes this using 7 products of block matrices.
    # Each product is: P_t * Q_t where P_t is a 2x2 matrix (linear combo of A blocks) and Q_t is a 2x2 matrix (linear combo of B blocks).
    # The result P_t * Q_t is a 2x2 matrix that contributes to C blocks.
    
    # For a specific block-level term t:
    #   P_t = sum_{(i_g,j_g)} coeff_a_t[(i_g,j_g)] * A[i_g, j_g]
    #   Q_t = sum_{(j_g,k_g)} coeff_b_t[(j_g,k_g)] * B[j_g, k_g]
    #   P_t * Q_t = sum_{(i_g,k_g)} coeff_c_t[(i_g,k_g)] * C[i_g, k_g]
    
    # Now, P_t is a 2x2 matrix whose (a,b) entry is:
    #   sum_{(i_g,j_g)} coeff_a_t[(i_g,j_g)] * A[2*i_g+a, 2*j_g+b]
    # Q_t is a 2x2 matrix whose (c,d) entry is:
    #   sum_{(j_g,k_g)} coeff_b_t[(j_g,k_g)] * B[2*j_g+c, 2*k_g+d]
    # (P_t * Q_t) is a 2x2 matrix whose (a,d) entry is:
    #   sum_c (P_t)_{a,c} * (Q_t)_{c,d}
    #   = sum_c [sum_{(i_g,j_g)} coeff_a_t[(i_g,j_g)] * A[2*i_g+a, 2*j_g+c]] * [sum_{(j_g',k_g)} coeff_b_t[(j_g',k_g)] * B[2*j_g'+c, 2*k_g+d]]
    
    # And this should equal:
    #   sum_{(i_g,k_g)} coeff_c_t[(i_g,k_g)] * C[2*i_g+a, 2*k_g+d]
    #   = sum_{(i_g,k_g)} coeff_c_t[(i_g,k_g)] * sum_{j_g} A[2*i_g+a, 2*j_g] * B[2*j_g, 2*k_g+d]
    
    # This is getting very complex. Let me take a different approach.
    
    # Instead of trying to expand the block-level Strassen into rank-1 terms analytically,
    # let me just use the 2x2x2 Strassen for each of the 4x4x4 sub-problems.
    
    # Actually, the simplest correct approach: 
    # For the 4x4x4 part, use the standard Strassen recursion which gives rank 49.
    # The 4x4x4 Strassen: 7 block-level multiplications, each of which is a 2x2x2 multiplication (rank 7).
    # Total: 7 * 7 = 49.
    
    # To implement this, for each of the 7 block-level Strassen terms, I need to:
    # 1. Identify which A blocks and B blocks are combined (with coefficients)
    # 2. The product of these combined blocks is a 2x2x2 tensor
    # 3. Decompose that 2x2x2 tensor using Strassen (7 rank-1 terms)
    
    # But the "combined blocks" are linear combinations of 2x2 matrices, so the product
    # is not simply a 2x2x2 tensor of individual entries.
    
    # Let me think about this differently. The 4x4x4 matrix multiplication tensor T[i,j,k] for i,j,k in [0,4)
    # can be decomposed as follows using Strassen:
    # 
    # Split i into i_g (block) and a (within block), similarly j = j_g*2 + b, k = k_g*2 + d.
    # 
    # T[i,j,k] = 1 for all i,j,k in [0,4).
    # 
    # Strassen at the block level:
    #   C[i_g, k_g] = sum_{j_g} A[i_g, j_g] * B[j_g, k_g]
    # 
    # Where A[i_g, j_g] is the 2x2 submatrix A[2*i_g:2*i_g+2, 2*j_g:2*j_g+2], etc.
    # 
    # The Strassen algorithm computes C[i_g, k_g] using 7 products of 2x2 matrices.
    # Each product M_t = P_t * Q_t where P_t and Q_t are 2x2 matrices (linear combos of A and B blocks).
    # 
    # M_t is a 2x2 matrix, and M_t = P_t * Q_t. The entries of M_t are:
    #   M_t[a,d] = sum_c P_t[a,c] * Q_t[c,d]
    # 
    # P_t[a,c] = sum_{(i_g,j_g)} coeff_a[(i_g,j_g)] * A[2*i_g+a, 2*j_g+c]
    # Q_t[c,d] = sum_{(j_g,k_g)} coeff_b[(j_g,k_g)] * B[2*j_g+c, 2*k_g+d]
    # 
    # So M_t[a,d] = sum_c sum_{(i_g,j_g)} sum_{(j_g',k_g)} coeff_a[(i_g,j_g)] * coeff_b[(j_g',k_g)] 
    #                * A[2*i_g+a, 2*j_g+c] * B[2*j_g'+c, 2*k_g+d]
    # 
    # And M_t[a,d] contributes to C[2*i_g+a, 2*k_g+d] with coefficient coeff_c[(i_g,k_g)] for each (i_g,k_g).
    # 
    # So the full contribution of block-level term t to the tensor is:
    #   sum_{a,d} M_t[a,d] * [contribution to C entries]
    # 
    # This is a rank-49 tensor (7 block-level terms × 7 within-block Strassen terms).
    
    # This is getting too complex to implement correctly by hand. Let me use a simpler approach:
    # just use 2x2x2 Strassen blocks for the entire 4x4x6 tensor, but with a smarter partitioning.
    
    # Actually, let me just go with the original 84-rank approach but verify it's correct,
    # and then try to find a structural improvement.
    
    # Let me restart with a cleaner approach.
    
    # APPROACH: Use 2x2x2 Strassen for all blocks, giving rank 84.
    # Then try to reduce by finding linear dependencies.
    
    # Actually, let me just implement the 77-rank approach correctly.
    # The key insight: for the 4x4x4 part, I can use the fact that it's a "product" of
    # a 4x4x4 tensor, and the Strassen algorithm gives rank 49.
    # For the 4x4x2 part, I use 2x2x2 Strassen blocks, giving rank 28.
    
    # Let me implement the 4x4x4 Strassen (rank 49) by directly constructing the 49 rank-1 terms.
    
    # The 4x4x4 Strassen algorithm:
    # Block-level: 7 terms M1..M7, each is a product of two 2x2 matrices.
    # Each 2x2 matrix product has rank 7 (Strassen).
    # Total: 49 rank-1 terms.
    
    # For block-level term t with:
    #   P_t = sum_{(i_g,j_g)} ca[(i_g,j_g)] * A_block(i_g, j_g)
    #   Q_t = sum_{(j_g,k_g)} cb[(j_g,k_g)] * B_block(j_g, k_g)
    #   M_t = P_t * Q_t
    #   C_block(i_g, k_g) gets contribution cc[(i_g,k_g)] * M_t
    # 
    # M_t is a 2x2 matrix. Using Strassen on M_t = P_t * Q_t:
    #   M_t[0,0] = S1 - S2 + S3 + S6
    #   M_t[0,1] = S1 - S5 + S7
    #   M_t[1,0] = S1 - S3 + S4
    #   M_t[1,1] = S2 + S5 - S7
    # where S1..S7 are the 7 Strassen products of P_t and Q_t.
    # 
    # Each S_s is a scalar: S_s = (linear combo of P_t entries) * (linear combo of Q_t entries)
    # 
    # P_t entries: P_t[a,c] = sum_{(i_g,j_g)} ca[(i_g,j_g)] * A[2*i_g+a, 2*j_g+c]
    # Q_t entries: Q_t[c,d] = sum_{(j_g,k_g)} cb[(j_g,k_g)] * B[2*j_g+c, 2*k_g+d]
    # 
    # So S_s = (sum_{a,c} alpha_s[a,c] * P_t[a,c]) * (sum_{c,d} beta_s[c,d] * Q_t[c,d])
    #        = (sum_{a,c} sum_{(i_g,j_g)} alpha_s[a,c] * ca[(i_g,j_g)] * A[2*i_g+a, 2*j_g+c])
    #          * (sum_{c,d} sum_{(j_g,k_g)} beta_s[c,d] * cb[(j_g,k_g)] * B[2*j_g+c, 2*k_g+d])
    # 
    # And S_s contributes to M_t[a',d'] with coefficient gamma_s[a',d'] (from the Strassen reconstruction).
    # M_t[a',d'] contributes to C[2*i_g+a', 2*k_g+d'] with coefficient cc[(i_g,k_g)].
    # 
    # So the full contribution of (t, s) to the tensor is:
    #   sum_{a,c} sum_{(i_g,j_g)} alpha_s[a,c] * ca[(i_g,j_g)] * A[2*i_g+a, 2*j_g+c]
    #   times
    #   sum_{c,d} sum_{(j_g,k_g)} beta_s[c,d] * cb[(j_g,k_g)] * B[2*j_g+c, 2*k_g+d]
    #   times
    #   sum_{a',d',i_g,k_g} gamma_s[a',d'] * cc[(i_g,k_g)] * C[2*i_g+a', 2*k_g+d']
    # 
    # This gives a rank-1 tensor for each (t,s) pair.
    
    # Strassen 2x2x2 coefficients:
    # S1 = (P[0,0]+P[1,1]) * (Q[0,0]+Q[1,1])
    # S2 = (P[0,1]+P[1,1]) * Q[0,0]
    # S3 = P[0,0] * (Q[1,1]-Q[0,0])
    # S4 = P[1,0] * (Q[1,0]-Q[0,0])
    # S5 = (P[0,0]+P[0,1]) * Q[1,1]
    # S6 = (P[1,0]-P[0,0]) * (Q[0,0]+Q[0,1])
    # S7 = (P[0,1]-P[1,1]) * (Q[1,0]+Q[1,1])
    # 
    # M[0,0] = S1 - S2 + S3 + S6
    # M[0,1] = S1 - S5 + S7
    # M[1,0] = S1 - S3 + S4
    # M[1,1] = S2 + S5 - S7
    
    # alpha_s[a,c]: coefficient of P[a,c] in S_s
    # beta_s[c,d]: coefficient of Q[c,d] in S_s
    # gamma_s[a',d']: coefficient of S_s in M[a',d']
    
    strassen_2x2x2_coeffs = [
        # S1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # S2
        ({(0,1):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,1):1}),
        # S3
        ({(0,0):1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,0):-1}),
        # S4
        ({(1,0):1}, {(1,0):1, (0,0):-1}, {(1,0):1}),
        # S5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}),
        # S6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}),
        # S7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):-1}),
    ]
    
    # Now build all terms for the 4x4x4 part:
    for ca, cb, cc in block_strassen_4x4x4:
        for alpha, beta, gamma in strassen_2x2x2_coeffs:
            # Build A coefficients: U[i*m+j, col]
            a_coeffs = {}
            for (a, c), val_a in alpha.items():
                for (i_g, j_g), val_ca in ca.items():
                    i = 2 * i_g + a
                    j = 2 * j_g + c
                    a_coeffs[(i, j)] = a_coeffs.get((i, j), 0) + val_a * val_ca
            
            # Build B coefficients: V[j*p+k, col]
            b_coeffs = {}
            for (c, d), val_b in beta.items():
                for (j_g, k_g), val_cb in cb.items():
                    j = 2 * j_g + c
                    k = 2 * k_g + d
                    b_coeffs[(j, k)] = b_coeffs.get((j, k), 0) + val_b * val_cb
            
            # Build C coefficients: W[k*n+i, col]
            c_coeffs = {}
            for (a, d), val_g in gamma.items():
                for (i_g, k_g), val_cc in cc.items():
                    i = 2 * i_g + a
                    k = 2 * k_g + d
                    c_coeffs[(i, k)] = c_coeffs.get((i, k), 0) + val_g * val_cc
            
            terms.append((a_coeffs, b_coeffs, c_coeffs))
    
    # Part 2: M_{4,4,2} for k in [4,6) using 2x2x2 Strassen blocks (rank 28)
    # A is 4x4, B is 4x2 (columns 4,5), C is 4x2 (columns 4,5)
    # Split i into [0,2) and [2,4), j into [0,2) and [2,4), k into [4,6) (single group of 2)
    # For each (i_g, j_g): 2x2x2 Strassen block at (i_g*2, j_g*2, 4)
    for i_g in range(2):
        for j_g in range(2):
            add_strassen_2x2x2(2 * i_g, 2 * j_g, 4)
    
    # Now build the matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col, (a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
        for (i, j), coeff in a_coeffs.items():
            U[i * m + j, col] = coeff
        for (j, k), coeff in b_coeffs.items():
            V[j * p + k, col] = coeff
        for (i, k), coeff in c_coeffs.items():
            W[k * n + i, col] = coeff
    
    # Verify
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
