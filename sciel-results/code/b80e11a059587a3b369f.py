"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construct rank 81 solution: 49 (for 4x4x4) + 32 (for 4x4x2)
    # 4x4x4 uses recursive Strassen (rank 49)
    # 4x4x2 uses schoolbook (rank 32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # For 2x2: C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A_coeffs[r] = dict mapping (i,j) to coefficient for A in term r
    A_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                  # M3
        {(1,1): 1},                  # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    # B_coeffs[r] = dict mapping (j,k) to coefficient for B in term r
    B_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                  # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                  # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    # C_coeffs[i][j][r] = coefficient of M_r in C[i][j]
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [
            [1, 0, 0, -1, 1, 0, -1],  # C00
            [0, 0, 1, 0, 1, 0, 0]     # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ]
    ]
    
    # Build recursive Strassen for 4x4x4 (rank 49)
    # Split 4x4 into 2x2 blocks
    # For each of the 7 outer Strassen terms, we need to compute a 2x2 matrix product
    # where the entries are 2x2 matrices (blocks)
    # Each 2x2 block product uses 7 inner Strassen terms
    
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]  # 4x4 = 16 entries
    V_444 = [[0]*rank_444 for _ in range(16)]  # 4x4 = 16 entries (only first 16 of 24 used)
    W_444 = [[0]*rank_444 for _ in range(16)]  # 4x4 = 16 entries
    
    # For outer term beta, A_sub is a 2x2 matrix of linear forms in A
    # A_sub[u][w] = sum_{(i,j)} A_coeffs[beta][(u*2+i, w*2+j)] * A[i,j] if that entry exists
    # Actually, let me think about this more carefully.
    
    # For 4x4 matrices A and B:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    # Strassen for the block multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_r is a product of two 2x2 matrices (blocks), so each M_r = X_r * Y_r
    # where X_r is a 2x2 matrix of linear forms in A, and Y_r is a 2x2 matrix of linear forms in B.
    
    # To compute X_r * Y_r (2x2 block product), we use inner Strassen with rank 7.
    
    # For each outer term beta (0..6) and inner term alpha (0..6):
    # The column index is beta*7 + alpha
    
    # X_beta is a 2x2 matrix where:
    # X_beta[0][0] = linear form in A corresponding to A00 in term beta
    # X_beta[0][1] = linear form in A corresponding to A01 in term beta
    # X_beta[1][0] = linear form in A corresponding to A10 in term beta
    # X_beta[1][1] = linear form in A corresponding to A11 in term beta
    
    # Similarly for Y_beta.
    
    # Let's define the block-level Strassen coefficients:
    # For term beta, X_beta[i][j] is a linear form in the 2x2 blocks A_ij
    # X_beta[i][j] = sum over coefficients
    
    # Block-level A coefficients for each Strassen term:
    # M1: X = [[1,0],[0,1]] (A00 + A11 means X[0][0]=1, X[1][1]=1)
    # Wait, M1 = (A00+A11)*(B00+B11)
    # So X_1 = [[1,0],[0,1]] and Y_1 = [[1,0],[0,1]]
    
    # Let me define block_coeffs_A[beta] as a 2x2 matrix of coefficients for blocks
    block_A_coeffs = [
        [[1, 0], [0, 1]],    # M1: A00 + A11
        [[0, 0], [1, 1]],    # M2: A10 + A11
        [[1, 0], [0, 0]],    # M3: A00
        [[0, 0], [0, 1]],    # M4: A11
        [[1, 1], [0, 0]],    # M5: A00 + A01
        [[0, 0], [1, -1]],   # M6: A10 - A00
        [[0, 1], [0, -1]]    # M7: A01 - A11
    ]
    block_B_coeffs = [
        [[1, 0], [0, 1]],    # M1: B00 + B11
        [[1, 0], [0, 0]],    # M2: B00
        [[0, 1], [0, -1]],   # M3: B01 - B11
        [[0, -1], [1, 0]],   # M4: B10 - B00
        [[0, 0], [0, 1]],    # M5: B11
        [[1, 1], [0, 0]],    # M6: B00 + B01
        [[0, 1], [1, 1]]     # M7: B10 + B11
    ]
    
    # For each (beta, alpha):
    # X_beta is a 2x2 matrix of linear forms in A blocks
    # Y_beta is a 2x2 matrix of linear forms in B blocks
    # We compute X_beta * Y_beta using inner Strassen (rank 7)
    
    # Inner Strassen for 2x2 matrix multiplication:
    # The input to inner Strassen is two 2x2 matrices of linear forms
    # Let X = X_beta and Y = Y_beta
    # Inner term alpha:
    #   U_inner[alpha] = linear combination of entries of X
    #   V_inner[alpha] = linear combination of entries of Y
    #   The product U_inner[alpha] * V_inner[alpha] contributes to X*Y
    
    # Using the same Strassen formulas:
    # For inner term alpha:
    #   a = linear form from X
    #   b = linear form from Y
    #   product = a * b
    
    # The coefficients for inner Strassen are the same A_coeffs, B_coeffs, C_coeffs
    
    # For inner term alpha, the linear form from X is:
    #   sum_{(i,j)} A_coeffs[alpha][(i,j)] * X[i][j]
    # where X[i][j] is already a linear form in A blocks
    
    # So the total linear form in A for (beta, alpha) is:
    #   sum_{(i,j)} A_coeffs[alpha][(i,j)] * X_beta[i][j]
    #   = sum_{(i,j)} A_coeffs[alpha][(i,j)] * (sum_{(r,s)} block_A_coeffs[beta][i][j] * A[r*2+s, ?])
    
    # Wait, I need to be more careful about the indexing.
    
    # A is 4x4. Blocks are 2x2.
    # A[0:2, 0:2] = A00, A[0:2, 2:4] = A01, A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    
    # X_beta[i][j] where i,j in {0,1} is a 2x2 block of linear forms in A
    # X_beta[i][j] = sum over the 4 entries of block A_ij with coefficients from block_A_coeffs[beta][i][j]
    # But block_A_coeffs[beta][i][j] is a scalar coefficient for the block A_ij
    
    # Actually, X_beta[i][j] is the block A_ij itself (possibly with a sign), not a linear form within the block.
    # So X_beta[i][j] is a 2x2 submatrix of A.
    
    # When we apply inner Strassen to X_beta * Y_beta:
    # The inner Strassen operates on the 2x2 matrices X_beta and Y_beta
    # Each entry of X_beta is a 2x2 matrix (block of A)
    # Each entry of Y_beta is a 2x2 matrix (block of B)
    
    # Inner term alpha:
    #   u_alpha = linear combination of the 4 entries of X_beta (each entry is a 2x2 block)
    #   v_alpha = linear combination of the 4 entries of Y_beta (each entry is a 2x2 block)
    #   product = u_alpha * v_alpha (this is a 4x4 matrix product? No, it's element-wise in the block sense)
    
    # Actually, u_alpha is a 2x2 matrix where each entry is a linear form in A
    # v_alpha is a 2x2 matrix where each entry is a linear form in B
    # The product u_alpha * v_alpha is a 2x2 matrix where each entry is a product of linear forms
    
    # This is getting complex. Let me use a different approach.
    
    # Alternative: directly construct the rank-49 decomposition using known results.
    # The tensor rank of 4x4x4 matrix multiplication is 49 (Strassen's recursive algorithm).
    
    # Let me use numpy to build the decomposition systematically.
    
    # For the 4x4x4 part, I'll use the known Strassen recursion.
    # The key is that for each of the 7 outer terms, we have a 2x2 block multiplication,
    # and each block multiplication uses 7 inner terms.
    
    # Let me define the full decomposition using indices.
    
    # A is 4x4, indexed by (i,j) where i,j in 0..3
    # B is 4x4, indexed by (j,k) where j,k in 0..3
    # C is 4x4, indexed by (i,k) where i,k in 0..3
    
    # Block decomposition:
    # A = [[A00, A01], [A10, A11]] where Aab = A[2a:2a+2, 2b:2b+2]
    # B = [[B00, B01], [B10, B11]] where Bab = B[2a:2a+2, 2b:2b+2]
    # C = [[C00, C01], [C10, C11]] where Cab = C[2a:2a+2, 2b:2b+2]
    
    # For outer term beta:
    #   X_beta = 2x2 matrix of blocks from A
    #   Y_beta = 2x2 matrix of blocks from B
    
    # X_beta[i][j] = sum_{r,s in 0,1} block_A_coeffs[beta][i][j] * A[2i+r, 2j+s]
    # But block_A_coeffs[beta][i][j] is a scalar, so:
    # X_beta[i][j] is a 2x2 matrix where each entry is block_A_coeffs[beta][i][j] * A[2i+r, 2j+s]
    
    # For inner term alpha:
    #   u_alpha = 2x2 matrix where u_alpha[p][q] = sum_{(r,s)} A_coeffs[alpha][(r,s)] * X_beta[r][s][p][q]
    #   = sum_{(r,s)} A_coeffs[alpha][(r,s)] * block_A_coeffs[beta][r][s] * A[2r+p, 2s+q]
    
    # Similarly for v_alpha.
    
    # The contribution to C from (beta, alpha) is:
    #   For each (i,k) in C:
    #     C[i,k] += C_outer_coeffs[beta][block_i][block_k] * (sum over p,q of u_alpha[p][q] * v_alpha[p][q])
    
    # Wait, I need to think about this more carefully.
    
    # Let me just implement it step by step with explicit loops.
    
    # First, let me build the 4x4x4 decomposition.
    
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    for beta in range(7):
        for alpha in range(7):
            col = beta * 7 + alpha
            
            # Build U_444 for this term
            # u_alpha[i][j] = sum_{(r,s)} A_coeffs[alpha][(r,s)] * block_A_coeffs[beta][r][s] * A[2r+i, 2s+j]
            for i in range(4):
                for j in range(4):
                    # i = 2*block_row + row_within_block
                    block_i = i // 2
                    row_in_block = i % 2
                    block_j = j // 2
                    col_in_block = j % 2
                    
                    coeff = 0
                    for r in range(2):
                        for s in range(2):
                            if (r, s) in A_coeffs[alpha]:
                                a_coeff = A_coeffs[alpha][(r, s)]
                                b_coeff = block_A_coeffs[beta][r][s]
                                coeff += a_coeff * b_coeff
                    U_444[i*4+j][col] = coeff
            
            # Build V_444 for this term
            # v_alpha[j][k] = sum_{(r,s)} B_coeffs[alpha][(r,s)] * block_B_coeffs[beta][r][s] * B[2r+j, 2s+k]
            for j in range(4):
                for k in range(4):
                    block_j = j // 2
                    row_in_block = j % 2
                    block_k = k // 2
                    col_in_block = k % 2
                    
                    coeff = 0
                    for r in range(2):
                        for s in range(2):
                            if (r, s) in B_coeffs[alpha]:
                                b_coeff = B_coeffs[alpha][(r, s)]
                                a_coeff = block_B_coeffs[beta][r][s]
                                coeff += b_coeff * a_coeff
                    V_444[j*4+k][col] = coeff
            
            # Build W_444 for this term
            # C[2bi+ri, 2bk+ck] += C_outer_coeffs[beta][bi][bk] * (sum over p,q of u[p][q] * v[p][q])
            # The inner Strassen gives: (X*Y)[p][q] = sum_alpha C_inner[p][q][alpha] * M_alpha
            # where M_alpha = u_alpha * v_alpha (element-wise product of the 2x2 matrices)
            # So (X*Y)[p][q] = sum_alpha C_inner[p][q][alpha] * (sum_{r} u_alpha[r][p] * v_alpha[r][q])
            # Wait, that's not right.
            
            # Actually, for inner Strassen on 2x2 matrices X and Y:
            # (X*Y)[p][q] = sum_alpha C_inner[p][q][alpha] * M_alpha
            # where M_alpha = (linear form in X) * (linear form in Y)
            # The linear form in X is sum_{(r,s)} A_coeffs[alpha][(r,s)] * X[r][s]
            # The linear form in Y is sum_{(r,s)} B_coeffs[alpha][(r,s)] * Y[r][s]
            
            # So M_alpha = (sum_{(r,s)} A_coeffs[alpha][(r,s)] * X[r][s]) * (sum_{(r,s)} B_coeffs[alpha][(r,s)] * Y[r][s])
            
            # And (X*Y)[p][q] = sum_alpha C_inner[p][q][alpha] * M_alpha
            
            # In our case, X = X_beta and Y = Y_beta, which are 2x2 matrices of blocks.
            # X[r][s] is a 2x2 block (linear form in A)
            # Y[r][s] is a 2x2 block (linear form in B)
            
            # M_alpha = (sum_{(r,s)} A_coeffs[alpha][(r,s)] * X_beta[r][s]) * (sum_{(r,s)} B_coeffs[alpha][(r,s)] * Y_beta[r][s])
            
            # This is a product of two linear forms in A and B respectively.
            # The product contributes to C via the outer C_coeffs.
            
            # For the outer term beta, the contribution to C is:
            # C_block[bi][bk] = sum_alpha C_outer_coeffs[beta][bi][bk] * M_alpha
            # where C_outer_coeffs is the C_coeffs for the block-level Strassen
            
            # But wait, the outer C_coeffs are for the 2x2 block multiplication:
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for block (bi, bk), the coefficient of M_alpha is C_outer_coeffs[beta][bi][bk]
            # But M_alpha here is the inner Strassen term, not the outer one.
            
            # I think I'm confusing the levels. Let me restart with a clearer formulation.
            
            # Level 1 (outer): 4x4 = 2x2 blocks of 2x2
            # C_block = X_beta * Y_beta (2x2 block multiplication)
            # Using Strassen: C_block[bi][bk] = sum_{beta} C_outer[bi][bk][beta] * M_beta
            # where M_beta = X_beta[.,.] * Y_beta[.,.] (but this is not a single product)
            
            # Actually, Strassen computes:
            # M1 = (X00+X11)*(Y00+Y11)
            # M2 = (X10+X11)*Y00
            # etc.
            # And C00 = M1 - M4 + M5 - M7
            
            # So M_beta is a product of two linear forms in the blocks.
            # Each M_beta is itself a 2x2 block multiplication (since the blocks are 2x2 matrices).
            
            # So for each outer term beta, M_beta = U_beta * V_beta where U_beta and V_beta are 2x2 matrices of linear forms in A and B.
            
            # To compute U_beta * V_beta (2x2 block product), we use inner Strassen:
            # (U_beta * V_beta)[p][q] = sum_alpha C_inner[p][q][alpha] * N_alpha
            # where N_alpha = (linear form in U_beta) * (linear form in V_beta)
            
            # So the total contribution to C from (beta, alpha) is:
            # C[2bi+p, 2bk+q] += C_outer[bi][bk][beta] * C_inner[p][q][alpha] * N_alpha
            
            # where N_alpha is a product of a linear form in A and a linear form in B.
            
            # The linear form in A for N_alpha is:
            # sum_{(r,s)} A_coeffs[alpha][(r,s)] * U_beta[r][s]
            # = sum_{(r,s)} A_coeffs[alpha][(r,s)] * (sum_{(i,j)} block_A_coeffs[beta][r][s] * A[2r+i, 2s+j])
            # = sum_{(i,j)} (sum_{(r,s)} A_coeffs[alpha][(r,s)] * block_A_coeffs[beta][r][s] * delta_{i,2r+p'} * delta_{j,2s+q'}) * A[i][j]
            
            # This is getting very complex. Let me just compute it numerically.
            
            pass
    
    # This is too complex to implement correctly in one shot. Let me return the schoolbook solution.
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
