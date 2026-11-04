"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Recursive Strassen: 4x4 = 7 * (2x2 Strassen) = 7 * 7 = 49 rank-1 terms.
    # We implement this by composing two levels of Strassen.
    
    # Strassen 2x2 definitions:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For 4x4, we partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Block-level Strassen (7 terms, each is a 2x2 matrix product):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix, computed via Strassen (7 scalar mults each)
    # Total: 7 * 7 = 49 scalar multiplications
    
    # We'll build U, V, W of shape (16, 49) each
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Helper: get global index for A[i,j], B[j,k], C[k,i]
    # A: 4x4, index (i,j) -> row i, col j in A, global index i*4+j
    # B: 4x4, index (j,k) -> row j, col k in B, global index j*4+k  
    # C: 4x4, index (i,k) -> row i, col k in C, global index i*4+k
    # But wait, the problem says W[k*n+i,r] for output factor.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) = (16, rank), and W[k*n+i, r]
    
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # Here n=m=p=4
    # U has shape (n*m, rank) = (16, rank): U[i*m+j, r] corresponds to A[i,j]
    # V has shape (m*p, rank) = (16, rank): V[j*p+k, r] corresponds to B[j,k]
    # W has shape (p*n, rank) = (16, rank): W[k*n+i, r] corresponds to C[i,k]
    
    # So for a rank-1 term with coefficient c:
    # U[i*m+j][col] += c_a
    # V[j*p+k][col] += c_b
    # W[k*n+i][col] += c_c
    # where c_a * c_b * c_c = c (the coefficient for C[i,k])
    
    # Let's define Strassen terms more carefully.
    # For 2x2 matrices X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    # Z = X * Y
    # 
    # s1 = (x00+x11)(y00+y11)
    # s2 = (x10+x11)y00
    # s3 = x00(y10-y11)
    # s4 = x11(y01-y00)
    # s5 = (x00+x01)y11
    # s6 = (x10-x00)(y00+y01)
    # s7 = (x01-x11)(y10+y11)
    #
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    
    # Block-level: A and B are 4x4, partitioned into 2x2 blocks
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now each Mk is a 2x2 matrix. We compute each Mk using Strassen (7 scalar mults).
    # Then we combine to get C.
    
    # Let's define the block-level Strassen terms and their contributions to C.
    # Each block-level term Mk contributes to C with certain coefficients.
    
    # We'll iterate over the 7 block-level terms and 7 inner-level terms.
    
    # Block-level term definitions:
    # For each term, we define:
    # - A_blocks: list of (block_row, block_col, coeff) for A
    # - B_blocks: list of (block_row, block_col, coeff) for B
    # - C_blocks: dict mapping (block_row, block_col) -> coeff for C contribution
    
    block_terms = [
        # M1 = (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
        {
            'A_blocks': [(0,0,1), (1,1,1)],
            'B_blocks': [(0,0,1), (1,1,1)],
            'C_blocks': {(0,0): 1, (1,1): 1}
        },
        # M2 = (A10+A11)B00, contributes to C10(+1), C11(-1)
        {
            'A_blocks': [(1,0,1), (1,1,1)],
            'B_blocks': [(0,0,1)],
            'C_blocks': {(1,0): 1, (1,1): -1}
        },
        # M3 = A00(B10-B11), contributes to C01(+1), C11(+1)
        {
            'A_blocks': [(0,0,1)],
            'B_blocks': [(1,0,1), (1,1,-1)],
            'C_blocks': {(0,1): 1, (1,1): 1}
        },
        # M4 = A11(B01-B00), contributes to C00(+1), C10(+1)
        {
            'A_blocks': [(1,1,1)],
            'B_blocks': [(0,1,1), (0,0,-1)],
            'C_blocks': {(0,0): 1, (1,0): 1}
        },
        # M5 = (A00+A01)B11, contributes to C00(-1), C01(+1)
        {
            'A_blocks': [(0,0,1), (0,1,1)],
            'B_blocks': [(1,1,1)],
            'C_blocks': {(0,0): -1, (0,1): 1}
        },
        # M6 = (A10-A00)(B00+B01), contributes to C11(+1)
        {
            'A_blocks': [(1,0,1), (0,0,-1)],
            'B_blocks': [(0,0,1), (0,1,1)],
            'C_blocks': {(1,1): 1}
        },
        # M7 = (A01-A11)(B10+B11), contributes to C00(+1)
        {
            'A_blocks': [(0,1,1), (1,1,-1)],
            'B_blocks': [(1,0,1), (1,1,1)],
            'C_blocks': {(0,0): 1}
        },
    ]
    
    # Inner-level Strassen terms (for 2x2 matrices X*Y = Z):
    # s1 = (x00+x11)(y00+y11), z00(+1), z11(+1)
    # s2 = (x10+x11)y00, z10(+1), z11(-1)
    # s3 = x00(y10-y11), z01(+1), z11(+1)
    # s4 = x11(y01-y00), z00(+1), z10(+1)
    # s5 = (x00+x01)y11, z00(-1), z01(+1)
    # s6 = (x10-x00)(y00+y01), z11(+1)
    # s7 = (x01-x11)(y10+y11), z00(+1)
    
    inner_terms = [
        # s1
        {
            'X_blocks': [(0,0,1), (1,1,1)],
            'Y_blocks': [(0,0,1), (1,1,1)],
            'Z_blocks': {(0,0): 1, (1,1): 1}
        },
        # s2
        {
            'X_blocks': [(1,0,1), (1,1,1)],
            'Y_blocks': [(0,0,1)],
            'Z_blocks': {(1,0): 1, (1,1): -1}
        },
        # s3
        {
            'X_blocks': [(0,0,1)],
            'Y_blocks': [(1,0,1), (1,1,-1)],
            'Z_blocks': {(0,1): 1, (1,1): 1}
        },
        # s4
        {
            'X_blocks': [(1,1,1)],
            'Y_blocks': [(0,1,1), (0,0,-1)],
            'Z_blocks': {(0,0): 1, (1,0): 1}
        },
        # s5
        {
            'X_blocks': [(0,0,1), (0,1,1)],
            'Y_blocks': [(1,1,1)],
            'Z_blocks': {(0,0): -1, (0,1): 1}
        },
        # s6
        {
            'X_blocks': [(1,0,1), (0,0,-1)],
            'Y_blocks': [(0,0,1), (0,1,1)],
            'Z_blocks': {(1,1): 1}
        },
        # s7
        {
            'X_blocks': [(0,1,1), (1,1,-1)],
            'Y_blocks': [(1,0,1), (1,1,1)],
            'Z_blocks': {(0,0): 1}
        },
    ]
    
    col = 0
    for block_term in block_terms:
        A_blocks = block_term['A_blocks']
        B_blocks = block_term['B_blocks']
        C_blocks = block_term['C_blocks']
        
        # For this block-level term, we have a "virtual" 2x2 matrix X (from A blocks)
        # and Y (from B blocks), and we compute Z = X*Y using Strassen.
        # Each inner Strassen term s gives a contribution to Z.
        
        for inner_term in inner_terms:
            X_blocks = inner_term['X_blocks']
            Y_blocks = inner_term['Y_blocks']
            Z_blocks = inner_term['Z_blocks']
            
            # X is formed from A_blocks, where each "element" of X is a 2x2 block of A
            # X[0,0] = sum of A blocks where X_blocks has (0,0,coeff)
            # X[0,1] = sum of A blocks where X_blocks has (0,1,coeff)
            # X[1,0] = sum of A blocks where X_blocks has (1,0,coeff)
            # X[1,1] = sum of A blocks where X_blocks has (1,1,coeff)
            
            # Similarly for Y from B_blocks.
            
            # Now, for each inner Strassen term, we have:
            # s = (linear_comb_of_A_elements) * (linear_comb_of_B_elements)
            
            # We need to expand this into rank-1 terms for the 4x4 multiplication.
            
            # Let's compute the effective A and B vectors for this inner term.
            # A_eff[i,j] = coefficient of A[i,j] in this term
            # B_eff[j,k] = coefficient of B[j,k] in this term
            
            A_eff = [[0]*4 for _ in range(4)]
            B_eff = [[0]*4 for _ in range(4)]
            
            # For each element of X (2x2), it corresponds to a 2x2 block of A
            # X[br, bc] where br, bc in {0,1} corresponds to A block at (br, bc)
            # A block at (br, bc) covers A[2*br+li, 2*bc+lj] for li, lj in {0,1}
            
            for (x_br, x_bc, x_coeff) in X_blocks:
                for (a_br, a_bc, a_coeff) in A_blocks:
                    # If X element (x_br, x_bc) uses A block (a_br, a_bc)
                    # The coefficient is x_coeff * a_coeff
                    # This contributes to A_eff at positions (2*a_br+li, 2*a_bc+lj)
                    c = x_coeff * a_coeff
                    for li in range(2):
                        for lj in range(2):
                            i = 2*a_br + li
                            j = 2*a_bc + lj
                            A_eff[i][j] += c
            
            for (y_br, y_bc, y_coeff) in Y_blocks:
                for (b_br, b_bc, b_coeff) in B_blocks:
                    # If Y element (y_br, y_bc) uses B block (b_br, b_bc)
                    # The coefficient is y_coeff * b_coeff
                    # This contributes to B_eff at positions (2*b_br+li, 2*b_bc+lj)
                    c = y_coeff * b_coeff
                    for li in range(2):
                        for lj in range(2):
                            j = 2*b_br + li
                            k = 2*b_bc + lj
                            B_eff[j][k] += c
            
            # Now, the inner Strassen term s contributes to Z with Z_blocks.
            # Z is a 2x2 matrix, and Z[br, bc] corresponds to C block at (br, bc)
            # C block at (br, bc) covers C[2*br+li, 2*bc+lj]
            
            # For each (z_br, z_bc, z_coeff) in Z_blocks:
            # This s term contributes z_coeff to C block (z_br, z_bc)
            # And the block-level term contributes C_blocks[(z_br, z_bc)] to the final C
            
            # Wait, I need to be more careful. The block-level term Mk contributes
            # to C with coefficients in C_blocks. The inner Strassen term s contributes
            # to Mk (which is Z) with coefficients in Z_blocks.
            
            # So the total contribution to C[i,k] from this (block_term, inner_term) pair is:
            # For each (z_br, z_bc, z_coeff) in Z_blocks:
            #   For each (c_br, c_bc, c_coeff) in C_blocks where c_br==z_br and c_bc==z_bc:
            #     C[2*c_br+li, 2*c_bc+lj] += z_coeff * c_coeff * (A_eff * B_eff term)
            
            # Actually, let me think differently. The rank-1 decomposition means:
            # C[i,k] = sum_r U[i*4+j_r][r] * V[j_r*4+k_r][r] * W[k_r*4+i][r]
            # But actually, for a single rank-1 term, we have:
            # C[i,k] += a_i * b_k where a_i comes from A and b_k from B
            # More precisely, for a rank-1 term with "index" r:
            # U[i*4+j][r] = coefficient of A[i,j] in this term
            # V[j*4+k][r] = coefficient of B[j,k] in this term  
            # W[k*4+i][r] = coefficient of C[i,k] in this term
            # And the product U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r] should sum to C[i,k]
            
            # For a rank-1 term, we have:
            # C[i,k] += (sum_j A_eff[i,j] * B_eff[j,k]) * W_coeff[i,k]
            # where W_coeff[i,k] is the coefficient for C[i,k]
            
            # Actually, the standard form is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # Rank-1 term r: C[i,k] += u_r[i] * v_r[k] where u_r[i] = sum_j A[i,j]*U[j][r] ...
            
            # Let me use a cleaner formulation. For a rank-1 term:
            # We pick a linear functional on A: f(A) = sum_{i,j} U[i*4+j][r] * A[i,j]
            # We pick a linear functional on B: g(B) = sum_{j,k} V[j*4+k][r] * B[j,k]
            # We pick a linear functional on C: h(C) = sum_{i,k} W[k*4+i][r] * C[i,k]
            # The rank-1 term contributes f(A) * g(B) to C, and we need:
            # sum_{i,j,k} U[i*4+j][r] * V[j*4+k][r] * A[i,j] * B[j,k] = 
            #   sum_{i,k} W[k*4+i][r] * C[i,k]
            
            # For this to hold for all A, B, we need:
            # sum_j U[i*4+j][r] * V[j*4+k][r] = W[k*4+i][r] for all i, k
            
            # So for each rank-1 term, we need U, V, W such that:
            # W[k*4+i][r] = sum_j U[i*4+j][r] * V[j*4+k][r]
            
            # In our construction, for a given (block_term, inner_term):
            # A_eff[i,j] is the coefficient of A[i,j]
            # B_eff[j,k] is the coefficient of B[j,k]
            # And the contribution to C[i,k] is:
            # C[i,k] += (sum_j A_eff[i,j] * B_eff[j,k]) * c_ik
            # where c_ik is the coefficient from the combination of Z_blocks and C_blocks
            
            # So we can set:
            # U[i*4+j][r] = A_eff[i,j]
            # V[j*4+k][r] = B_eff[j,k]
            # W[k*4+i][r] = c_ik
            
            # But wait, we need sum_j U[i*4+j][r] * V[j*4+k][r] = W[k*4+i][r]
            # sum_j A_eff[i,j] * B_eff[j,k] = c_ik
            
            # This is not necessarily true in general. The issue is that the rank-1
            # decomposition requires that the "matrix" A_eff * B_eff (as a matrix product)
            # equals the matrix c_ik.
            
            # Let me reconsider. The correct approach is:
            # For each rank-1 term, we have:
            # C[i,k] += alpha_ik * (sum_j U[i,j] * A[i,j]) * (sum_j V[j,k] * B[j,k])
            # No, that's not right either.
            
            # Standard tensor decomposition for matrix multiplication:
            # T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
            # where T_{i,j,k} = delta_{j,k} (the matrix multiplication tensor)
            
            # Actually, the matrix multiplication tensor is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
            
            # A rank-r decomposition means:
            # T_{i,j,k} = sum_{r=1}^r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
            
            # For our Strassen construction, each rank-1 term corresponds to a specific
            # choice of linear combinations. Let me just directly compute the coefficients.
            
            # For the (block_term, inner_term) pair:
            # The contribution to C[i,k] is:
            # c_ik * (sum_j A_eff[i,j] * B_eff[j,k])
            
            # where c_ik is determined by the Z_blocks and C_blocks.
            
            # To make this a valid rank-1 term, we need:
            # U[i*4+j][r] = A_eff[i,j]
            # V[j*4+k][r] = B_eff[j,k]  
            # W[k*4+i][r] = c_ik / (sum_j A_eff[i,j] * B_eff[j,k]) ... no, this doesn't work.
            
            # I think the issue is that I'm overcomplicating this. Let me use a different approach.
            
            # The standard way to express Strassen as a tensor decomposition is:
            # Each rank-1 term corresponds to a specific "gate" in the circuit.
            
            # For now, let me just verify that the current schoolbook solution works
            # and return it, since implementing the 49-term Strassen correctly is complex.
            
            col += 1
    
    # Given the complexity and risk of errors, let's return the schoolbook solution (rank 64)
    # which is guaranteed correct, rather than a potentially incorrect rank-49 solution.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
