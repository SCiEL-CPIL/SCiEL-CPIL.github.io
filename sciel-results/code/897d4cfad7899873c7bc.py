"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 16 for 4x4x1 part. Total 65.
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 multiplication (cols 0-3 of B, cols 0-3 of C)
    # We use Strassen's algorithm recursively.
    # First, define the 2x2x2 Strassen decomposition (Rank 7).
    # Indices for 2x2 matrices: 0..3
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen Rank 7 Coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # Term 1: (A00+A11) * (B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # C contribution: C00, C11?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to fill W2 such that sum(r) U2[:,r] V2[:,r] W2[:,r] = Tensor
    # Actually, we just need the U, V, W vectors for each rank-1 term.
    # The rank-1 terms are defined by the M_l.
    # For each M_l, we have a specific U_l, V_l.
    # And W_l is determined by the reconstruction coefficients.
    
    # Let's define U_l and V_l for l=1..7
    # l=1: A00+A11, B00+B11
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    
    # l=2: A10+A11, B00
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    
    # l=3: A00, B10-B11
    U2[0][2]=1
    V2[2][2]=1; V2[3][2]=-1
    
    # l=4: A11, B01-B00
    U2[3][3]=1
    V2[1][3]=1; V2[0][3]=-1
    
    # l=5: A00+A01, B11
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    
    # l=6: A10-A00, B00+B10
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[2][5]=1
    
    # l=7: A01-A11, B01+B11
    U2[1][6]=1; U2[3][6]=-1
    V2[1][6]=1; V2[3][6]=1
    
    # Now W2 (C coefficients)
    # C00 (idx 0): M1(1) + M4(1) - M5(1) + M7(1)
    # C01 (idx 1): M3(1) + M5(1)
    # C10 (idx 2): M2(1) + M4(1)
    # C11 (idx 3): M1(1) - M2(1) + M3(1) + M6(1)
    
    W2[0][0] = 1 # M1 to C00
    W2[0][3] = 1 # M4 to C00
    W2[0][4] = -1 # M5 to C00
    W2[0][6] = 1 # M7 to C00
    
    W2[1][2] = 1 # M3 to C01
    W2[1][4] = 1 # M5 to C01
    
    W2[2][1] = 1 # M2 to C10
    W2[2][3] = 1 # M4 to C10
    
    W2[3][0] = 1 # M1 to C11
    W2[3][1] = -1 # M2 to C11
    W2[3][2] = 1 # M3 to C11
    W2[3][5] = 1 # M6 to C11
    
    # Now build 4x4x4 using this 2x2x2 decomposition
    # A is 4x4. Blocks:
    # A00: rows 0-1, cols 0-1 (idx 0,1,4,5 in 4x4? No, 0,1,4,5 are 0,0;0,1;1,0;1,1)
    # Let's map 2x2 local indices (0,1,2,3) to 4x4 global indices.
    # Block (br, bc) where br, bc in {0,1}.
    # Global row = br*2 + r, col = bc*2 + c.
    # Local index in 2x2 block = r*2+c.
    
    # For each of the 7 Strassen steps l=0..6:
    # We form L_A^l and L_B^l.
    # Then we multiply them using the 2x2x2 decomposition.
    # This generates 7 rank-1 terms for each l.
    # Total 49 terms.
    
    col_offset = 0
    for l in range(7):
        # Determine which A blocks and B blocks are involved in L_A^l and L_B^l
        # And their coefficients.
        
        # L_A^l is a sum of A_blocks with coeffs.
        # L_B^l is a sum of B_blocks with coeffs.
        
        # We need to define the block coefficients for each l.
        # l=0: A00(1), A11(1); B00(1), B11(1)
        # l=1: A10(1), A11(1); B00(1)
        # l=2: A00(1); B10(1), B11(-1)
        # l=3: A11(1); B01(1), B00(-1)
        # l=4: A00(1), A01(1); B11(1)
        # l=5: A10(1), A00(-1); B00(1), B10(1)
        # l=6: A01(1), A11(-1); B01(1), B11(1)
        
        # Define lists of (block_row, block_col, coeff) for A and B
        A_blocks = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1), (1,1,-1)]
        ]
        B_blocks = [
            [(0,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (0,0,-1)],
            [(1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,1,1), (1,1,1)]
        ]
        
        for r in range(7):
            # U_global, V_global, W_global for this term
            # U_global is sum over A_blocks of coeff * U2_local_mapped
            # U2_local is a vector of size 4 (for the 2x2 matrix being multiplied)
            # We need to map the 2x2 local indices to the 4x4 global indices for the specific blocks.
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 20 # 4x5, but we only use cols 0-3 for this part
            W_vec = [0.0] * 20 # 5x4, but we only use cols 0-3 for this part
            
            # Fill U_vec
            for (br, bc, coeff) in A_blocks[l]:
                # U2_local[r] corresponds to local indices 0,1,2,3
                # Local index 0 -> row 0, col 0 in block
                # Local index 1 -> row 0, col 1 in block
                # Local index 2 -> row 1, col 0 in block
                # Local index 3 -> row 1, col 1 in block
                for li in range(4):
                    val = U2[li][r]
                    if val != 0:
                        lr = li // 2
                        lc = li % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        global_idx = gr * 4 + gc
                        U_vec[global_idx] += coeff * val
            
            # Fill V_vec
            for (br, bc, coeff) in B_blocks[l]:
                for li in range(4):
                    val = V2[li][r]
                    if val != 0:
                        lr = li // 2
                        lc = li % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        # B is 4x5. Index = row * 5 + col
                        global_idx = gr * 5 + gc
                        V_vec[global_idx] += coeff * val
            
            # Fill W_vec
            # W2_local[r] corresponds to local indices 0,1,2,3 of C (2x2)
            # C is 4x4. Blocks C00, C01, C10, C11.
            # The result of L_A^l * L_B^l is S_l.
            # S_l contributes to C blocks according to the reconstruction.
            # Wait, the W2 vector already contains the reconstruction coefficients for the 2x2 C.
            # So W2[li][r] is the coefficient for C_local[li] from M_l.
            # But S_l is a 2x2 matrix.
            # The blocks of C (4x4) are C00, C01, C10, C11.
            # C00 gets contributions from M1, M4, M5, M7.
            # C01 gets contributions from M3, M5.
            # etc.
            # The W2 vector is defined for the 2x2 C of the sub-problem.
            # In the recursive step, the "C" of the sub-problem is one of the blocks of the main C?
            # No. The sub-problem computes S_l, which is a 2x2 matrix.
            # S_l is NOT a block of C. S_l is an intermediate product.
            # The reconstruction combines S_l to form C.
            # My W2 definition above was for the final C of the 2x2 problem.
            # But here, the 2x2 problem is just a step.
            # The output of the 2x2 problem is S_l.
            # So W2 should be the identity for the 2x2 output?
            # No. The rank-1 decomposition of the 2x2 multiplication tensor gives U, V, W.
            # The W part of the rank-1 term is the vector in C-space.
            # When we use this decomposition to compute S_l, the "C-space" is the space of S_l.
            # So W2[li][r] is the coefficient for S_l[li].
            # Then, we need to map S_l[li] to the global C blocks.
            # S_l is a 2x2 matrix.
            # How does S_l map to C?
            # C00 = M1 + M4 - M5 + M7.
            # C01 = M3 + M5.
            # C10 = M2 + M4.
            # C11 = M1 - M2 + M3 + M6.
            # Here M1..M7 are the scalar results of the 7 steps?
            # No, in Strassen, M1..M7 are 2x2 matrices.
            # So C00 = M1 + M4 - M5 + M7 (matrix addition).
            # This means the block C00 of the final C is the sum of the matrices M1, M4, -M5, M7.
            # So for each element (li) of M_l (which is a 2x2 matrix), it contributes to the corresponding element of C00.
            # So if W2[li][r] is non-zero, it means the element li of the result matrix of step l contributes to the element li of the C block.
            # Which C block?
            # Step l contributes to C00, C01, C10, C11 based on the reconstruction.
            # My W2 definition above was for the 2x2 C of a standalone 2x2 multiplication.
            # But here, the "C" of the sub-multiplication is just the intermediate matrix M_l.
            # So for the sub-multiplication, W2 should be the Identity matrix (4x4)?
            # No, the rank-1 decomposition of the 2x2 multiplication tensor has W vectors that are the basis of C.
            # If we are computing M_l, then the "C" space is the space of M_l.
            # So the W vectors for the sub-decomposition are the standard basis of the 2x2 matrix space.
            # i.e., W_sub[li][r] is 1 if the term r contributes to element li of M_l.
            # In the standard Strassen decomposition, the W vectors are NOT identity.
            # They are the coefficients that reconstruct C from M's.
            # BUT, in the recursive step, we are not reconstructing the final C. We are computing M_l.
            # So we should use the decomposition of the 2x2 multiplication tensor where the output is M_l.
            # The tensor is $M_{2,2,2}$.
            # The rank-1 terms are $u \otimes v \otimes w$.
            # $w$ is in the space of the output matrix.
            # So for the sub-problem, $w$ is just the basis vector for the output element.
            # So we should use the U, V from Strassen, and W as Identity?
            # No, the Strassen decomposition $U, V, W$ satisfies $\sum U V W^T = \text{Tensor}$.
            # If we swap the roles, we can't just use Identity.
            # We must use the correct W from the Strassen decomposition.
            # But the Strassen W is for the final C.
            # If we use the Strassen decomposition to compute M_l, we are effectively computing the product of two 2x2 matrices.
            # The result is a 2x2 matrix.
            # So the W vectors should be the basis of the 2x2 matrix space.
            # Does the Strassen decomposition provide W vectors that are basis vectors?
            # No, they are linear combinations.
            # However, any valid decomposition works.
            # We can use the Strassen U, V and Identity W?
            # If we use Identity W, does $\sum U V I = \text{Tensor}$?
            # No. The tensor is $\sum_{i,j,k} A_{ij} B_{jk} C_{ik}$.
            # $\sum_r U_r \otimes V_r \otimes W_r = \text{Tensor}$.
            # If we set $W_r = e_r$ (basis), then $\sum U_r V_r e_r^T$ is a set of matrices.
            # This only works if the decomposition is such that $W_r$ are basis vectors.
            # Strassen's W are not basis vectors.
            
            # So we MUST use the Strassen W vectors.
            # But Strassen W vectors reconstruct the FINAL C of the 2x2 problem.
            # If the 2x2 problem is just computing M_l, then the "Final C" IS M_l.
            # So we use the Strassen W vectors.
            # And then we map M_l to the global C blocks.
            
            # Mapping M_l to Global C:
            # M_l is a 2x2 matrix.
            # It contributes to C00, C01, C10, C11.
            # Coefficients:
            # C00: M1(1), M4(1), M5(-1), M7(1)
            # C01: M3(1), M5(1)
            # C10: M2(1), M4(1)
            # C11: M1(1), M2(-1), M3(1), M6(1)
            
            # So for each element li of M_l (0..3), we add to the corresponding element of the target C blocks.
            
            # Define coefficients for M_l to C blocks
            # l=0 (M1): C00(1), C11(1)
            # l=1 (M2): C10(1), C11(-1)
            # l=2 (M3): C01(1), C11(1)
            # l=3 (M4): C00(1), C10(1)
            # l=4 (M5): C00(-1), C01(1)
            # l=5 (M6): C11(1)
            # l=6 (M7): C00(1)
            
            M_to_C = [
                [(0, 1), (3, 1)], # M1 -> C00, C11
                [(2, 1), (3, -1)], # M2 -> C10, C11
                [(1, 1), (3, 1)], # M3 -> C01, C11
                [(0, 1), (2, 1)], # M4 -> C00, C10
                [(0, -1), (1, 1)], # M5 -> C00, C01
                [(3, 1)], # M6 -> C11
                [(0, 1)] # M7 -> C00
            ]
            
            for (block_idx, coeff) in M_to_C[l]:
                # block_idx 0->C00, 1->C01, 2->C10, 3->C11
                br = block_idx // 2
                bc = block_idx % 2
                for li in range(4):
                    val = W2[li][r]
                    if val != 0:
                        lr = li // 2
                        lc = li % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        # W index: k*n + i. C is 4x5.
                        # Col is gc (0..3). Row is gr (0..3).
                        # Index = gc * 4 + gr
                        global_idx = gc * 4 + gr
                        W_vec[global_idx] += coeff * val
            
            # Add to global matrices
            for i in range(16):
                if U_vec[i] != 0:
                    U[i][col_offset] = U_vec[i]
            for i in range(20):
                if V_vec[i] != 0:
                    V[i][col_offset] = V_vec[i]
            for i in range(20):
                if W_vec[i] != 0:
                    W[i][col_offset] = W_vec[i]
            
            col_offset += 1

    # Part 2: 4x4x1 multiplication (col 4 of B, col 4 of C)
    # Rank 16
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col_offset] = 1
            V[j * 5 + 4][col_offset] = 1
            W[4 * 4 + i][col_offset] = 1
            col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
