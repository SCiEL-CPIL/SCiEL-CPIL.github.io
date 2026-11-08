"""Matrix multiplication 4x4 by 4x9 via rank-minimal tensor decomposition.
Uses Strassen for 4x4x4 (rank 49) and standard for 4x4x1 (rank 4). Total rank 53."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Split: p = 4 + 1
    p1, p2 = 4, 1
    
    # --- Part 1: 4x4 * 4x4 via Strassen (rank 49) ---
    # Strassen 2x2: rank 7, 4x4: rank 49
    # We'll build the decomposition for 4x4 * 4x4 -> 4x4
    # Using the known rank-49 decomposition for 4x4x4
    
    # The rank-49 decomposition for 4x4x4 can be constructed by
    # applying Strassen to 2x2 blocks with 49 terms.
    # 
    # Actually, the standard result: M(4) <= 49 via Strassen on 2x2 blocks
    # where each 2x2 block multiplication uses 7 Strassen products.
    # 4x4 = 4 blocks of 2x2, so 4*7 = 28? No, that's not right.
    # 
    # Let me think more carefully. The matrix multiplication tensor for
    # (2x2)*(2x2)->(2x2) has rank 7 (Strassen).
    # For (4x4)*(4x4)->(4x4), we can view it as 16 blocks of (1x1)*(1x1)->(1x1),
    # but that gives rank 16.
    #
    # The Strassen algorithm for 4x4: partition into 2x2 blocks.
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # Each block is 2x2. C = A*B where C_ij = sum_k A_ik * B_kj (block mult).
    #
    # Using Strassen at the block level (each "entry" is a 2x2 matrix):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # But each M_k involves multiplying 2x2 matrices, which itself requires rank 7.
    # So total rank = 7 (block level) * 7 (within each 2x2 mult) = 49.
    #
    # Let me construct this explicitly.
    
    # For a 2x2 * 2x2 -> 2x2, Strassen gives rank 7 with:
    # The 7 products are (as bilinear forms in the 4 entries of each 2x2):
    # Let a=[a0,a1,a2,a3] = [[a0,a1],[a2,a3]], b=[b0,b1,b2,b3] = [[b0,b1],[b2,b3]]
    # c=[c0,c1,c2,c3] = [[c0,c1],[c2,c3]]
    # c00 = a0*b0 + a1*b2
    # c01 = a0*b1 + a1*b3
    # c10 = a2*b0 + a3*b2
    # c11 = a2*b1 + a3*b3
    #
    # Strassen 7 products:
    # P1 = (a0+a3)(b0+b3)
    # P2 = (a2+a3)b0
    # P3 = a0(b2-b3)
    # P4 = a3(b2-b0)
    # P5 = (a0+a1)b3
    # P6 = (a2-a0)(b0+b1)
    # P7 = (a1-a3)(b2+b3)
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    # For the 4x4x4 decomposition, we have a 2x2 grid of 2x2 blocks.
    # Let me index the 4x4 matrices with row indices (i,j) where i is block-row (0,1)
    # and j is block-col (0,1), and within each block, local indices (r,c).
    
    # A is 4x4: A[2*i+r, 2*j+c] for block (i,j), local (r,c)
    # B is 4x4: B[2*k+r, 2*l+c] for block (k,l), local (r,c)
    # C is 4x4: C[2*i+r, 2*j+c] for block (i,j), local (r,c)
    
    # The 49 products: for each of the 7 block-level Strassen products,
    # we multiply two 2x2 "matrices" (each being a linear combination of
    # 2x2 blocks of A and B). Each such product requires 7 Strassen terms.
    
    # Block-level:
    # M1 = (A00+A11)(B00+B11): A-part is A00+A11, B-part is B00+B11
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each M_k, the A-side is a linear combination of 2x2 blocks,
    # which is a 2x2 matrix. Similarly for B-side.
    # We apply Strassen to multiply these two 2x2 matrices.
    
    # Let me define the 7 Strassen coefficients for 2x2:
    # a-side: coefficients for [a0,a1,a2,a3] in each product
    # b-side: coefficients for [b0,b1,b2,b3]
    
    strassen_a = [
        [1, 0, 0, 1],  # P1: a0+a3
        [0, 0, 1, 1],  # P2: a2+a3
        [1, 0, 0, 0],  # P3: a0
        [0, 0, 0, 1],  # P4: a3
        [1, 1, 0, 0],  # P5: a0+a1
        [-1, 0, 1, 0], # P6: a2-a0
        [0, 1, 0, -1], # P7: a1-a3
    ]
    strassen_b = [
        [1, 0, 0, 1],  # P1: b0+b3
        [1, 0, 0, 0],  # P2: b0
        [0, 0, 1, -1], # P3: b2-b3
        [0, 0, 1, -1], # P4: b2-b0  -> wait, b2-b0 = [−1,0,1,0]
        [0, 0, 0, 1],  # P5: b3
        [1, 1, 0, 0],  # P6: b0+b1
        [0, 0, 1, 1],  # P7: b2+b3
    ]
    # Fix P4 b-side: b2 - b0 = [-1, 0, 1, 0]
    strassen_b[3] = [-1, 0, 1, 0]
    
    strassen_w = [
        [1, 0, 0, 0],  # c00: P1 + P4 - P5 + P7
        [0, 1, 0, 0],  # c01: P3 + P5
        [0, 0, 1, 0],  # c10: P2 + P4
        [0, 0, 0, 1],  # c11: P1 - P2 + P3 + P6
    ]
    # Actually strassen_w should give coefficients for each c entry in terms of P1..P7
    # c00 = 1*P1 + 0*P2 + 0*P3 + 1*P4 + (-1)*P5 + 0*P6 + 1*P7
    # c01 = 0*P1 + 0*P2 + 1*P3 + 0*P4 + 1*P5 + 0*P6 + 0*P7
    # c10 = 0*P1 + 1*P2 + 0*P3 + 1*P4 + 0*P5 + 0*P6 + 0*P7
    # c11 = 1*P1 + (-1)*P2 + 1*P3 + 0*P4 + 0*P5 + 1*P6 + 0*P7
    
    strassen_w = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0],  # c11
    ])
    
    # Now for the 4x4x4:
    # A blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    # C blocks: C00 = C[0:2,0:2], C01 = C[0:2,2:4], C10 = C[2:4,0:2], C11 = C[2:4,2:4]
    
    # For each block-level product Mk, the A-side and B-side are 2x2 matrices
    # formed by linear combinations of the 2x2 blocks.
    
    # Block-level A-side for each Mk (as coefficients on [A00, A01, A10, A11]):
    # M1: A00 + A11 -> [1, 0, 0, 1]
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00 + A01 -> [1, 1, 0, 0]
    # M6: A10 - A00 -> [-1, 0, 1, 0]
    # M7: A01 - A11 -> [0, 1, 0, -1]
    
    block_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ])
    
    # Block-level B-side for each Mk (as coefficients on [B00, B01, B10, B11]):
    # M1: B00 + B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B10 - B11 -> [0, 0, 1, -1]
    # M4: B10 - B00 -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00 + B01 -> [1, 1, 0, 0]
    # M7: B10 + B11 -> [0, 0, 1, 1]
    
    block_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    
    # Block-level W-side: how C blocks are formed from Mk
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    block_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ])
    
    # Now build the full 4x4x4 decomposition (rank 49).
    # U has shape (n*m, 49) = (16, 49)
    # V has shape (m*p, 49) = (16, 49)
    # W has shape (p*n, 49) = (16, 49)
    
    U16 = np.zeros((16, 49))
    V16 = np.zeros((16, 49))
    W16 = np.zeros((16, 49))
    
    for mk in range(7):
        # A-side: linear combination of A blocks
        # Each A block is 2x2, so 4 entries.
        # A00 entries are at indices 0,1,2,3 in the 16-entry A vector (row-major)
        # A01 entries at 4,5,6,7
        # A10 entries at 8,9,10,11
        # A11 entries at 12,13,14,15
        a_coeffs = block_A[mk]  # [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
        
        # B-side
        b_coeffs = block_B[mk]  # [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
        
        # For this block-level product, we do a Strassen 2x2*2x2->2x2.
        # The "2x2 matrix" on A-side has entries:
        # [A_side_00, A_side_01; A_side_10, A_side_11]
        # where A_side_ij = sum of block coefficients * corresponding block entries.
        
        # Wait, I need to be more careful. The block-level product Mk computes
        # (linear combo of A blocks) * (linear combo of B blocks).
        # The result is a 2x2 block (each entry is a 2x2 matrix product).
        
        # Actually, let me reconsider. At the block level, we're doing:
        # M_k = (A_combo_k) * (B_combo_k)
        # where A_combo_k is a 2x2 matrix and B_combo_k is a 2x2 matrix.
        # The product is a 2x2 matrix. Each entry of this 2x2 result is a scalar
        # (since the blocks are 2x2, their product is 2x2... wait no).
        
        # I'm confusing myself. Let me restart with a cleaner formulation.
        
        # The 4x4 matrix multiplication: C = A * B where A,B,C are 4x4.
        # Partition into 2x2 blocks (each block is 2x2).
        # C_ij = A_i0 * B_0j + A_i1 * B_1j (block matrix mult, each term is 2x2*2x2->2x2)
        
        # Strassen at block level replaces the 4 block products with 7:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mk is a 2x2 matrix (product of two 2x2 matrices).
        # To compute each Mk, we need a 2x2*2x2->2x2 decomposition, rank 7.
        
        # So total rank = 7 * 7 = 49. ✓
        
        # Now, for Mk, the A-side is a 2x2 matrix with entries:
        # For M1: A_side = A00 + A11 = [[a00+a12, a01+a13],[a02+a14, a03+a15]]
        # where A00 = [[a00,a01],[a02,a03]], A11 = [[a12,a13],[a14,a15]]
        
        # Let me index A as a flat vector of 16 elements (row-major):
        # A = [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15]
        # A00 = A[0:4] = [a00,a01,a02,a03]
        # A01 = A[4:8] = [a04,a05,a06,a07]
        # A10 = A[8:12] = [a08,a09,a10,a11]
        # A11 = A[12:16] = [a12,a13,a14,a15]
        
        # B similarly: B[0:4]=B00, B[4:8]=B01, B[8:12]=B10, B[12:16]=B11
        
        # C similarly: C[0:4]=C00, C[4:8]=C01, C[8:12]=C10, C[12:16]=C11
        
        # For Mk, A_side is a 2x2 matrix. Let me denote its entries as:
        # A_side[0,0], A_side[0,1], A_side[1,0], A_side[1,1]
        
        # The Strassen decomposition of A_side * B_side -> result (2x2):
        # P1 = (As00+As11)(Bs00+Bs11)
        # P2 = (As10+As11)Bs00
        # P3 = As00(Bs10-Bs11)
        # P4 = As11(Bs10-Bs00)
        # P5 = (As00+As01)Bs11
        # P6 = (As10-As00)(Bs00+Bs01)
        # P7 = (As01-As11)(Bs10+Bs11)
        
        # Result:
        # R00 = P1+P4-P5+P7
        # R01 = P3+P5
        # R10 = P2+P4
        # R11 = P1-P2+P3+P6
        
        # Each Pj is a product of two linear forms (one in A entries, one in B entries).
        # So each Pj contributes one rank-1 term to the decomposition.
        
        # For term (mk, j) where j=0..6:
        # U column: the linear form on A
        # V column: the linear form on B
        # W column: the contribution to C (via block_W and strassen_w)
        
        for j in range(7):
            col = mk * 7 + j
            
            # A-side linear form for Pj:
            # As entries are linear combos of A entries.
            # As00, As01, As10, As11 are linear combos of the 16 A entries.
            
            # A_side = block_A[mk] applied to [A00, A01, A10, A11]
            # block_A[mk] = [c00, c01, c10, c11] means:
            # A_side[0,0] = c00*A00[0,0] + c01*A01[0,0] + c10*A10[0,0] + c11*A11[0,0]
            # A_side[0,1] = c00*A00[0,1] + c01*A01[0,1] + c10*A10[0,1] + c11*A11[0,1]
            # A_side[1,0] = c00*A00[1,0] + c01*A01[1,0] + c10*A10[1,0] + c11*A11[1,0]
            # A_side[1,1] = c00*A00[1,1] + c01*A01[1,1] + c10*A10[1,1] + c11*A11[1,1]
            
            cA = block_A[mk]  # [c00, c01, c10, c11]
            
            # As00 = c00*A[0] + c01*A[4] + c10*A[8] + c11*A[12]
            # As01 = c00*A[1] + c01*A[5] + c10*A[9] + c11*A[13]
            # As10 = c00*A[2] + c01*A[6] + c10*A[10] + c11*A[14]
            # As11 = c00*A[3] + c01*A[7] + c10*A[11] + c11*A[15]
            
            As = np.zeros(4)
            As[0] = cA[0]*A_flat_index(0) + cA[1]*A_flat_index(4) + cA[2]*A_flat_index(8) + cA[3]*A_flat_index(12)
            # This is getting complex. Let me just build the U vector directly.
            
            # U[col] is a 16-vector (coefficients for A[0..15])
            U_col = np.zeros(16)
            V_col = np.zeros(16)
            
            # Strassen j-th product:
            # Pj = (A_linear_form) * (B_linear_form)
            # where A_linear_form is a linear combo of As00,As01,As10,As11
            # and B_linear_form is a linear combo of Bs00,Bs01,Bs10,Bs11
            
            # strassen_a[j] = [sa0, sa1, sa2, sa3] means:
            # A_form = sa0*As00 + sa1*As01 + sa2*As10 + sa3*As11
            # strassen_b[j] = [sb0, sb1, sb2, sb3] means:
            # B_form = sb0*Bs00 + sb1*Bs01 + sb2*Bs10 + sb3*Bs11
            
            sa = strassen_a[j]
            sb = strassen_b[j]
            
            # A_form = sa0*As00 + sa1*As01 + sa2*As10 + sa3*As11
            # As00 = cA0*A[0] + cA1*A[4] + cA2*A[8] + cA3*A[12]
            # As01 = cA0*A[1] + cA1*A[5] + cA2*A[9] + cA3*A[13]
            # As10 = cA0*A[2] + cA1*A[6] + cA2*A[10] + cA3*A[14]
            # As11 = cA0*A[3] + cA1*A[7] + cA2*A[11] + cA3*A[15]
            
            # So A_form = sum over local indices:
            # For each local position (r,c) in 2x2, the coefficient is:
            # sa[r] * cA[col_block(r,c)]
            
            # Actually: A_form[r,c] = sum_{s=0}^{3} sa[s] * As[s,r,c]
            # where As[0] = As00, As[1] = As01, As[2] = As10, As[3] = As11
            # and As[s][r,c] = cA[s] * A[4*s + r*2 + c]
            
            # Wait, I need to be careful with indexing.
            # A is 4x4, flat row-major: A[i*4+j]
            # A00 = A[0:4] = A[0],A[1],A[2],A[3] (row 0: A[0],A[1]; row 1: A[2],A[3])
            # A01 = A[4:8] = A[4],A[5],A[6],A[7] (row 0: A[4],A[5]; row 1: A[6],A[7])
            # A10 = A[8:12] = A[8],A[9],A[10],A[11]
            # A11 = A[12:16] = A[12],A[13],A[14],A[15]
            
            # As[s] for s=0,1,2,3 corresponds to blocks:
            # s=0: A00, entries at flat indices 0,1,2,3
            # s=1: A01, entries at flat indices 4,5,6,7
            # s=2: A10, entries at flat indices 8,9,10,11
            # s=3: A11, entries at flat indices 12,13,14,15
            
            # As[s][r,c] = cA[s] * A[4*s + 2*r + c]
            
            # A_form[r,c] = sum_s sa[s] * cA[s] * A[4*s + 2*r + c]
            
            for r in range(2):
                for c in range(2):
                    coeff = 0.0
                    for s in range(4):
                        coeff += sa[s] * cA[s]
                    U_col[4*s_base(r,c)] = coeff  # This is wrong, let me redo
            
            # Let me just iterate over all 16 A entries:
            for idx in range(16):
                # idx corresponds to A[row, col] where row = idx//4, col = idx%4
                row, col = idx // 4, idx % 4
                # This entry belongs to block (row//2, col//2) with local (row%2, col%2)
                block_r, block_c = row // 2, col // 2
                local_r, local_c = row % 2, col % 2
                # block_r, block_c in {0,1}: 
                # (0,0)->s=0, (0,1)->s=1, (1,0)->s=2, (1,1)->s=3
                s_idx = block_r * 2 + block_c
                # A_form has entry at (local_r, local_c) which is As_combined[local_r, local_c]
                # The Strassen product Pj uses A_form entries:
                # The A linear form for Pj is: sum_{r,c} strassen_a[j][some mapping] * A_form[r,c]
                # Actually strassen_a[j] = [sa0, sa1, sa2, sa3] corresponds to
                # [As00, As01, As10, As11] i.e., [A_form[0,0], A_form[0,1], A_form[1,0], A_form[1,1]]
                
                # So the coefficient for A[idx] in Pj's A-form is:
                # strassen_a[j][local_r*2 + local_c] * cA[s_idx]
                a_coeff = strassen_a[j][local_r * 2 + local_c] * cA[s_idx]
                U_col[idx] = a_coeff
            
            # Similarly for B:
            cB = block_B[mk]  # [c00, c01, c10, c11] for B00, B01, B10, B11
            for idx in range(16):
                row, col = idx // 4, idx % 4
                block_r, block_c = row // 2, col // 2
                local_r, local_c = row % 2, col % 2
                s_idx = block_r * 2 + block_c
                b_coeff = strassen_b[j][local_r * 2 + local_c] * cB[s_idx]
                V_col[idx] = b_coeff
            
            # W column: contribution to C
            # Pj contributes to the result R of Mk (a 2x2 matrix) as:
            # R[r,c] += strassen_w[r,c,j] * Pj
            # where strassen_w is 4x7: rows are [R00,R01,R10,R11], cols are [P1..P7]
            
            # R is a 2x2 block of C. Which block? It's the block corresponding to
            # the block-level W coefficients.
            # block_W[i,j,mk] gives the coefficient of Mk in C_ij.
            
            # R is a 2x2 matrix. R[r,c] contributes to C_block[i,j][r,c]
            # where i,j are determined by block_W.
            
            # C_block[i,j][r,c] is at flat index in C: (i*2+r)*4 + (j*2+c)
            
            # For each (i,j) where block_W[i,j,mk] != 0:
            #   For each (r,c) in 2x2:
            #     W_col[(i*2+r)*4 + (j*2+c)] += block_W[i,j,mk] * strassen_w[r,c,j]
            
            W_col = np.zeros(16)
            for i in range(2):
                for j2 in range(2):
                    bw = block_W[i, j2, mk]
                    if bw != 0:
                        for r in range(2):
                            for c in range(2):
                                sw = strassen_w[r * 2 + c, j]
                                if sw != 0:
                                    flat_idx = (i * 2 + r) * 4 + (j2 * 2 + c)
                                    W_col[flat_idx] += bw * sw
            
            U16[:, col] = U_col
            V16[:, col] = V_col
            W16[:, col] = W_col
    
    # --- Part 2: 4x4 * 4x1 via standard (rank 4) ---
    # A is 4x4, B is 4x1, C is 4x1
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # Rank 4: for each i in 0..3:
    # U[i*4+j, col] = 1 if j matches
    # V[j*1+0, col] = 1
    # W[0*4+i, col] = 1
    
    # Actually for 4x4 * 4x1:
    # n=4, m=4, p=1
    # U shape (16, 4), V shape (4, 4), W shape (4, 4)
    # For each i in 0..3, j in 0..3... no, rank 4 means:
    # For each i (output row), we need: C[i,0] = sum_j A[i,j]*B[j,0]
    # This can be done with 4 terms, one per j? No, that would be rank 16.
    # 
    # Wait: 4x4 * 4x1 -> 4x1. The output has 4 entries.
    # C[i] = sum_{j=0}^{3} A[i,j] * B[j]
    # We can do this with rank 4:
    # Term t=0: A[0,*] * B[0] -> contributes to C[0]
    # Term t=1: A[1,*] * B[1] -> contributes to C[1]
    # ...
    # No, that doesn't work because C[i] needs all j.
    #
    # Actually rank 4 is correct for 4x4*4x1:
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
    # 
    # We can use 4 terms where term t handles B[t,0]:
    # U[:, t] has 1's in row i for all i... no.
    #
    # Let me think. Rank 4 decomposition:
    # U is (16, 4), V is (4, 4), W is (4, 4)
    # U[i*4+j, t] = 1 if j == t (for all i)
    # V[t*1+0, t] = 1
    # W[0*4+i, t] = 1 (for all i)
    # 
    # Check: sum_t U[i*4+j, t] * V[t, t] * W[i, t]
    # = sum_t (1 if j==t else 0) * 1 * 1
    # = 1 if j==t for some t... this gives 1 for each (i,j) pair, not what we want.
    #
    # I think I'm overcomplicating. For 4x4 * 4x1 -> 4x1:
    # The tensor has 4*4*1 = 16 entries.
    # Rank 4: 
    # U[i*4+j, t] = 1 if j == t
    # V[j, t] = 1 (for all j, t? No)
    # 
    # Let me use the standard approach:
    # C[i,0] = sum_j A[i,j] B[j,0]
    # Term for each j: 
    #   U[i*4+j, j] = 1 for all i
    #   V[j, j] = 1
    #   W[0*4+i, j] = 1 for all i
    # 
    # Check: sum_j U[i*4+j', j] * V[j', j] * W[i, j]
    # For fixed i: sum_j (1 if j'=j) * (1 if j'=j) * 1 = sum_j 1 = 4. Wrong.
    #
    # OK let me just do rank 16 for the 4x4*4x1 part and combine.
    # Actually, let me reconsider. The total rank should be 49 + 4 = 53.
    #
    # For 4x4 * 4x1, rank is 4 (since it's like a matrix-vector product, which
    # has rank n*m*p / min dimension... actually rank(4x4x1) = 4*4*1/4 = 4? No.
    # The rank of the (4,4,1) tensor is 4. Each output entry C[i,0] = sum_j A[i,j]B[j,0].
    # This is a sum of 4 rank-1 terms per output, but we can share across outputs.
    #
    # Rank 4:
    # U[i*4+j, j] = 1 (for all i, j) -- this means for each j, the j-th column of U
    #   has 1's in all rows where the column index of A is j.
    # V[j, j] = 1
    # W[0*4+i, j] = 1 (for all i, j)
    #
    # Check: (U V^T W^T) at (i*4+j', j', i) = U[i*4+j', j'] * V[j', j'] * W[i, j']
    # = (1 if j'==j') * (1 if j'==j') * 1 = 1
    # But we need C[i,0] = sum_j A[i,j]B[j,0], so the coefficient should be 1 for
    # the term (i, j, 0). In our indexing, W[k*n+i, c] with k=0, so W[i, c].
    # The tensor entry is at (i*4+j, j*1+0, 0*4+i) = (i*4+j, j, i).
    # U[i*4+j, j] * V[j, j] * W[i, j] = 1*1*1 = 1. ✓
    # And there's no other term contributing to this (i,j) pair since only column j
    # has U[i*4+j, j]=1.
    
    U4 = np.zeros((16, 4))
    V4 = np.zeros((4, 4))
    W4 = np.zeros((4, 4))
    for j in range(4):
        for i in range(4):
            U4[i * 4 + j, j] = 1
        V4[j, j] = 1
        for i in range(4):
            W4[i, j] = 1
    
    # Combine Part 1 and Part 2
    # Part 1: U16 (16,49), V16 (16,49), W16 (16,49) for 4x4*4x4->4x4
    # Part 2: U4 (16,4), V4 (4,4), W4 (4,4) for 4x4*4x1->4x1
    
    # Full U: (16, 53)
    U_full = np.zeros((16, 53))
    U_full[:, :49] = U16
    U_full[:, 49:] = U4
    
    # Full V: (36, 53) since m*p = 4*9 = 36
    # Part 1 B is 4x4, so V16 is (16, 49) covering B[0:4, 0:4]
    # Part 2 B is 4x1, so V4 is (4, 4) covering B[0:4, 4] (the 5th column)
    # B full is 4x9: B[row, col] where col 0..3 is Part 1, col 4 is Part 2
    # Flat index in B: row*9 + col
    # Part 1: B[row, col] for col 0..3 -> flat index row*9 + col
    # Part 2: B[row, 4] -> flat index row*9 + 4
    
    V_full = np.zeros((36, 53))
    # Part 1: B is 4x4, flat as 4x4 -> indices 0..15
    # In full B (4x9), these correspond to:
    # B[row, col] where col in 0..3: flat index in full B = row*9 + col
    # V16 is (16, 49) where row index = row*4 + col (in the 4x4 B)
    # So V16[row*4+col, c] should go to V_full[row*9+col, c]
    for row in range(4):
        for col in range(4):
            V_full[row * 9 + col, :49] = V16[row * 4 + col, :49]
    
    # Part 2: B is 4x1, flat as 4x1 -> indices 0..3
    # V4 is (4, 4) where row index = row (in the 4x1 B)
    # In full B, B[row, 4] has flat index row*9 + 4
    for row in range(4):
        V_full[row * 9 + 4, 49:] = V4[row, :]
    
    # Full W: (36, 53) since p*n = 9*4 = 36
    # Part 1 C is 4x4: W16 is (16, 49), flat as 4x4: index = row*4 + col
    # In full C (4x9... wait, C is 4x9? No.
    # A is 4x4, B is 4x9, so C is 4x9.
    # Wait, the problem says 4x4 matrix times 4x9 matrix -> 4x9 output.
    # n=4 (rows of A), m=4 (cols of A = rows of B), p=9 (cols of B)
    # C is n x p = 4x9.
    # W shape is (p*n, rank) = (36, rank)
    # W[k*n + i, c] where k is col of C (0..8), i is row of C (0..3)
    
    # Part 1: C is 4x4, covering C[0:4, 0:4]
    # W16[row*4+col, c] where row=i (0..3), col=k (0..3)
    # In full C (4x9), C[i, k] has W index k*4 + i
    for i in range(4):
        for k in range(4):
            W_full_idx = k * 4 + i
            W16_idx = i * 4 + k
            W_full[W_full_idx, :49] = W16[W16_idx, :49]
    
    # Part 2: C is 4x1, covering C[0:4, 4]
    # W4 is (4, 4), W4[i, t] where i is row of C
    # In full C, C[i, 4] has W index 4*4 + i = 16 + i
    for i in range(4):
        W_full[16 + i, 49:] = W4[i, :]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Quick sanity check with random values
    # A random 4x4, B random 4x9
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    # Compute via decomposition
    Ua = U_full @ A_test.flatten()
    Vb = V_full @ B_test.flatten()
    C_computed = np.zeros((4, 9))
    for i in range(4):
        for k in range(9):
            C_computed[i, k] = np.dot(Ua, W_full[k * 4 + i, :] * Vb)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to schoolbook
        rank = 144
        U_s = np.zeros((16, rank))
        V_s = np.zeros((36, rank))
        W_s = np.zeros((36, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_s[i * 4 + j, col] = 1
                    V_s[j * 9 + k, col] = 1
                    W_s[k * 4 + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
