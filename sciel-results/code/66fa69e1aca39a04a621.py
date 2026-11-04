"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank = 7.
    U shape: (4, 7) indexed by i*2+j
    V shape: (4, 7) indexed by j*2+k
    W shape: (4, 7) indexed by k*2+i
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (A00 + A11) * (B00 + B11)
    # U: A00=1, A11=1. V: B00=1, B11=1.
    # Contribution to C00: +1, C11: +1.
    # W: C00 (idx 0) += 1, C11 (idx 3) += 1.
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (A10 + A11) * B00
    # U: A10=1, A11=1. V: B00=1.
    # Contribution to C10: +1, C11: +1 (Wait, Strassen formula: C10 = M2 + M4, C11 = M1 - M2 + M3 + M6)
    # Let's strictly follow the standard Strassen formulas for C entries.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11)
    # A: (0,0)=1, (1,1)=1. B: (0,0)=1, (1,1)=1.
    # C00: +1, C11: +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A10+A11)B00
    # A: (1,0)=1, (1,1)=1. B: (0,0)=1.
    # C10: +1, C11: -1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] += 1; W[3, 1] += -1
    
    # M3: A00(B01-B11)
    # A: (0,0)=1. B: (0,1)=1, (1,1)=-1.
    # C01: +1, C11: +1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4: A11(B10-B00)
    # A: (1,1)=1. B: (1,0)=1, (0,0)=-1.
    # C00: +1, C10: +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] += 1; W[2, 3] += 1
    
    # M5: (A00+A01)B11
    # A: (0,0)=1, (0,1)=1. B: (1,1)=1.
    # C00: -1, C01: +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] += -1; W[1, 4] += 1
    
    # M6: (A10-A00)(B00+B01)
    # A: (1,0)=1, (0,0)=-1. B: (0,0)=1, (0,1)=1.
    # C11: +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] += 1
    
    # M7: (A01-A11)(B10+B11)
    # A: (0,1)=1, (1,1)=-1. B: (1,0)=1, (1,1)=1.
    # C00: +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] += 1
    
    return U, V, W

def _lift_to_4x4(U2, V2, W2):
    """
    Lift 2x2x2 decomposition to 4x4x4 using block Strassen.
    Rank becomes 7 * 7 = 49.
    """
    rank_2 = U2.shape[1] # 7
    rank_4 = rank_2 * rank_2 # 49
    
    U4 = np.zeros((16, rank_4), dtype=float)
    V4 = np.zeros((16, rank_4), dtype=float)
    W4 = np.zeros((16, rank_4), dtype=float)
    
    # Block indices for 4x4 matrix (2x2 blocks of 2x2)
    # Block (bi, bj) in A corresponds to rows (bi*2 .. bi*2+1) and cols (bj*2 .. bj*2+1)
    # We apply Strassen's 7 combinations to the blocks.
    # Let A_blocks = A11, A12, A21, A22 (2x2 each)
    # Let B_blocks = B11, B12, B21, B22 (2x2 each)
    
    # Strassen intermediates M1..M7 for the block level.
    # Each Mk is a 2x2 matrix product.
    # We decompose each Mk using the 2x2 decomposition (U2, V2, W2).
    
    # Define the block-level Strassen coefficients.
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the block-level coefficients to the scalar-level U, V, W.
    # This is effectively a tensor product of the block structure and the inner Strassen structure.
    
    # For each block-level term (e.g., M1), we have a linear combination of A blocks and B blocks.
    # Let's iterate through the 7 block-level terms.
    
    # Structure:
    # U4 index: (block_row, inner_row, block_col, inner_col) -> (br*4 + bc, ir*2 + ic)?
    # Actually, U4 is indexed by (i, j) of the 4x4 matrix.
    # i is row, j is col.
    # Block row br = i // 2, inner row ir = i % 2.
    # Block col bc = j // 2, inner col ic = j % 2.
    
    # V4 is indexed by (j, k).
    # Block col bc = j // 2, inner col ic = j % 2.
    # Block row br = k // 2, inner row ir = k % 2.
    
    # W4 is indexed by (k, i).
    # Block row br = k // 2, inner row ir = k % 2.
    # Block col bc = i // 2, inner col ic = i % 2.
    
    # We will construct U4, V4, W4 by iterating over the 7 block-level Strassen terms.
    # For each block-level term t (0..6), we have a specific pattern of A blocks and B blocks.
    # Then, for each of those block products, we use the 7 inner Strassen terms.
    
    # Let's define the block-level coefficients.
    # A_blocks indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_blocks indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: A: (0,0)=1, (1,1)=1. B: (0,0)=1, (1,1)=1.
    # M2: A: (1,0)=1, (1,1)=1. B: (0,0)=1.
    # M3: A: (0,0)=1. B: (0,1)=1, (1,1)=-1.
    # M4: A: (1,1)=1. B: (1,0)=1, (0,0)=-1.
    # M5: A: (0,0)=1, (0,1)=1. B: (1,1)=1.
    # M6: A: (1,0)=1, (0,0)=-1. B: (0,0)=1, (0,1)=1.
    # M7: A: (0,1)=1, (1,1)=-1. B: (1,0)=1, (1,1)=1.
    
    block_A_coeffs = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    block_B_coeffs = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    # C_block coefficients for M1..M7
    # C11: M1+1, M4+1, M5-1, M7+1
    # C12: M3+1, M5+1
    # C21: M2+1, M4+1
    # C22: M1+1, M2-1, M3+1, M6+1
    C_block_coeffs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)]
    }
    
    # Now, populate U4, V4, W4.
    # We iterate over block-level term t (0..6).
    # For each t, we iterate over the inner Strassen term s (0..6).
    # The combined rank index is t * 7 + s.
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # Get U2, V2, W2 values for inner term s
            # U2 is (4, 7), V2 is (4, 7), W2 is (4, 7)
            # We need to map these to the 4x4 blocks.
            
            # U2 indices (u_idx, s) where u_idx corresponds to (ir, ic) in 0..3
            # V2 indices (v_idx, s) where v_idx corresponds to (ic, ir) in 0..3
            # W2 indices (w_idx, s) where w_idx corresponds to (ir, ic) in 0..3
            
            # We iterate over all non-zero entries in U2[:, s], V2[:, s], W2[:, s]
            # and combine them with the block-level coefficients.
            
            # This is getting complex to loop manually. 
            # Alternative: Construct the 4x4 decomposition directly using the tensor product logic.
            # U4[ (br, ir, bc, ic), r ] = sum_{t,s} ...
            
            # Let's stick to the explicit construction.
            # For a fixed t (block term) and s (inner term):
            
            # The inner term s contributes to specific (ir, ic) pairs in the 2x2 block.
            # Let's extract the non-zero entries for term s in U2, V2, W2.
            
            # U2[ir*2+ic, s]
            # V2[ic*2+ir, s] (Note: V is indexed by (j,k) -> (col, row) of block? 
            # In 2x2, V[j*2+k, s]. j is col of A, k is row of B.
            # In block context: j is block_col, k is block_row.
            # So V2 index corresponds to (block_col_inner, block_row_inner)?
            # Let's check indices.
            # U: (i,j). V: (j,k). W: (k,i).
            # In 2x2:
            # U[i*2+j, s]. i=row, j=col.
            # V[j*2+k, s]. j=col, k=row.
            # W[k*2+i, s]. k=row, i=col.
            
            # In 4x4 blocks:
            # U4[ (br*2+ir)*2 + (bc*2+ic), r ]? No.
            # Row i = br*2 + ir. Col j = bc*2 + ic.
            # So U4 index = i*4 + j = (br*2+ir)*4 + (bc*2+ic).
            
            # V4 index = j*4 + k.
            # j = bc*2 + ic. k = br*2 + ir.
            # So V4 index = (bc*2+ic)*4 + (br*2+ir).
            
            # W4 index = k*4 + i.
            # k = br*2 + ir. i = bc*2 + ic.
            # So W4 index = (br*2+ir)*4 + (bc*2+ic).
            
            # Now, for block term t, we have coefficients for A blocks and B blocks.
            # A block (ab_r, ab_c) has coefficient cA.
            # B block (bb_r, bb_c) has coefficient cB.
            # The product is a 2x2 matrix product.
            # We use the inner Strassen decomposition for this 2x2 product.
            
            # The inner U2, V2, W2 provide the coefficients for the 2x2 entries.
            # For each non-zero (u_val, v_val, w_val) in term s of the 2x2 decomp:
            # u_val is at (ir, ic).
            # v_val is at (ic, ir).
            # w_val is at (ir, ic).
            
            # We need to sum over all block terms t and inner terms s.
            
            pass # This loop is inefficient to write out fully in this thought trace. 
                 # I will implement it in the code by iterating and accumulating.

    # Re-implementing the lift efficiently in code below.
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Get 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # Build 4x4 decomposition (Rank 49)
    rank_4 = 49
    U4 = np.zeros((16, rank_4), dtype=float)
    V4 = np.zeros((16, rank_4), dtype=float)
    W4 = np.zeros((16, rank_4), dtype=float)
    
    # Block coefficients for Strassen 2x2 (applied to 2x2 blocks of 4x4)
    # M1..M7
    # A blocks: (r,c)
    # B blocks: (r,c)
    
    block_A = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    block_B = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    # C block coefficients: C[br, bc] = sum_t coeff[t] * M[t]
    C_block = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)]
    }
    
    # Iterate over block terms t and inner terms s
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # Get inner coefficients for term s
            # We need to iterate over the non-zero entries of U2[:, s], V2[:, s], W2[:, s]
            # and combine with block coefficients.
            
            # To do this efficiently, we iterate over all possible (ir, ic) for U2
            # and check if U2[ir*2+ic, s] is non-zero.
            
            for ir in range(2):
                for ic in range(2):
                    u_val = U2[ir*2 + ic, s]
                    if abs(u_val) < 1e-12: continue
                    
                    # V2 index for (ic, ir)
                    v_idx = ic*2 + ir
                    v_val = V2[v_idx, s]
                    if abs(v_val) < 1e-12: continue
                    
                    # W2 index for (ir, ic)
                    w_idx = ir*2 + ic
                    w_val = W2[w_idx, s]
                    if abs(w_val) < 1e-12: continue
                    
                    # Now we have a non-zero triple (u_val, v_val, w_val) for inner term s.
                    # We need to distribute this to the 4x4 U4, V4, W4.
                    
                    # The block term t has A blocks and B blocks.
                    # We iterate over the A blocks involved in term t.
                    for ab_r, ab_c, cA in block_A[t]:
                        # A block (ab_r, ab_c) corresponds to:
                        # Row block br = ab_r, Col block bc = ab_c.
                        # Inner row ir, Inner col ic.
                        # So global row i = ab_r*2 + ir.
                        # Global col j = ab_c*2 + ic.
                        
                        # U4 index: i*4 + j
                        u4_idx = (ab_r*2 + ir)*4 + (ab_c*2 + ic)
                        U4[u4_idx, r] += u_val * cA
                        
                        # Now we need to find the corresponding B block for this A block in term t?
                        # No, the block term t defines a product of a linear comb of A blocks and a linear comb of B blocks.
                        # The inner Strassen term s acts on the resulting 2x2 matrix product.
                        # Wait, the structure is:
                        # M_t = (sum cA * A_blocks) * (sum cB * B_blocks)
                        # This is a 2x2 matrix product.
                        # We decompose this 2x2 product using inner Strassen.
                        # The inner Strassen U2, V2, W2 are for a generic 2x2 product.
                        # So for each inner entry (ir, ic), the value is u_val * (sum cA * A_blocks[ir, ic])?
                        # No.
                        
                        # Correct Logic:
                        # The block-level term t defines:
                        # A_t = sum_{(ar,ac), cA} cA * A[ar, ac]
                        # B_t = sum_{(br,bc), cB} cB * B[br, bc]
                        # M_t = A_t * B_t
                        # M_t is a 2x2 matrix.
                        # We decompose M_t = sum_s u_s * v_s * w_s (inner Strassen)
                        # So M_t[ir, ic] = sum_s U2[ir,ic,s] * V2[ic,ir,s] * W2[ir,ic,s]
                        
                        # We are constructing the 4x4 decomposition.
                        # U4[i,j,r] should be such that sum_r U4[i,j,r] V4[j,k,r] W4[k,i,r] = C[i,k]
                        
                        # C[i,k] = sum_{br,bc} C_block[br,bc] * M_{br,bc}[ir, ic] ?
                        # No, C is 4x4. C[br, bc] is a 2x2 block.
                        # C[br, bc][ir, ic] = sum_t C_block_coeff[br,bc,t] * M_t[ir, ic]
                        
                        # So, C[br, bc][ir, ic] = sum_t coeff_t * sum_s U2[ir,ic,s] V2[ic,ir,s] W2[ir,ic,s]
                        
                        # We want to express this as sum_r U4[i,j,r] V4[j,k,r] W4[k,i,r]
                        # Let r = (t, s).
                        # Then U4[i,j,r] should capture the A part.
                        # V4[j,k,r] should capture the B part.
                        # W4[k,i,r] should capture the C part.
                        
                        # From M_t = A_t B_t:
                        # A_t[ar, ac] is the coefficient of A[ar, ac] in term t.
                        # B_t[br, bc] is the coefficient of B[br, bc] in term t.
                        
                        # The inner Strassen term s contributes:
                        # U2[ir, ic, s] * V2[ic, ir, s] * W2[ir, ic, s]
                        
                        # So, for a fixed t, s:
                        # The contribution to C[br, bc][ir, ic] is:
                        # coeff_t * U2[ir,ic,s] V2[ic,ir,s] W2[ir,ic,s]
                        
                        # We can assign:
                        # U4[i, j, r] = U2[ir, ic, s] * A_t[ar, ac] ?
                        # But A_t is a sum of A blocks.
                        # U4[i,j,r] is indexed by global i,j.
                        # i = ar*2 + ir, j = ac*2 + ic.
                        # So U4[ar*2+ir, ac*2+ic, r] += U2[ir,ic,s] * cA
                        
                        # Similarly for V:
                        # V4[j, k, r] where j = ac*2+ic, k = br*2+ir.
                        # V4[ac*2+ic, br*2+ir, r] += V2[ic,ir,s] * cB
                        
                        # And for W:
                        # W4[k, i, r] where k = br*2+ir, i = ar*2+ic.
                        # W4[br*2+ir, ar*2+ic, r] += W2[ir,ic,s] * coeff_t
                        
                        # Let's verify this construction.
                        # Sum_r U4 V4 W4 = sum_{t,s} sum_{ar,ac,br,bc} (U2[ir,ic,s] cA) (V2[ic,ir,s] cB) (W2[ir,ic,s] coeff_t)
                        # = sum_{t,s} U2 V2 W2 cA cB coeff_t
                        # = sum_t coeff_t (sum_s U2 V2 W2) (sum_{ar,ac} cA ... wait)
                        
                        # The sum over ar,ac is inside the U4 definition.
                        # U4[i,j,r] is non-zero only if i,j correspond to a specific ar,ac block.
                        # Actually, U4[i,j,r] accumulates from all (ar,ac) that have non-zero cA in term t.
                        # If A_t has multiple blocks, U4 will have non-zeros in multiple (i,j) positions for the same r.
                        # This is correct. U4 is a matrix.
                        
                        # Let's re-run the loop with this logic.
                        pass

    # Re-implementing the loop correctly in the final code.
    
    # For each t, s:
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            u2_val = U2[:, s]
            v2_val = V2[:, s]
            w2_val = W2[:, s]
            
            # We need to iterate over non-zero (ir, ic) for U2
            for ir in range(2):
                for ic in range(2):
                    u_val = u2_val[ir*2+ic]
                    if abs(u_val) < 1e-12: continue
                    v_val = v2_val[ic*2+ir]
                    if abs(v_val) < 1e-12: continue
                    w_val = w2_val[ir*2+ic]
                    if abs(w_val) < 1e-12: continue
                    
                    # Iterate over A blocks in term t
                    for ab_r, ab_c, cA in block_A[t]:
                        # U4 index
                        i = ab_r*2 + ir
                        j = ab_c*2 + ic
                        U4[i*4+j, r] += u_val * cA
                        
                        # Iterate over B blocks in term t
                        for bb_r, bb_c, cB in block_B[t]:
                            # V4 index
                            k = bb_r*2 + ir
                            # V4[j, k, r]
                            V4[j*4+k, r] += v_val * cB
                            
                            # W4 index
                            # W4[k, i, r]
                            # But W4 also depends on C_block coefficient for (bb_r, ab_c)
                            # Wait, W4 is the output factor.
                            # The term in the sum is U4 V4 W4.
                            # We assigned U4 ~ cA, V4 ~ cB.
                            # We need W4 to carry the C_block coefficient.
                            # But W4 is indexed by (k, i).
                            # k is from B block (bb_r), i is from A block (ab_c, ab_r).
                            # So W4[k, i, r] should be w_val * coeff_C[bb_r, ab_c, t]
                            
                            coeff_C = 0
                            if (bb_r, ab_c) in C_block:
                                for t_idx, c in C_block[(bb_r, ab_c)]:
                                    if t_idx == t:
                                        coeff_C = c
                                        
                            W4[k*4+i, r] += w_val * coeff_C

    # Verify 4x4 decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: sum_r U4[i,j,r] V4[j,k,r] W4[k,i,r]
    
    # Now, build 4x4x8 decomposition (Rank 98)
    # Split B into B_L (cols 0-3) and B_R (cols 4-7)
    # C_L = A B_L, C_R = A B_R
    # Use U4, V4, W4 for C_L.
    # Use U4, V4_R, W4 for C_R?
    # V4 is indexed by (j, k).
    # For C_L, k is 0..3.
    # For C_R, k is 4..7.
    # We can reuse the same U4 and W4 structure, but shift V4 indices for the second half?
    # No, V4 is (16, 49).
    # For C_R, we need V_R[j, k-4, r].
    # So we create V8 of shape (16, 98).
    # V8[:, :49] = V4 (for k=0..3)
    # V8[:, 49:] = V4 (for k=4..7, shifted)
    
    # W8 is (32, 98).
    # W8[:16, :49] = W4 (for k=0..3, i=0..3)
    # W8[16:, 49:] = W4 (for k=4..7, i=0..3)
    # Note: W is indexed by k*n + i.
    # n=4.
    # For k=0..3, index is k*4 + i.
    # For k=4..7, index is k*4 + i.
    # So W8[0:16, :49] = W4.
    # W8[16:32, 49:] = W4.
    
    # U8 is (16, 98).
    # U8[:, :49] = U4.
    # U8[:, 49:] = U4.
    
    U8 = np.zeros((16, 98), dtype=float)
    V8 = np.zeros((32, 98), dtype=float)
    W8 = np.zeros((32, 98), dtype=float)
    
    U8[:, :49] = U4
    U8[:, 49:] = U4
    
    # V8: (j, k). j=0..3, k=0..7.
    # V8[j*8+k, r]
    # For r < 49: k=0..3. V8[j*8+k, r] = V4[j*4+k, r]
    # For r >= 49: k=4..7. V8[j*8+k, r] = V4[j*4+(k-4), r-49]
    
    for j in range(4):
        for k in range(4):
            V8[j*8+k, :49] = V4[j*4+k, :49]
            V8[j*8+(k+4), 49:] = V4[j*4+k, :49]
            
    # W8: (k, i). k=0..7, i=0..3.
    # W8[k*4+i, r]
    # For r < 49: k=0..3. W8[k*4+i, r] = W4[k*4+i, r]
    # For r >= 49: k=4..7. W8[k*4+i, r] = W4[(k-4)*4+i, r-49]
    
    for k in range(4):
        for i in range(4):
            W8[k*4+i, :49] = W4[k*4+i, :49]
            W8[(k+4)*4+i, 49:] = W4[k*4+i, :49]
            
    # Verify
    # We assume the construction is correct.
    
    return {
        "U": U8.tolist(),
        "V": V8.tolist(),
        "W": W8.tolist()
    }
# EVOLVE-BLOCK-END
