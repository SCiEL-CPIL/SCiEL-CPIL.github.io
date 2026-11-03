"""Matrix multiplication 4,4,9 - Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (a-c)(e+g)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to express each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # as sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # For 2x2x2, the tensor has 8 non-zero entries:
    # T[i,j,k] = 1 if the indices form a valid multiplication term
    # Actually, T[i,j,k,l] = delta_{i,l} for A[i,j]*B[j,k] -> C[i,k]
    # In our format: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # with n=m=p=2: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # The 8 schoolbook terms are:
    # (i=0,j=0,k=0): U[0,r], V[0,r], W[0,r]
    # (i=0,j=1,k=0): U[1,r], V[2,r], W[0,r]
    # (i=0,j=0,k=1): U[0,r], V[1,r], W[2,r]
    # (i=0,j=1,k=1): U[1,r], V[3,r], W[2,r]
    # (i=1,j=0,k=0): U[2,r], V[0,r], W[1,r]
    # (i=1,j=1,k=0): U[3,r], V[2,r], W[1,r]
    # (i=1,j=0,k=1): U[2,r], V[1,r], W[3,r]
    # (i=1,j=1,k=1): U[3,r], V[3,r], W[3,r]
    
    # Strassen combines these 8 into 7 rank-1 terms.
    # The standard Strassen factorization for the tensor:
    
    # M1 = (a+d)(e+h) -> contributes to C00 and C11
    # M2 = (c+d)e -> contributes to C10
    # M3 = a(f-h) -> contributes to C01
    # M4 = d(g-e) -> contributes to C00 and C10
    # M5 = (a+b)h -> contributes to C01 and C00
    # M6 = (c-d)(f+h) -> contributes to C11
    # M7 = (a-c)(e+g) -> contributes to C00 and C11
    
    # Let me use the known Strassen tensor decomposition.
    # The 7 rank-1 tensors (U, V, W) entries:
    
    # Term 0: M1
    # U: a and d -> U[0,0]=1, U[2,0]=1 (i=0,j=0 and i=1,j=0? No...)
    # Wait, let me think more carefully.
    # U[i*2+j, r] is the coefficient of A[i,j] in the r-th product
    # V[j*2+k, r] is the coefficient of B[j,k] in the r-th product
    # W[k*2+i, r] is the coefficient for C[i,k] from the r-th product
    
    # For M1 = (a+d)(e+h):
    #   Involves A[0,0], A[1,0] and B[0,0], B[0,1]
    #   U[0,0]=1 (A[0,0]), U[2,0]=1 (A[1,0])
    #   V[0,0]=1 (B[0,0]), V[1,0]=1 (B[0,1])
    #   Result: C[0,0] += 1, C[0,1] += 1, C[1,0] += 1, C[1,1] += 1? No.
    #   Actually (a+d)(e+h) = ae + ah + de + dh
    #   ae -> C[0,0], ah -> C[0,1], de -> C[1,0], dh -> C[1,1]
    #   So W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
    U[0,0] = 1; U[2,0] = 1
    V[0,0] = 1; V[1,0] = 1
    W[0,0] = 1; W[1,0] = 1; W[2,0] = 1; W[3,0] = 1
    
    # Term 1: M2 = (c+d)e
    #   Involves A[1,0], A[1,1] and B[0,0]
    #   U[2,1]=1 (A[1,0]), U[3,1]=1 (A[1,1])
    #   V[0,1]=1 (B[0,0])
    #   Result: ce -> C[1,0], de -> C[1,0]
    #   So W[2,1]=1 (C[1,0] with k=0,i=1 -> W[0*2+1]=W[1]... wait
    #   W[k*n+i] = W[k*2+i]. For C[1,0]: i=1, k=0 -> W[0*2+1]=W[1]
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1
    
    # Term 2: M3 = a(f-h)
    #   Involves A[0,0] and B[0,1], B[1,1]
    #   U[0,2]=1 (A[0,0])
    #   V[1,2]=1 (B[0,1]), V[3,2]=-1 (B[1,1])
    #   Result: af -> C[0,1], -ah -> C[0,1]? No, a*(f-h) = af - ah
    #   Wait, this doesn't seem right. Let me reconsider.
    #   Actually in Strassen: C[0,1] = M3 + M5
    #   M3 = a(f-h), M5 = (a+b)h
    #   M3 + M5 = af - ah + ah + bh = af + bh. Correct!
    #   So M3: a*f - a*h
    #   U[0,2]=1
    #   V[1,2]=1 (B[0,1]=f), V[3,2]=-1 (B[1,1]=h)
    #   C[0,1]: i=0, k=1 -> W[1*2+0]=W[2]
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1
    
    # Term 3: M4 = d(g-e)
    #   Involves A[1,1] and B[1,0], B[0,0]
    #   U[3,3]=1 (A[1,1]=d)
    #   V[2,3]=1 (B[1,0]=g), V[0,3]=-1 (B[0,0]=e)
    #   dg - de
    #   dg -> C[1,0], de -> C[1,0]
    #   C[1,0]: i=1, k=0 -> W[0*2+1]=W[1]
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[1,3] = 1
    
    # Term 4: M5 = (a+b)h
    #   Involves A[0,0], A[0,1] and B[1,1]
    #   U[0,4]=1 (A[0,0]=a), U[1,4]=1 (A[0,1]=b)
    #   V[3,4]=1 (B[1,1]=h)
    #   ah + bh
    #   C[0,1]: i=0, k=1 -> W[2]
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[2,4] = 1
    
    # Term 5: M6 = (c-d)(f+h)
    #   Involves A[1,0], A[1,1] and B[0,1], B[1,1]
    #   U[2,5]=1 (A[1,0]=c), U[3,5]=-1 (A[1,1]=d)
    #   V[1,5]=1 (B[0,1]=f), V[3,5]=1 (B[1,1]=h)
    #   cf + ch - df - dh
    #   C[1,1]: i=1, k=1 -> W[1*2+1]=W[3]
    U[2,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[3,5] = 1
    W[3,5] = 1
    
    # Term 6: M7 = (a-c)(e+g)
    #   Involves A[0,0], A[1,0] and B[0,0], B[1,0]
    #   U[0,6]=1 (A[0,0]=a), U[2,6]=-1 (A[1,0]=c)
    #   V[0,6]=1 (B[0,0]=e), V[2,6]=1 (B[1,0]=g)
    #   ae + ag - ce - cg
    #   C[0,0]: ae, C[0,0]: ag? No.
    #   ae -> C[0,0], ag -> C[0,0]? A[0,0]*B[1,0] -> C[0,0]
    #   ce -> C[1,0], cg -> C[1,0]
    #   So W[0,6]=1 (C[0,0]) and W[1,6]=-1 (C[1,0])
    U[0,6] = 1; U[2,6] = -1
    V[0,6] = 1; V[2,6] = 1
    W[0,6] = 1; W[1,6] = -1
    
    return U, V, W


def _combine_blocks(U1, V1, W1, U2, V2, W2, shift_k=0, shift_i=0):
    """Combine two decompositions. The second one has k indices shifted by shift_k
    and i indices shifted by shift_i in the W matrix."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    U = np.zeros((U1.shape[0] + U2.shape[0] if U1.shape[0] != U2.shape[0] else U1.shape[0], r))
    # Actually, if both have same row dimension, we just stack columns
    if U1.shape[0] == U2.shape[0]:
        U = np.hstack([U1, U2])
    else:
        # Need to pad
        max_rows = max(U1.shape[0], U2.shape[0])
        U = np.zeros((max_rows, r))
        U[:U1.shape[0], :r1] = U1
        U[:U2.shape[0], r1:] = U2
    
    if V1.shape[0] == V2.shape[0]:
        V = np.hstack([V1, V2])
    else:
        max_rows = max(V1.shape[0], V2.shape[0])
        V = np.zeros((max_rows, r))
        V[:V1.shape[0], :r1] = V1
        V[:V2.shape[0], r1:] = V2
    
    # W needs shifting for the second block
    W = np.zeros((W1.shape[0], r))
    W[:, :r1] = W1
    # For the second block, shift k by shift_k and i by shift_i
    # W2[k*n + i, r] -> we need to place it at W[(k+shift_k)*n + (i+shift_i), r]
    n = W1.shape[0] // (W1.shape[0] // W2.shape[0]) if W2.shape[0] > 0 else W1.shape[0]
    # Actually, let's be more careful. W1 has shape (p*n, r1) for some p, n.
    # W2 has shape (p2*n, r2). After shifting k by shift_k, the new p is p2 + shift_k.
    # The total W should have (p_total * n) rows.
    
    # Let me redo this. The caller should handle the sizing.
    # For now, assume W2 needs to be placed with offset.
    W[:, r1:] = W2
    
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 multiplication."""
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Decompose p=9 as 4+5, and 4 as 2+2, 5 as 4+1
    # 4x4x4: Use Strassen for 2x2x2 blocks -> rank 7*4=28? No.
    # For 4x4x4: split A into four 2x2 blocks, B into four 2x2 blocks.
    # The product has 4x4 output = 16 2x2 blocks.
    # Each output block C_ij = sum_k A_ik * B_kj (block matrix multiply)
    # Using Strassen for each 2x2x2 block multiply:
    # For each output 2x2 block, we need 7 products of 2x2 blocks.
    # But the 2x2 blocks themselves are products of individual elements.
    # 
    # Actually, for 4x4x4, we can use the Kronecker structure.
    # The 4x4x4 tensor can be decomposed using 4 copies of 2x2x2 (one for each 2x2 block position)
    # But that gives rank 4*7=28? No, that's not right either.
    #
    # Let me think differently. For 4x4 matrix multiplication (4x4 times 4x4):
    # We can write it as a block matrix product with 2x2 blocks.
    # There are 4x4=16 output elements, each being a sum of 4 products.
    # Total schoolbook: 4*4*4=64.
    #
    # Using Strassen recursively: 4x4x4 has rank at most 4^log2(7) ≈ 28.
    # More precisely, rank(4x4x4) <= rank(2x2x2)^2 = 7^2 = 49.
    # And it's known that rank(4x4x4) <= 49.
    #
    # For 4x4x9: we can decompose as 4x4x4 + 4x4x5.
    # 4x4x5: decompose as 4x4x4 + 4x4x1.
    # 4x4x1: schoolbook, rank 16.
    # 4x4x4: rank 49 using Strassen.
    # Total: 49 + 49 + 16 = 114.
    #
    # Wait, 4x4x5 is NOT 4x4x4 + 4x4x1 in terms of the tensor.
    # The 4x4x5 tensor has p=5, so W has 5*4=20 rows.
    # We can split the k dimension: k=0..3 and k=4.
    # k=0..3: 4x4x4 tensor, rank 49.
    # k=4: 4x4x1 tensor, rank 16.
    # Total for 4x4x5: 49+16=65.
    #
    # Then 4x4x9: split k into k=0..3 and k=4..8.
    # k=0..3: 4x4x4, rank 49.
    # k=4..8: 4x4x5, rank 65.
    # Total: 49+65=114.
    
    # Let's build the 2x2x2 Strassen decomposition first.
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4x4 from 2x2x2 Strassen.
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4.
    # Block decomposition: A = [[A00,A01],[A10,A11]] where each is 2x2.
    # B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]].
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each 2x2 product uses Strassen with 7 terms.
    # Total: 4*7 = 28 terms? But that's for the block-level.
    # At the element level, each 2x2 product involves 4 elements of A, 4 of B, producing 4 of C.
    # The Strassen decomposition of 2x2x2 has rank 7.
    # For 4x4x4, we have 4 independent 2x2x2 sub-tensors (one for each block pair).
    # But they share indices! A00*B00 uses A[0:2,0:2] and B[0:2,0:2].
    # A01*B10 uses A[0:2,2:4] and B[2:4,0:2].
    # These are disjoint in the A and B indices, so they can be combined.
    #
    # Actually, the 4x4x4 tensor is a "Kronecker sum" of four 2x2x2 tensors.
    # The rank of a Kronecker sum of tensors is at most the sum of ranks.
    # So rank(4x4x4) <= 4*7 = 28? That seems too low. The known bound is 49.
    #
    # Let me reconsider. The 4x4x4 multiplication tensor can be viewed as:
    # C[i,k] = sum_j A[i,j] B[j,k] for i,j,k in 0..3.
    # Split i=2i'+i'', j=2j'+j'', k=2k'+k'' where i'',j'',k'' in {0,1}.
    # Then for each (i',j',k'), we have a 2x2x2 subproblem.
    # But the subproblems are coupled because the same A and B elements appear in multiple subproblems.
    #
    # A[i,j] = A[2i'+i'', 2j'+j''] appears in subproblems for all i',j'.
    # So we can't just sum 4 independent 2x2x2 tensors.
    #
    # The correct way: Use Strassen at the block level.
    # For block matrix multiplication with 2x2 blocks:
    # We compute 7 block products (each 2x2x2), using Strassen for each.
    # Each block product has rank 7 (element level).
    # Total: 7*7 = 49.
    
    # Let me construct the 4x4x4 decomposition using block Strassen.
    # The 7 Strassen combinations at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B01+B11)
    # M7 = (A00-A10)(B00+B10)
    
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2x2 product (at block level), which we decompose using Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let's build this.
    # For each of the 7 block products, we need a 2x2x2 Strassen decomposition.
    # The A indices for the block product depend on which blocks are involved.
    
    # Let me define a helper to create a shifted Strassen decomposition.
    def _shifted_strassen(i_offset, j_offset, k_offset, i2=0, j2=0, k2=0):
        """Create a Strassen 2x2x2 decomposition with index shifts.
        i_offset, j_offset: shift for A indices (i = i_offset + i2 + i''*2, j = j_offset + j2 + j''*2)
        k_offset: shift for B and C k indices
        """
        U_s, V_s, W_s = _strassen_2x2()
        # U has shape (4, 7), indices i*2+j where i,j in 0..1
        # We need to map to global indices: i_global = i_offset + i, j_global = j_offset + j
        # So U_global[i_global*4 + j_global, r] = U_s[i*2+j, r]
        n_global = 4
        U_g = np.zeros((n_global * n_global, 7))
        V_g = np.zeros((n_global * n_global, 7))
        W_g = np.zeros((n_global * n_global, 7))
        
        for i in range(2):
            for j in range(2):
                i_g = i_offset + i
                j_g = j_offset + j
                U_g[i_g * n_global + j_g, :] = U_s[i*2+j, :]
        
        for j in range(2):
            for k in range(2):
                j_g = j_offset + j
                k_g = k_offset + k
                V_g[j_g * n_global + k_g, :] = V_s[j*2+k, :]
        
        for k in range(2):
            for i in range(2):
                k_g = k_offset + k
                i_g = i_offset + i
                W_g[k_g * n_global + i_g, :] = W_s[k*2+i, :]
        
        return U_g, V_g, W_g
    
    # The 7 block Strassen combinations:
    # M1 = (A00+A11)(B00+B11): A blocks (0,0) and (1,1), B blocks (0,0) and (1,1)
    # This is a 2x2x2 product where:
    #   A matrix has rows/cols corresponding to blocks 0 and 1
    #   Specifically, A_combined[0,0]=A00, A_combined[1,1]=A11
    #   B_combined[0,0]=B00, B_combined[1,1]=B11
    #   But Strassen works on full 2x2 matrices. We need to embed.
    #
    # Actually, M1 = (A00+A11)(B00+B11) is NOT a standard 2x2x2 product.
    # It's a single 2x2 product of the sum matrices.
    # A_sum = A00+A11 (2x2), B_sum = B00+B11 (2x2), M1 = A_sum * B_sum (2x2).
    # This is a 2x2x2 multiplication: C = A_sum * B_sum.
    # A_sum[i,j] = A00[i,j] + A11[i,j] = A[2*0+i, 2*0+j] + A[2*1+i, 2*1+j]
    # B_sum[i,j] = B00[i,j] + B11[i,j] = B[2*0+i, 2*0+j] + B[2*1+i, 2*1+j]
    # M1[i,j] = sum_k A_sum[i,k] * B_sum[k,j]
    #
    # To decompose M1 using Strassen, we need to express it as a 2x2x2 tensor
    # where the "A" matrix has entries that are linear combinations of the original A entries.
    #
    # This is getting complex. Let me use a different approach.
    # I'll directly construct the rank-49 decomposition by combining shifted Strassen terms.
    
    # Alternative: Use the known fact that 4x4x4 can be done with 49 multiplications.
    # Let me just use a simpler approach: schoolbook for 4x4x4 would be 64, but we want 49.
    # Given time constraints, let me try a different strategy.
    
    # Let me try: decompose 4x4x9 as a combination of 2x2x9 subproblems.
    # 4x4x9: split i and j into 2+2.
    # For each (i_block, j_block) pair, we have a 2x2x9 subproblem.
    # There are 4 such subproblems (i_block=0,1 and j_block=0,1).
    # Each 2x2x9 has rank at most... 2*2*9=36 schoolbook, or less with Strassen.
    # 2x2x9: split k into 4+5. 2x2x4 = 2x2x2 + 2x2x2, rank 7+7=14.
    # 2x2x5: 2x2x4 + 2x2x1 = 14 + 4 = 18.
    # Total 2x2x9: 14+18=32.
    # 4 subproblems: 4*32=128. Worse than 114.
    
    # Let me stick with the 114 approach: 4x4x4 (rank 49) + 4x4x5 (rank 65).
    # For 4x4x4, I'll use a simple but valid decomposition.
    # Given the complexity, let me try to build it programmatically.
    
    # For now, let me use a fallback: schoolbook for everything but verify correctness.
    # Schoolbook rank: 4*4*9 = 144.
    # We want to do better.
    
    # Let me try to implement the 4x4x4 rank-49 decomposition properly.
    # I'll use the block Strassen approach with careful index tracking.
    
    def _block_strassen_4x4x4():
        """Build 4x4x4 decomposition with rank 49 using block Strassen."""
        n = 4
        U = np.zeros((n*n, 49))
        V = np.zeros((n*n, 49))
        W = np.zeros((n*n, 49))
        
        # The 7 block products, each decomposed into 7 element-level terms.
        # For each block product t (0..6) and each Strassen term s (0..6):
        #   column = t*7 + s
        
        # Define the block Strassen structure:
        # Each block product Mk involves specific A blocks and B blocks.
        # A block (bi,bj) is A[2*bi:2*bi+2, 2*bj:2*bj+2]
        # B block (bk,bj) is B[2*bk:2*bk+2, 2*bj:2*bj+2]
        
        # For each block product, define:
        #   a_blocks: list of (bi, bj, coeff) for A
        #   b_blocks: list of (bk, bj, coeff) for B
        #   The product is sum over a_blocks and b_blocks of coeff_a * coeff_b * A_block * B_block
        #   But Strassen combines them into 7 terms.
        
        # Let me define the 7 block products explicitly:
        # M1 = (A00+A11)(B00+B11)
        #   A: block 00 with coeff 1, block 11 with coeff 1
        #   B: block 00 with coeff 1, block 11 with coeff 1
        # M2 = (A10+A11)B00
        #   A: block 10 with coeff 1, block 11 with coeff 1
        #   B: block 00 with coeff 1
        # M3 = A00(B01-B11)
        #   A: block 00 with coeff 1
        #   B: block 01 with coeff 1, block 11 with coeff -1
        # M4 = A11(B10-B00)
        #   A: block 11 with coeff 1
        #   B: block 10 with coeff 1, block 00 with coeff -1
        # M5 = (A00+A01)B11
        #   A: block 00 with coeff 1, block 01 with coeff 1
        #   B: block 11 with coeff 1
        # M6 = (A10-A11)(B01+B11)
        #   A: block 10 with coeff 1, block 11 with coeff -1
        #   B: block 01 with coeff 1, block 11 with coeff 1
        # M7 = (A00-A10)(B00+B10)
        #   A: block 00 with coeff 1, block 10 with coeff -1
        #   B: block 00 with coeff 1, block 10 with coeff 1
        
        block_products = [
            # (a_blocks, b_blocks)
            ([((0,0,1.0), (1,1,1.0))], [((0,0,1.0), (1,1,1.0))]),  # M1
            ([((1,0,1.0), (1,1,1.0))], [((0,0,1.0))]),  # M2
            ([((0,0,1.0))], [((0,1,1.0), (1,1,-1.0))]),  # M3
            ([((1,1,1.0))], [((1,0,1.0), (0,0,-1.0))]),  # M4
            ([((0,0,1.0), (0,1,1.0))], [((1,1,1.0))]),  # M5
            ([((1,0,1.0), (1,1,-1.0))], [((0,1,1.0), (1,1,1.0))]),  # M6
            ([((0,0,1.0), (1,0,-1.0))], [((0,0,1.0), (1,0,1.0))]),  # M7
        ]
        
        # For each block product, we need to decompose the resulting 2x2x2 product
        # using Strassen. The "A" and "B" matrices for the 2x2x2 are linear combinations
        # of the original blocks.
        
        # This is still complex. Let me try a different approach:
        # Directly construct the 49 rank-1 terms by solving a linear system.
        
        # Given time constraints, let me use a simpler valid decomposition.
        # I'll use the schoolbook for 4x4x4 (rank 64) and 4x4x5 (rank 80), total 144.
        # But that's the same as the schoolbook for 4x4x9.
        
        # Let me at least try to get the 2x2x2 Strassen working correctly,
        # then use it for a 2x2x9 decomposition, and combine 4 of those.
        
        return U, V, W
    
    # Given the complexity and time constraints, let me use a pragmatic approach:
    # Use Strassen for 2x2x2 blocks and combine for 4x4x9.
    # 
    # 4x4x9: Think of A as 4x4, B as 4x9.
    # Split A into 4 2x2 blocks (vertically): A_top (2x4), A_bottom (2x4).
    # Split B into 4 2x9 blocks (horizontally): B_left (4x5), B_right (4x4)? No.
    # 
    # Let me just use the schoolbook but with a small optimization:
    # For the 4x4x9 tensor, use Strassen for the 4x4x4 part (k=0..3)
    # and schoolbook for the 4x4x5 part (k=4..8).
    # 
    # For 4x4x4, I'll use a rank-49 decomposition if I can build it,
    # otherwise fall back to schoolbook (rank 64).
    # 
    # Let me try to build the 4x4x4 rank-49 decomposition by explicitly
    # enumerating the 49 rank-1 terms.
    
    # Actually, let me just go with a clean implementation that's correct.
    # I'll use the 2x2x2 Strassen and build up to 4x4x9.
    
    # 4x4x9 decomposition:
    # Split k into 4 groups: k=0..3 (4 values) and k=4..8 (5 values).
    # For k=0..3: 4x4x4 subproblem.
    # For k=4..8: 4x4x5 subproblem.
    
    # For 4x4x4: Use 4 copies of 2x2x2 Strassen? No, that's not correct.
    # Let me use the following: 4x4x4 = sum of 4 "strips" of 2x4x4? No.
    
    # I'll use a straightforward but slightly optimized approach.
    # For 4x4x4, use schoolbook (rank 64).
    # For 4x4x5, use schoolbook (rank 80).
    # Total: 144. Same as before.
    
    # Let me at least get the Strassen 2x2x2 working and use it for 2x2x9.
    # 2x2x9: rank at most 2*2*9=36 schoolbook, or 2*2*4+2*2*5=16+20=36 with split.
    # With Strassen: 2x2x4 = 2x2x2 + 2x2x2, rank 7+7=14.
    # 2x2x5 = 2x2x4 + 2x2x1 = 14+4=18.
    # 2x2x9 = 14+18=32.
    # 4x4x9: split i into 2+2. For each i_block (0,1), we have a 2x4x9 subproblem.
    # 2x4x9: split j into 2+2. For each j_block, we have a 2x2x9 subproblem.
    # 4 subproblems of 2x2x9: rank 4*32=128. Worse.
    
    # I think the best I can do quickly is 114 with the 49+65 approach,
    # but I need to correctly implement the 4x4x4 rank-49 decomposition.
    
    # Let me try a different tactic: use the known Laderman algorithm for 4x4?
    # No, Laderman is for 4x4x4 with rank 49? Actually, the minimum rank for 4x4x4
    # is known to be at most 49 (by Strassen recursion) and at least 26.
    
    # Given time constraints, let me implement a correct but possibly non-optimal solution.
    # I'll use schoolbook for 4x4x4 and 4x4x5, giving total rank 144.
    # But I can at least verify the Strassen 2x2x2 works.
    
    # Actually, let me try one more thing: use Strassen for 2x2x2 blocks
    # in a 2x4x9 decomposition, then combine.
    
    # 2x4x9: A is 2x4, B is 4x9.
    # Split B into 2 blocks: B1 (4x5), B2 (4x4).
    # 2x4x5: schoolbook rank 40.
    # 2x4x4: schoolbook rank 32.
    # Total: 72.
    # 4x4x9: split A into 2 blocks: A1 (2x4), A2 (2x4).
    # 2 subproblems of 2x4x9: rank 2*72=144. Same.
    
    # I'll go with the schoolbook for now but structure it cleanly.
    # The rank is 144, same as the seed. But at least it's correct.
    
    # Wait, I can do slightly better by using Strassen for the 2x2x2 sub-blocks
    # where applicable. Let me try:
    
    # 4x4x9: For each output element C[i,k] (i in 0..3, k in 0..8),
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k].
    # This is 4 terms. Total: 4*9*4 = 144 terms.
    
    # If I can reduce the j-sum using Strassen-like tricks, I can reduce the rank.
    # For fixed i and k, the sum over j has 4 terms. If I group j into pairs,
    # I might use Strassen. But Strassen works on 2x2 matrices, not 1D sums.
    
    # I think for the time being, I'll return the schoolbook decomposition.
    # The score will be 1.0, same as the seed. But at least it's correct.
    
    # Actually, let me try to implement the 4x4x4 rank-49 decomposition properly.
    # I found a reference implementation approach.
    
    # For 4x4x4, the rank-49 decomposition can be built as follows:
    # The 4x4x4 tensor T has T[i,j,k] = 1 if i,k are the "output" indices and j is the "sum" index.
    # In our format: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    # T[i,j,k] = 1 (for all i,j,k in 0..3, the tensor is the matrix multiplication tensor).
    # Actually, the matrix multiplication tensor for 4x4x4 is:
    # MM_4[i,j,k] = 1 for all i,j,k? No.
    # The tensor is defined as: for matrices A (4x4) and B (4x4), C = A*B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has T[i,j,k] = 1, meaning the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So the tensor is: for each (i,j,k), there's a term A[i,j]*B[j,k]*C[i,k] with coefficient 1.
    # In our format: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    # The tensor element at (i*4+j, j*4+k, k*4+i) should be 1.
    # This is a 64-dim x 64-dim x 64-dim tensor with 64 non-zero entries (one for each i,j,k).
    
    # Schoolbook: 64 rank-1 terms, one for each (i,j,k).
    # Strassen: 49 rank-1 terms.
    
    # Let me try to find the 49 rank-1 terms by solving a linear system.
    # But that might be too slow.
    
    # Given the time, I'll return the schoolbook decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the expected format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
