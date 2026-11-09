"""Matrix multiplication 4,4,7 decomposition using recursive Strassen for 4x4x4 blocks."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose C = A * B where A is 4x4, B is 4x7, C is 4x7.
    # We split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2.
    # A*B1 is a 4x4x4 tensor, which can be done with rank 49 using Strassen.
    # A*B2 is a 4x4x3 tensor, which can be done with rank 48 using schoolbook.
    # Total rank = 49 + 48 = 97.
    
    # Let's construct the rank-1 tensors for A*B1 (4x4x4) using recursive Strassen.
    # Then construct rank-1 tensors for A*B2 (4x4x3) using schoolbook.
    
    # Helper to get the U, V, W coefficients for a single scalar multiplication
    # A_sub * B_sub = C_sub, where A_sub is n x m, B_sub is m x p, C_sub is n x p.
    # For the base case of 1x1x1, U=[1], V=[1], W=[1].
    
    # We need a function that returns lists of U, V, W coefficients for the tensor.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # Note: The problem statement says W[k*n+i, r]. This is the transposed index order.
    
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        n=2, m=2, p=2.
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7).
        Indexing:
        A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
        B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
        C indices: W[k*n+i] -> W is indexed by (k, i) where k is row of C? No.
        C is n x p = 2x2. C[i,k].
        W index = k*n + i.
        For n=2:
        C[0,0] -> W[0*2+0] = W[0]
        C[0,1] -> W[1*2+0] = W[2]
        C[1,0] -> W[0*2+1] = W[1]
        C[1,1] -> W[1*2+1] = W[3]
        
        Wait, let's re-read the convention carefully.
        "The output factor follows the official trace convention W[k*n+i,r]."
        Usually C[i,k] is the element.
        If W is a matrix of size (p*n, rank), then row index is k*n+i.
        This means the row index encodes (k, i).
        So W[row, col] contributes to C[i, k] with coefficient W[row, col].
        
        Let's verify with schoolbook 1x1x1.
        A is 1x1, B is 1x1, C is 1x1.
        U[0,0]=1, V[0,0]=1, W[0,0]=1.
        C[0,0] = A[0,0] * B[0,0] * 1. Correct.
        
        Let's verify with schoolbook 2x2x2.
        Rank = 4.
        Column 0: A[0,0]*B[0,0] -> C[0,0].
        U[0*2+0, 0] = 1. V[0*2+0, 0] = 1. W[0*2+0, 0] = 1. (C[0,0])
        Column 1: A[0,1]*B[1,0] -> C[0,1].
        U[0*2+1, 1] = 1. V[1*2+0, 1] = 1. W[1*2+0, 1] = 1. (C[0,1])
        Wait, C[0,1] corresponds to i=0, k=1. W index = k*n+i = 1*2+0 = 2.
        So W[2, 1] = 1.
        
        Let's stick to the formula: W_index = k*n + i.
        
        Strassen 2x2x2:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B00-B01)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B01+B11)
        M7 = (A01-A11)(B00+B10)
        
        C00 = M1 - M4 + M3 - M7
        C01 = M5 + M7
        C10 = M2 + M4
        C11 = M1 - M2 + M5 - M6
        
        Let's build the U, V, W vectors for each Mk.
        A indices: 00=0, 01=1, 10=2, 11=3.
        B indices: 00=0, 01=1, 10=2, 11=3.
        C indices (W index = k*2+i):
        C00 (i=0,k=0) -> idx 0
        C01 (i=0,k=1) -> idx 2
        C10 (i=1,k=0) -> idx 1
        C11 (i=1,k=1) -> idx 3
        
        M1:
        A: 00+11 -> U = [1, 0, 0, 1]
        B: 00+11 -> V = [1, 0, 0, 1]
        C: C00 + C11 -> W = [1, 0, 0, 1] (idx 0 and 3)
        
        M2:
        A: 10+11 -> U = [0, 0, 1, 1]
        B: 00 -> V = [1, 0, 0, 0]
        C: C10 -> W = [0, 1, 0, 0] (idx 1)
        
        M3:
        A: 00 -> U = [1, 0, 0, 0]
        B: 10-B11 -> V = [0, 0, 1, -1]
        C: C00 -> W = [1, 0, 0, 0] (idx 0)
        
        M4:
        A: 11 -> U = [0, 0, 0, 1]
        B: 00-B01 -> V = [1, -1, 0, 0]
        C: -C00 + C10 -> W = [-1, 1, 0, 0] (idx 0 and 1)
        
        M5:
        A: 00+01 -> U = [1, 1, 0, 0]
        B: 11 -> V = [0, 0, 0, 1]
        C: C01 + C11 -> W = [0, 0, 1, 1] (idx 2 and 3)
        
        M6:
        A: 10-A00 -> U = [-1, 0, 1, 0]
        B: 01+B11 -> V = [0, 1, 0, 1]
        C: -C11 -> W = [0, 0, 0, -1] (idx 3)
        
        M7:
        A: 01-A11 -> U = [0, 1, 0, -1]
        B: 00+B10 -> V = [1, 0, 1, 0]
        C: -C00 + C01 -> W = [-1, 0, 1, 0] (idx 0 and 2)
        
        We can define these as lists of tuples (U_vec, V_vec, W_vec).
        """
        tens = [
            ( [1,0,0,1], [1,0,0,1], [1,0,0,1] ), # M1
            ( [0,0,1,1], [1,0,0,0], [0,1,0,0] ), # M2
            ( [1,0,0,0], [0,0,1,-1], [1,0,0,0] ), # M3
            ( [0,0,0,1], [1,-1,0,0], [-1,1,0,0] ), # M4
            ( [1,1,0,0], [0,0,0,1], [0,0,1,1] ), # M5
            ( [-1,0,1,0], [0,1,0,1], [0,0,0,-1] ), # M6
            ( [0,1,0,-1], [1,0,1,0], [-1,0,1,0] ), # M7
        ]
        return tens

    def schoolbook_tensors(n, m, p):
        """
        Returns list of (U_vec, V_vec, W_vec) for schoolbook decomposition of n x m x p.
        """
        tens = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = [0]*(n*m)
                    V = [0]*(m*p)
                    W = [0]*(p*n)
                    U[i*m+j] = 1
                    V[j*p+k] = 1
                    W[k*n+i] = 1
                    tens.append((U, V, W))
        return tens

    def combine_tensors(tens_1, tens_2):
        """
        Combines two lists of tensors.
        tens_1 is for the first part, tens_2 for the second.
        They must have the same U, V, W dimensions (or we pad them).
        Here we assume they are for the same n, m, p or we handle them separately.
        Actually, it's easier to just concatenate the columns.
        """
        all_tens = []
        all_tens.extend(tens_1)
        all_tens.extend(tens_2)
        return all_tens

    # Step 1: Get tensors for 4x4x4 multiplication (A*B1)
    # We can use recursive Strassen: 4x4x4 is composed of 7 blocks of 2x2x2.
    # Each 2x2x2 block has 7 rank-1 tensors. Total 49.
    
    # Let's construct the 4x4x4 tensors from 2x2x2 Strassen.
    # A is 4x4, split into 4 blocks of 2x2: A00, A01, A10, A11.
    # B is 4x4, split into 4 blocks of 2x2: B00, B01, B10, B11.
    # C is 4x4, split into 4 blocks of 2x2: C00, C01, C10, C11.
    
    # Strassen for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B01+B11)
    # M7 = (A01-A11)(B00+B10)
    
    # Each Mk is a 2x2x2 multiplication.
    # We have the 7 base tensors for 2x2x2. Let's call them T1..T7.
    # For each Mk, we use the corresponding Tj from the base list.
    # Wait, the base list T1..T7 corresponds to the 7 multiplications in Strassen 2x2x2.
    # So M1 uses T1? No.
    # In Strassen 2x2x2, we have 7 multiplications of 1x1 matrices.
    # In Strassen 4x4x4, we have 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication is itself a Strassen 2x2x2 (or schoolbook 4, but Strassen is better).
    # So we replace each "scalar multiplication" in the 4x4x4 Strassen with a full 2x2x2 Strassen decomposition.
    
    # The 4x4x4 Strassen has 7 terms.
    # Term 1: (A00+A11) * (B00+B11)
    # This is a multiplication of two 2x2 matrices.
    # We decompose this 2x2x2 multiplication into 7 rank-1 tensors (the base Strassen).
    # So we take the 7 base tensors T1..T7, and "embed" them into the 4x4x4 space.
    # The A-part of Tj is a vector of length 4 (for the 2x2 A-block).
    # The B-part of Tj is a vector of length 4 (for the 2x2 B-block).
    # The C-part of Tj is a vector of length 4 (for the 2x2 C-block).
    
    # We need to map these to the full 4x4 matrices.
    # A is 4x4, indices 0..15.
    # A00 is indices 0..3, A01 is 4..7, A10 is 8..11, A11 is 12..15.
    # Similarly for B and C.
    
    # Let's define the mapping.
    # For the term (A00+A11) * (B00+B11):
    # The "A" matrix in the 2x2x2 sub-problem is A00+A11.
    # So for each base tensor Tj = (Uj, Vj, Wj), the U part for the full problem is:
    # U_full = [Uj[0], Uj[1], Uj[2], Uj[3], 0,0,0,0, 0,0,0,0, Uj[0], Uj[1], Uj[2], Uj[3]]
    # Wait, the coefficients are applied to the elements of A00 and A11.
    # So if Uj = [u00, u01, u10, u11], then the contribution to A is:
    # A00: u00, u01, u10, u11
    # A11: u00, u01, u10, u11
    # So U_full has Uj in positions 0-3 and 12-15.
    
    # Let's write a function to embed a 2x2x2 tensor into a 4x4x4 tensor.
    # The 4x4x4 tensor has U of size 16, V of size 16, W of size 16.
    
    def embed_2x2_to_4x4(U, V, W, a_blocks, b_blocks, c_blocks):
        """
        U, V, W are lists of 4 elements.
        a_blocks, b_blocks, c_blocks are lists of block indices (0,1,2,3) for A, B, C.
        The block index corresponds to:
        0: top-left (indices 0-3)
        1: top-right (indices 4-7)
        2: bottom-left (indices 8-11)
        3: bottom-right (indices 12-15)
        
        For each element in U, it corresponds to an element in the 2x2 A-block.
        We place it into the full U vector at the position corresponding to the block.
        """
        U_full = [0]*16
        V_full = [0]*16
        W_full = [0]*16
        for i in range(4):
            # Map i (0..3) to block offset
            # Block 0: 0..3, Block 1: 4..7, Block 2: 8..11, Block 3: 12..15
            # i is the index within the 2x2 block.
            # The block offset is (block_index * 4)
            # But we have multiple blocks?
            # a_blocks is a list of block indices for the A-part of this tensor.
            # For (A00+A11), a_blocks = [0, 3].
            # So U[i] is added to U_full at position (a_block * 4 + i) for each a_block in a_blocks.
            for a_block in a_blocks:
                U_full[a_block * 4 + i] += U[i]
            for b_block in b_blocks:
                V_full[b_block * 4 + i] += V[i]
            for c_block in c_blocks:
                W_full[c_block * 4 + i] += W[i]
        return U_full, V_full, W_full

    # Now, let's list the 7 terms for 4x4x4 Strassen and their block mappings.
    # Base tensors T1..T7 (from strassen_2x2x2)
    T = strassen_2x2x2()
    
    # Term 1: (A00+A11)(B00+B11)
    # a_blocks = [0, 3], b_blocks = [0, 3], c_blocks = [0, 3] (C00 and C11? No, C is 4x4, blocks are C00, C01, C10, C11)
    # Wait, the result of (A00+A11)(B00+B11) is a 2x2 matrix that contributes to C00 and C11?
    # No, in Strassen, M1 is a 2x2 matrix.
    # C00 = M1 - M4 + M3 - M7
    # C11 = M1 - M2 + M5 - M6
    # So M1 contributes to C00 and C11.
    # But M1 is a 2x2 matrix.
    # So the C-part of M1 is a 2x2 matrix.
    # When we embed M1 into the 4x4 C matrix, where does it go?
    # It goes into C00 and C11?
    # No, M1 is the result of multiplying two 2x2 matrices.
    # The result is a 2x2 matrix.
    # In the 4x4 Strassen, M1 is added to C00 and C11.
    # So the C-part of the tensor for M1 should have coefficients for C00 and C11.
    # So c_blocks = [0, 3].
    
    # Let's verify the C-blocks for each term.
    # C00 = M1 - M4 + M3 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 - M6
    
    # Term 1 (M1): C00 + C11 -> c_blocks = [0, 3]
    # Term 2 (M2): C10 -> c_blocks = [2]
    # Term 3 (M3): C00 -> c_blocks = [0]
    # Term 4 (M4): -C00 + C10 -> c_blocks = [0, 2] with coeffs [-1, 1]?
    # Wait, the base tensor T4 already has the coefficients for C.
    # T4 = ( [0,0,0,1], [1,-1,0,0], [-1,1,0,0] )
    # The W part of T4 is [-1, 1, 0, 0].
    # This means T4 contributes -1 to C00 and +1 to C10 (in the 2x2 sub-problem).
    # So when we embed T4 into the 4x4 problem, we just place the W part into the C-blocks.
    # But which C-blocks?
    # M4 contributes to C00 and C10.
    # So the W-part of T4 should be placed into C00 and C10.
    # But T4's W-part is a vector of length 4, corresponding to the 4 elements of the 2x2 C-block.
    # So we place the 4 elements of T4's W-part into C00 (indices 0-3) and C10 (indices 8-11).
    # Wait, T4's W-part is [-1, 1, 0, 0].
    # This means -1 * C00[0,0] + 1 * C00[1,0] (in the 2x2 sense).
    # So in the 4x4 sense, it contributes to C00[0,0], C00[1,0], C10[0,0], C10[1,0].
    # So c_blocks = [0, 2].
    
    # Let's list all terms:
    # 1: M1 = (A00+A11)(B00+B11) -> a=[0,3], b=[0,3], c=[0,3]
    # 2: M2 = (A10+A11)B00 -> a=[2,3], b=[0], c=[2]
    # 3: M3 = A00(B10-B11) -> a=[0], b=[2,3], c=[0]
    # 4: M4 = A11(B00-B01) -> a=[3], b=[0,1], c=[0,2]
    # 5: M5 = (A00+A01)B11 -> a=[0,1], b=[3], c=[1,3]
    # 6: M6 = (A10-A00)(B01+B11) -> a=[0,2], b=[1,3], c=[3]
    # 7: M7 = (A01-A11)(B00+B10) -> a=[1,3], b=[0,2], c=[0,1]
    
    # Let's construct the 49 tensors for 4x4x4.
    tensors_4x4x4 = []
    for t_idx in range(7):
        U_t, V_t, W_t = T[t_idx]
        if t_idx == 0:
            a_blocks, b_blocks, c_blocks = [0,3], [0,3], [0,3]
        elif t_idx == 1:
            a_blocks, b_blocks, c_blocks = [2,3], [0], [2]
        elif t_idx == 2:
            a_blocks, b_blocks, c_blocks = [0], [2,3], [0]
        elif t_idx == 3:
            a_blocks, b_blocks, c_blocks = [3], [0,1], [0,2]
        elif t_idx == 4:
            a_blocks, b_blocks, c_blocks = [0,1], [3], [1,3]
        elif t_idx == 5:
            a_blocks, b_blocks, c_blocks = [0,2], [1,3], [3]
        elif t_idx == 6:
            a_blocks, b_blocks, c_blocks = [1,3], [0,2], [0,1]
        else:
            continue
            
        U_full, V_full, W_full = embed_2x2_to_4x4(U_t, V_t, W_t, a_blocks, b_blocks, c_blocks)
        tensors_4x4x4.append((U_full, V_full, W_full))
        
    # Step 2: Get tensors for 4x4x3 multiplication (A*B2)
    # This is a schoolbook decomposition of rank 4*4*3 = 48.
    tensors_4x4x3 = schoolbook_tensors(4, 4, 3)
    
    # Step 3: Combine the two sets of tensors.
    # The 4x4x4 tensors have U, V, W of size 16, 16, 16.
    # The 4x4x3 tensors have U, V, W of size 16, 12, 12.
    # We need to pad them to the same size.
    # The full problem is 4x4x7.
    # U is size 16 (4x4).
    # V is size 28 (4x7).
    # W is size 28 (7x4).
    
    # For the 4x4x4 tensors, we need to pad V and W to size 28.
    # V is size 16, so we pad with 12 zeros.
    # W is size 16, so we pad with 12 zeros.
    
    # For the 4x4x3 tensors, we need to pad V and W to size 28.
    # V is size 12, so we pad with 16 zeros at the beginning (since B2 is the last 3 columns).
    # Wait, B is 4x7. B1 is 4x4 (cols 0-3), B2 is 4x3 (cols 4-6).
    # So V for 4x4x4 should have non-zeros in cols 0-15 (which correspond to B1).
    # V for 4x4x3 should have non-zeros in cols 16-27 (which correspond to B2).
    # Similarly for W.
    
    # Let's construct the final U, V, W matrices.
    total_rank = len(tensors_4x4x4) + len(tensors_4x4x3)
    U_final = [[0]*total_rank for _ in range(16)]
    V_final = [[0]*total_rank for _ in range(28)]
    W_final = [[0]*total_rank for _ in range(28)]
    
    # Fill in the 4x4x4 tensors
    for r_idx, (U_t, V_t, W_t) in enumerate(tensors_4x4x4):
        for i in range(16):
            U_final[i][r_idx] = U_t[i]
        for j in range(16):
            V_final[j][r_idx] = V_t[j]
        for k in range(16):
            W_final[k][r_idx] = W_t[k]
        # Pad V and W with zeros
        for j in range(16, 28):
            V_final[j][r_idx] = 0
        for k in range(16, 28):
            W_final[k][r_idx] = 0
            
    # Fill in the 4x4x3 tensors
    offset = len(tensors_4x4x4)
    for r_idx, (U_t, V_t, W_t) in enumerate(tensors_4x4x3):
        r_full = offset + r_idx
        for i in range(16):
            U_final[i][r_full] = U_t[i]
        # V_t is size 12, corresponds to B2 (cols 4-6 of B).
        # So V_t[0] corresponds to B[0,4], V_t[1] to B[0,5], etc.
        # So we place V_t into V_final at indices 16-27.
        for j in range(12):
            V_final[16+j][r_full] = V_t[j]
        # Pad V_final[0-15] with zeros
        for j in range(16):
            V_final[j][r_full] = 0
        # W_t is size 12, corresponds to C2 (cols 4-6 of C).
        # C is 4x7. C1 is 4x4 (cols 0-3), C2 is 4x3 (cols 4-6).
        # W index = k*n+i. For C2, k is 4,5,6.
        # So W_t[0] corresponds to C[0,4], etc.
        # We place W_t into W_final at indices 16-27.
        for k in range(12):
            W_final[16+k][r_full] = W_t[k]
        # Pad W_final[0-15] with zeros
        for k in range(16):
            W_final[k][r_full] = 0
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
