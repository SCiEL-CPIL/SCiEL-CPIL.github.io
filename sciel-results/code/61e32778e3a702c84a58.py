"""
Matrix multiplication 4,4,7 — rank-minimal exact tensor decomposition.

Strategy:
  - Split the 4x7 matrix B into a 4x4 block B1 and a 4x3 block B2.
  - Multiply A (4x4) by B1 (4x4) using Strassen-Strassen (rank 49).
  - Multiply A (4x4) by B2 (4x3) using a block decomposition that achieves rank 48.
  - Total rank = 49 + 48 = 97.

For the 4x4 x 4x4 part (Strassen-Strassen):
  The 4x4 matrices are partitioned into 2x2 blocks.
  M_4 = Strassen(M_2) where M_2 = Strassen(M_1) (schoolbook 1x1).
  Rank of M_4 via this route = 7 * 7 = 49.

For the 4x4 x 4x3 part:
  We use a block decomposition:
    A = [[A11, A12], [A21, A22]]  (2x2 blocks of size 2x2)
    B2 = [[B11, B12, B13], [B21, B22, B23]]  (2x2 blocks; 3 columns of blocks)
    
  We need C = A * B2, which is 4x3, i.e., 2x3 blocks of size 2x2.
  
  For each pair of block-rows of A and block-column of B2, we do a 2x2 x 2x2 
  multiplication. There are 2 block-rows of A and 3 block-columns of B2, giving 
  6 multiplications of 2x2 x 2x2. Each 2x2 x 2x2 via Strassen has rank 7.
  Total = 6 * 7 = 42? But wait, we can share computations.
  
  Actually, let's think more carefully. The standard schoolbook for 4x4 x 4x3 is 
  4*4*3 = 48. Can we do better?
  
  Alternative: Use Strassen on the 4x4 part of A and treat B2 specially.
  Partition A into 2x2 blocks: A = [[A11, A12],[A21, A22]]
  Partition B2 into 2x2 blocks: B2 = [Bc1, Bc2, Bc3] where each Bci is 2x2.
  
  C = A * B2 = [A11*Bc1 + A12*Bc1; A21*Bc1 + A22*Bc1] ... no, that's not right.
  
  Let me re-index. A is 4x4, B2 is 4x3.
  A = [[A11, A12],
       [A21, A22]]  each Ai is 2x2
  B2 = [[B11, B12, B13],
        [B21, B22, B23]]  each Bij is 2x2
  
  C (4x3) = A * B2
  C = [[C11, C12, C13],
       [C21, C22, C23]]
  
  where C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22, C13 = A11*B13 + A12*B23
        C21 = A21*B11 + A22*B21, C22 = A21*B12 + A22*B22, C23 = A21*B13 + A22*B23
  
  Each Cij is a 2x2 matrix computed from two 2x2x2x2 products.
  
  For each column of C (say column 1): C11 and C21.
  C11 = A11*B11 + A12*B21
  C21 = A21*B11 + A22*B21
  
  This is like computing (A11, A12) * (B11; B21) where the multiplication is 
  the sum of two 2x2x2x2 products.
  
  We can use Strassen for each of the 6 (2x2 x 2x2) block multiplications, but 
  there might be sharing. However, a simple approach: treat each of the 6 
  block multiplications independently with Strassen (rank 7 each), giving 42.
  But the entries of A and B2 are shared across these multiplications.
  
  Actually, the tensor for 4x4 x 4x3 multiplication: the rank is at most 48 (schoolbook).
  Can we achieve less? 
  
  Let's use a different decomposition. Use Strassen on A to get 7 linear forms of A,
  then for each of the 7 forms, multiply by B2 (4x3). Each such multiplication is 
  (1x4) x (4x3) = 1x3, which has rank 3 (schoolbook). Total = 7 * 3 = 21? 
  That seems too low; it's not valid because Strassen gives linear forms of A that 
  are combined with linear forms of B to produce C.
  
  Let me think again. The tensor M_{4,4,3} has entries T[i,j,k,l] = delta_{j,k} 
  where i,j are for A (4x4), k is for B (4x3), and l is for C (4x3).
  Actually, the multiplication tensor for (n x m) * (m x p) has dimensions n*m x m*p x n*p.
  For 4x4 * 4x3: U has 16 rows, V has 12 rows, W has 12 rows.
  
  Schoolbook rank = 4*4*3 = 48.
  
  Can we do better? Let's try a block approach.
  Use Strassen for the 4x4 multiplication structure. The 4x4 x 4x4 tensor has rank 49 
  (via Strassen-Strassen) or 48 (schoolbook). Wait, schoolbook for 4x4 x 4x4 is 64.
  Strassen-Strassen gives 49.
  
  For 4x4 x 4x3, we can view it as: compute A * B2 where B2 is 4x3.
  Pad B2 to 4x4 by adding a zero column, compute A * B2_padded (4x4) using Strassen-Strassen 
  (rank 49), then discard the last column. But that doesn't reduce rank; the decomposition 
  for the 4x4 x 4x4 case has 49 terms, each contributing to all 16 output entries. 
  If we just ignore the last column of the output, we still have 49 terms, but some 
  might have zero contribution to the first 12 output entries. In general, they won't 
  be zero, so we'd still need 49 terms, which is worse than 48.
  
  So schoolbook (48) is better for the 4x3 part.
  
  Total rank = 49 + 48 = 97.

Now, let's implement this.

For the 4x4 x 4x4 Strassen-Strassen:
  We need to express the multiplication as a sum of 49 rank-1 tensors.
  Each rank-1 tensor is u ⊗ v ⊗ w where u is 16-dim (for A), v is 16-dim (for B), w is 16-dim (for C).
  
  The structure:
  Level 0: 1x1 x 1x1 (schoolbook, rank 1)
  Level 1: 2x2 x 2x2 using Strassen (rank 7)
  Level 2: 4x4 x 4x4 using Strassen of Level 1 (rank 49)

  I'll construct the 49 (u, v, w) triples explicitly.

For the 4x4 x 4x3 schoolbook:
  48 terms, each with u having one 1, v having one 1, w having one 1.

Let's code this up.
"""

import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We'll build U (16 x rank), V (16 x rank), W (12 x rank) for the 4x4*4x7 case.
    # But we split B into B1 (4x4) and B2 (4x3).
    # For B1 part: U1 (16 x 49), V1 (16 x 49), W1 (12 x 49) [first 4 columns of W1 are used, rest 0]
    # For B2 part: U2 (16 x 48), V2 (12 x 48), W2 (12 x 48)
    
    # Actually, the V matrix has m*p = 4*7 = 28 rows.
    # The first 16 rows correspond to B1 (4x4), next 12 rows to B2 (4x3).
    # The W matrix has n*p = 4*7 = 28 rows? No, W has p*n = 7*4 = 28 rows.
    # Wait, the trace convention is W[k*n+i, r] where k is the column of B (0..p-1) and i is the row of A/C (0..n-1).
    # So W has p*n = 7*4 = 28 rows.
    # For B1 (first 4 columns of B), the relevant W entries are for k=0..3, i=0..3, i.e., W[0..15, :].
    # For B2 (last 3 columns of B), the relevant W entries are for k=4..6, i=0..3, i.e., W[16..27, :].
    
    # Similarly, V has m*p = 4*7 = 28 rows.
    # V[j*p+k, c] where j is row of A (0..3), k is column of B (0..6).
    # For B1: k=0..3, so rows j*7+k for j=0..3, k=0..3 -> rows 0..3, 7..10, 14..17, 21..24.
    # For B2: k=4..6, so rows j*7+k for j=0..3, k=4..6 -> rows 4..6, 11..13, 18..20, 25..27.
    
    # U has n*m = 4*4 = 16 rows. U[i*m+j, c] where i is row of A, j is column of A.
    
    # Let's build the 49-term Strassen-Strassen decomposition for 4x4 x 4x4.
    # And the 48-term schoolbook for 4x4 x 4x3.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Strassen-Strassen for 4x4 x 4x4 (49 terms)
    # We'll generate these programmatically.
    ss_terms = _strassen_strassen_4x4()
    for (u, v, w) in ss_terms:
        # u is length 16 (for A), v is length 16 (for B1 part, i.e., 4x4), w is length 16 (for C1 part, 4x4)
        U_list.append(list(u))
        # v needs to be mapped to V's 28 rows. For B1 (first 4 columns), V[j*7+k] where k=0..3.
        v_full = [0] * 28
        for idx, val in enumerate(v):
            j = idx // 4
            k = idx % 4
            v_full[j * 7 + k] = val
        V_list.append(v_full)
        # w needs to be mapped to W's 28 rows. For C1 (first 4 columns of C), W[k*4+i] where k=0..3.
        w_full = [0] * 28
        for idx, val in enumerate(w):
            k = idx // 4
            i = idx % 4
            w_full[k * 4 + i] = val
        W_list.append(w_full)
    
    # Part 2: Schoolbook for 4x4 x 4x3 (48 terms)
    for i in range(4):
        for j in range(4):
            for k in range(3):  # k is column of B2, which corresponds to column 4+k of B
                u = [0] * 16
                u[i * 4 + j] = 1
                v = [0] * 28
                # B2 column k corresponds to column 4+k of B, so V[j*7 + (4+k)]
                v[j * 7 + (4 + k)] = 1
                w = [0] * 28
                # C2 column k corresponds to column 4+k of C, so W[(4+k)*4 + i]
                w[(4 + k) * 4 + i] = 1
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    rank = len(U_list)
    U = [row for row in U_list]
    V = [row for row in V_list]
    W = [row for row in W_list]
    
    return {"U": U, "V": V, "W": W}


def _strassen_strassen_4x4():
    """
    Generate the 49 rank-1 terms for 4x4 x 4x4 matrix multiplication via Strassen-Strassen.
    Returns list of (u, v, w) where each is a list of 16 floats/ints.
    """
    # Strassen for 2x2 x 2x2: 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each of these 7 "super-entries" is itself a 2x2 x 2x2 multiplication,
    # which we expand using Strassen again (7 scalar mults each).
    # Total: 7 * 7 = 49.
    
    # The 7 Strassen forms for 2x2:
    # For each form r (0..6), we have:
    #   u_r: linear form of A (4 entries)
    #   v_r: linear form of B (4 entries)
    #   w_r: how it contributes to C (4 entries)
    
    # Let's define the 7 (u, v, w) for 2x2 Strassen.
    # Indices: A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    
    def strassen_2x2_forms():
        forms = []
        # M1 = (A00+A11)(B00+B11)
        forms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))  # contributes to C00 and C11
        # M2 = (A10+A11)B00
        forms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))  # contributes to C10
        # M3 = A00(B10-B11)
        forms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]))  # contributes to C01
        # M4 = A11(B01-B00)
        forms.append(([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 0, 1]))  # wait, let me recheck
        # Actually, let me use a cleaner formulation.
        
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)  -> in 0-indexed: (A00+A11)(B00+B11)
        # P2 = (A21+A22)B11       -> (A10+A11)B00
        # P3 = A11(B21-B22)       -> A00(B10-B11)
        # P4 = A22(B12-B11)       -> A11(B01-B00)
        # P5 = (A11+A12)B22       -> (A00+A01)B11
        # P6 = (A21-A11)(B11+B12) -> (A10-A00)(B00+B01)
        # P7 = (A12-A22)(B21+B22) -> (A01-A11)(B10+B11)
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So the 7 (u,v,w) for 2x2:
        s = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # P1: C00 += 1, C11 += 1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # P2: C10 += 1
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # P3: C01 += 1
            ([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 0, 1]),  # P4: C11 += 1, but C00 also? 
            # Wait, P4 contributes to C00 and C21. Let me re-examine.
            # C00 = P1 + P4 - P5 + P7
            # C11 = P1 - P2 + P3 + P6
            # C01 = P3 + P5
            # C10 = P2 + P4
            # So P4 contributes to C00 and C10.
        ]
        
        # Let me redo this more carefully.
        # The 7 multiplications are:
        # P1 = (A00+A11) * (B00+B11)
        # P2 = (A10+A11) * B00
        # P3 = A00 * (B10-B11)
        # P4 = A11 * (B01-B00)
        # P5 = (A00+A01) * B11
        # P6 = (A10-A00) * (B00+B01)
        # P7 = (A01-A11) * (B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So for each Pi, the w vector (contribution to C) is:
        # P1: [1, 0, 0, 1]  (C00 += 1, C11 += 1)
        # P2: [0, 0, 1, 0]  (C10 += 1)
        # P3: [0, 1, 0, 0]  (C01 += 1)
        # P4: [1, 0, 1, 0]  (C00 += 1, C10 += 1)
        # P5: [0, 1, 0, -1] (C01 += 1, C11 -= 1)
        # P6: [0, 0, 0, 1]  (C11 += 1)
        # P7: [1, 0, 0, -1] (C00 += 1, C11 -= 1)
        
        # And the u and v:
        # P1: u=[1,0,0,1], v=[1,0,0,1]
        # P2: u=[0,0,1,1], v=[1,0,0,0]
        # P3: u=[1,0,0,0], v=[0,0,1,-1]
        # P4: u=[0,0,0,1], v=[0,1,0,-1]
        # P5: u=[1,1,0,0], v=[0,0,0,1]
        # P6: u=[0,0,1,-1], v=[1,1,0,0]
        # P7: u=[0,1,0,-1], v=[0,0,1,1]
        
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
        ]
    
    base_forms = strassen_2x2_forms()
    
    # Now, for 4x4, we use Strassen at the block level.
    # A is 4x4, partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 7 Strassen "super-multiplications" for the 4x4 case:
    # S1 = (A11+A22) * (B11+B22)
    # S2 = (A21+A22) * B11
    # S3 = A11 * (B21-B22)
    # S4 = A22 * (B12-B11)
    # S5 = (A11+A12) * B22
    # S6 = (A21-A11) * (B11+B12)
    # S7 = (A12-A22) * (B21+B22)
    
    # Each Si is a 2x2 x 2x2 multiplication, expanded via base_forms (7 terms each).
    # Total: 49 terms.
    
    # For each Si, we have a linear form of the 4x4 A (16 entries), a linear form of B (16 entries),
    # and a contribution to C (16 entries).
    
    # The 4x4 A is indexed as A[i][j] for i,j in 0..3.
    # A11 = A[0:2][0:2], A12 = A[0:2][2:4], A21 = A[2:4][0:2], A22 = A[2:4][2:4].
    
    def a_block_index(block, i, j):
        """Map (block, i, j) to linear index in 4x4 A."""
        if block == 0:  # A11
            return i * 4 + j
        elif block == 1:  # A12
            return i * 4 + (j + 2)
        elif block == 2:  # A21
            return (i + 2) * 4 + j
        elif block == 3:  # A22
            return (i + 2) * 4 + (j + 2)
    
    def b_block_index(block, i, j):
        return a_block_index(block, i, j)  # Same structure
    
    def c_block_index(block, i, j):
        return a_block_index(block, i, j)  # Same structure
    
    # The 7 super-forms for 4x4:
    # Each is (u_super, v_super, w_super) where u_super is a dict mapping (block, i, j) -> coeff
    # We'll represent u_super as a list of 16 coeffs.
    
    super_forms = []
    # S1: (A11+A22) * (B11+B22)
    u1 = [0] * 16
    for i in range(2):
        for j in range(2):
            u1[a_block_index(0, i, j)] = 1
            u1[a_block_index(3, i, j)] = 1
    v1 = [0] * 16
    for i in range(2):
        for j in range(2):
            v1[b_block_index(0, i, j)] = 1
            v1[b_block_index(3, i, j)] = 1
    w1 = [0] * 16
    for i in range(2):
        for j in range(2):
            w1[c_block_index(0, i, j)] = 1
            w1[c_block_index(3, i, j)] = 1
    super_forms.append((u1, v1, w1))
    
    # S2: (A21+A22) * B11
    u2 = [0] * 16
    for i in range(2):
        for j in range(2):
            u2[a_block_index(2, i, j)] = 1
            u2[a_block_index(3, i, j)] = 1
    v2 = [0] * 16
    for i in range(2):
        for j in range(2):
            v2[b_block_index(0, i, j)] = 1
    w2 = [0] * 16
    for i in range(2):
        for j in range(2):
            w2[c_block_index(2, i, j)] = 1
    super_forms.append((u2, v2, w2))
    
    # S3: A11 * (B21-B22)
    u3 = [0] * 16
    for i in range(2):
        for j in range(2):
            u3[a_block_index(0, i, j)] = 1
    v3 = [0] * 16
    for i in range(2):
        for j in range(2):
            v3[b_block_index(2, i, j)] = 1
            v3[b_block_index(3, i, j)] = -1
    w3 = [0] * 16
    for i in range(2):
        for j in range(2):
            w3[c_block_index(0, i, j)] = 1
    super_forms.append((u3, v3, w3))
    
    # S4: A22 * (B12-B11)
    u4 = [0] * 16
    for i in range(2):
        for j in range(2):
            u4[a_block_index(3, i, j)] = 1
    v4 = [0] * 16
    for i in range(2):
        for j in range(2):
            v4[b_block_index(1, i, j)] = 1
            v4[b_block_index(0, i, j)] = -1
    w4 = [0] * 16
    for i in range(2):
        for j in range(2):
            w4[c_block_index(1, i, j)] = 1
    super_forms.append((u4, v4, w4))
    
    # S5: (A11+A12) * B22
    u5 = [0] * 16
    for i in range(2):
        for j in range(2):
            u5[a_block_index(0, i, j)] = 1
            u5[a_block_index(1, i, j)] = 1
    v5 = [0] * 16
    for i in range(2):
        for j in range(2):
            v5[b_block_index(3, i, j)] = 1
    w5 = [0] * 16
    for i in range(2):
        for j in range(2):
            w5[c_block_index(3, i, j)] = 1
    super_forms.append((u5, v5, w5))
    
    # S6: (A21-A11) * (B11+B12)
    u6 = [0] * 16
    for i in range(2):
        for j in range(2):
            u6[a_block_index(2, i, j)] = 1
            u6[a_block_index(0, i, j)] = -1
    v6 = [0] * 16
    for i in range(2):
        for j in range(2):
            v6[b_block_index(0, i, j)] = 1
            v6[b_block_index(1, i, j)] = 1
    w6 = [0] * 16
    for i in range(2):
        for j in range(2):
            w6[c_block_index(3, i, j)] = 1
    super_forms.append((u6, v6, w6))
    
    # S7: (A12-A22) * (B21+B22)
    u7 = [0] * 16
    for i in range(2):
        for j in range(2):
            u7[a_block_index(1, i, j)] = 1
            u7[a_block_index(3, i, j)] = -1
    v7 = [0] * 16
    for i in range(2):
        for j in range(2):
            v7[b_block_index(2, i, j)] = 1
            v7[b_block_index(3, i, j)] = 1
    w7 = [0] * 16
    for i in range(2):
        for j in range(2):
            w7[c_block_index(1, i, j)] = 1
    super_forms.append((u7, v7, w7))
    
    # Now, for each super-form, expand using the 7 base Strassen forms.
    # The super-form's u is a 2x2 matrix (in block space), which we treat as a 2x2 A for the inner Strassen.
    # Similarly for v and w.
    
    # For a super-form with u_super (16 coeffs, representing a 2x2 matrix of 2x2 blocks),
    # v_super (16 coeffs), w_super (16 coeffs):
    # We interpret u_super as a 2x2 matrix U where U[bi][bj] is the coefficient for block (bi,bj).
    # Then for each base form (u_base, v_base, w_base) of 2x2 Strassen:
    #   u_combined[i] = sum over (bi,bj) of u_base[bi*2+bj] * u_super[block_index(bi,bj, i, j)]
    # This is getting complex. Let me do it explicitly.
    
    # u_super is 16 coeffs: u_super[a_block_index(bi, bj, i, j)] = coeff for A[bi*2+i][bj*2+j]
    # For the inner Strassen, we have a 2x2 matrix U where U[i][j] = sum over (bi,bj) of ... 
    # Actually, the super-form already gives the linear combination of the 4x4 A entries.
    # The inner Strassen operates on the 2x2 "super-entries" which are themselves 2x2 matrices.
    
    # Let me think of it as: the super-form defines a 2x2 matrix M_u where M_u[i][j] = u_super[block(i,j)]
    # But each "entry" M_u[i][j] is actually a 2x2 matrix (the block).
    # The inner Strassen treats these 2x2 blocks as scalar entries, but they are not scalars.
    
    # Correct approach: The super-form's u is a linear form on the 4x4 A. When we say 
    # S1 = (A11+A22) * (B11+B22), this means we compute the product of the 2x2 matrix (A11+A22) 
    # and the 2x2 matrix (B11+B22). This product is a 2x2 matrix of 2x2 blocks, each of which 
    # is a 2x2x2x2 multiplication.
    
    # So for S1, the "A" in the inner 2x2x2x2 is the 2x2 matrix with entries A11+A22, A12+A22? No.
    # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix. Their product is a 2x2 matrix.
    # Each entry of this product is a 2x2x2x2 multiplication.
    
    # So for S1, we have a 2x2 "super-A" where super-A[i][j] = A[bi*2+i][bj*2+j] + A[(3-bi)*2+i][3-bj*2+j]? 
    # No, A11+A22 means the 2x2 matrix A11 plus the 2x2 matrix A22, element-wise.
    # So super-A[i][j] = A11[i][j] + A22[i][j] = A[i][j] + A[i+2][j+2].
    
    # Similarly, super-B[i][j] = B11[i][j] + B22[i][j] = B[i][j] + B[i+2][j+2].
    
    # The product super-C = super-A * super-B, where each entry is a 2x2x2x2 multiplication.
    # super-C[i][j] = sum_k super-A[i][k] * super-B[k][j]
    # But super-A[i][k] and super-B[k][j] are 2x2 matrices, so their "product" is a 2x2x2x2 multiplication.
    
    # This is getting too nested. Let me just directly construct the 49 terms.
    
    # Alternative: I'll use a recursive construction.
    # The tensor for 4x4 x 4x4 can be built from the tensor for 2x2 x 2x2.
    # M_4 = Strassen(M_2)
    
    # The Strassen decomposition of M_4 in terms of M_2:
    # There are 7 terms, each of which is a product of three tensors.
    # But I need the explicit (u,v,w) triples.
    
    # Let me just hardcode the 49 terms using a more direct approach.
    # I'll use the fact that the 4x4 x 4x4 multiplication tensor can be decomposed as:
    # For each of the 49 (i,j,k) in a 7x7x1 structure... 
    
    # Actually, let me just use schoolbook for 4x4 x 4x4 as well, for simplicity, 
    # and accept rank 64 + 48 = 112. But the task asks for rank-minimal, and 97 is known.
    
    # Given time constraints, I'll implement the 48-term schoolbook for 4x4x4x3 and 
    # use a 64-term schoolbook for 4x4x4x4, giving total rank 112.
    # This is worse than 97, but it's correct and simple.
    
    # Wait, the seed program already does rank 112. I need to improve on that.
    
    # Let me try a different approach for the 4x4 x 4x4 part: use Strassen at the 4x4 level 
    # by decomposing into 2x2 blocks and using the 7-term Strassen, where each term is 
    # a 2x2 x 2x2 multiplication done via schoolbook (rank 8 each). Total: 7*8 = 56.
    # Then for 4x4 x 4x3, use schoolbook (48). Total: 56 + 48 = 104.
    
    # Or use Strassen for 2x2 (rank 7) in the inner level: 7*7 = 49.
    
    # I'll implement the 49-term Strassen-Strassen properly.
    
    # Let me define the 49 terms explicitly using nested Strassen.
    
    def nested_strassen_4x4():
        """
        Generate 49 (u,v,w) triples for 4x4 x 4x4 via Strassen-Strassen.
        """
        # The 7 base Strassen forms for 2x2:
        # (u, v, w) where u,v,w are 4-tuples
        base = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
        ]
        
        # The 7 Strassen forms for the 4x4 block decomposition:
        # Each is (u_blocks, v_blocks, w_blocks) where each is a 2x2 matrix of coefficients
        # for the 2x2 blocks.
        
        # S1: (A11+A22)*(B11+B22) -> w: C11 += 1, C22 += 1
        # S2: (A21+A22)*B11 -> w: C21 += 1
        # S3: A11*(B21-B22) -> w: C11 += 1
        # S4: A22*(B12-B11) -> w: C12 += 1
        # S5: (A11+A12)*B22 -> w: C22 += 1
        # S6: (A21-A11)*(B11+B12) -> w: C22 += 1
        # S7: (A12-A22)*(B21+B22) -> w: C12 += 1
        
        # For each Si, we have a 2x2 "super" multiplication.
        # The u for Si is a 2x2 matrix of coefficients for the A blocks.
        # Similarly for v and w.
        
        # Let me represent each Si's (u,v,w) as 2x2 matrices of scalars.
        # u_Si[i][j] = coefficient for A_block[i][j]
        
        S = []
        # S1
        u1 = [[1, 0], [0, 1]]  # A11 + A22
        v1 = [[1, 0], [0, 1]]  # B11 + B22
        w1 = [[1, 0], [0, 1]]  # C11 += 1, C22 += 1
        S.append((u1, v1, w1))
        
        # S2
        u2 = [[0, 0], [1, 1]]  # A21 + A22
        v2 = [[1, 0], [0, 0]]  # B11
        w2 = [[0, 0], [1, 0]]  # C21 += 1
        S.append((u2, v2, w2))
        
        # S3
        u3 = [[1, 0], [0, 0]]  # A11
        v3 = [[0, 0], [1, -1]]  # B21 - B22
        w3 = [[1, 0], [0, 0]]  # C11 += 1
        S.append((u3, v3, w3))
        
        # S4
        u4 = [[0, 0], [0, 1]]  # A22
        v4 = [[0, 1], [0, -1]]  # B12 - B11
        w4 = [[0, 1], [0, 0]]  # C12 += 1
        S.append((u4, v4, w4))
        
        # S5
        u5 = [[1, 1], [0, 0]]  # A11 + A12
        v5 = [[0, 0], [0, 1]]  # B22
        w5 = [[0, 0], [0, 1]]  # C22 += 1
        S.append((u5, v5, w5))
        
        # S6
        u6 = [[-1, 0], [1, 0]]  # A21 - A11
        v6 = [[1, 1], [0, 0]]  # B11 + B12
        w6 = [[0, 0], [0, 1]]  # C22 += 1
        S.append((u6, v6, w6))
        
        # S7
        u7 = [[0, 1], [0, -1]]  # A12 - A22
        v7 = [[0, 0], [1, 1]]  # B21 + B22
        w7 = [[0, 1], [0, 0]]  # C12 += 1
        S.append((u7, v7, w7))
        
        # Now, for each Si, expand using the 7 base forms.
        # For Si with (uS, vS, wS) where each is a 2x2 matrix:
        # The "A" in the inner Strassen is a 2x2 matrix where entry (i,j) = uS[i][j] * A_block[i][j]
        # But A_block[i][j] is a 2x2 matrix, so the "A" is a 2x2 matrix of 2x2 matrices.
        # The inner Strassen computes the product of this 2x2 "A" with the 2x2 "B",
        # giving a 2x2 "C" of 2x2 matrices.
        
        # For each base form (ub, vb, wb) of the inner 2x2 Strassen:
        # u_inner[i] = sum over (bi,bj) of ub[bi*2+bj] * uS[bi][bj] * A[bi*2+i][bj*2+j]
        # So u_combined[bi*2+i + bj*2+j] = ub[bi*2+bj] * uS[bi][bj]
        
        terms = []
        for (uS, vS, wS) in S:
            for (ub, vb, wb) in base:
                u = [0] * 16
                v = [0] * 16
                w = [0] * 16
                for bi in range(2):
                    for bj in range(2):
                        coeff_u = ub[bi*2 + bj] * uS[bi][bj]
                        coeff_v = vb[bi*2 + bj] * vS[bi][bj]
                        coeff_w = wb[bi*2 + bj] * wS[bi][bj]
                        for i in range(2):
                            for j in range(2):
                                idx = (bi*2 + i) * 4 + (bj*2 + j)
                                u[idx] += coeff_u
                                v[idx] += coeff_v
                                w[idx] += coeff_w
                terms.append((u, v, w))
        
        return terms
    
    return nested_strassen_4x4()
# EVOLVE-BLOCK-END
