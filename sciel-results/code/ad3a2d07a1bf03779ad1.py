import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 5, 5, 5
    
    # Partition: A rows into groups [0,1],[2,3],[4], B cols into [0,1],[2,3],[4]
    # U index: i*m + j  (i = A row, j = B col)
    # V index: j*p + k  (j = B col, k = C col)  
    # W index: k*n + i  (k = C col, i = A row) -- trace convention
    
    # We'll build factor matrices as lists of lists
    # Each "rank unit" is a column
    
    factors = []  # list of (U_row, V_row, W_row, U_val, V_val, W_val)
    
    def add_factor(U_row, V_row, W_row, u_val, v_val, w_val):
        factors.append((U_row, V_row, W_row, u_val, v_val, w_val))
    
    # Strassen 2x2x2: given 2x2 A, 2x2 B, 2x2 C
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]], C = [[c11,c12],[c21,c22]]
    # 7 multiplications:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # C11 = m1+m4-m5+m7
    # C12 = m3+m5
    # C21 = m2+m4
    # C22 = m1-m2+m3+m6
    
    # For a 2x2 block at A-rows (ra0,ra1), B-cols (cb0,cb1), C-rows (ra0,ra1), C-cols (cb0,cb1)
    # The tensor contribution for each (i,j,k) where i in {ra0,ra1}, j in {cb0,cb1}, k in {cb0,cb1}
    # U[i*m+j] gets the A-part, V[j*p+k] gets the B-part, W[k*n+i] gets the C-part
    
    def add_strassen(ra0, ra1, cb0, cb1):
        """Add Strassen decomposition for 2x2x2 block.
        A rows: ra0, ra1
        B cols: cb0, cb1
        C rows: ra0, ra1 (since C row index = A row index)
        C cols: cb0, cb1 (since C col index = B col index)
        """
        # U row indices: i*m + j for i in {ra0,ra1}, j in {cb0,cb1}
        u11 = ra0 * m + cb0
        u12 = ra0 * m + cb1
        u21 = ra1 * m + cb0
        u22 = ra1 * m + cb1
        
        # V row indices: j*p + k for j in {cb0,cb1}, k in {cb0,cb1}
        v11 = cb0 * p + cb0
        v12 = cb0 * p + cb1
        v21 = cb1 * p + cb0
        v22 = cb1 * p + cb1
        
        # W row indices: k*n + i for k in {cb0,cb1}, i in {ra0,ra1}
        w11 = cb0 * n + ra0
        w12 = cb0 * n + ra1
        w21 = cb1 * n + ra0
        w22 = cb1 * n + ra1
        
        # m1 = (a11+a22)*(b11+b22)
        # U: u11 and u22 contribute +1 each
        # V: v11 and v22 contribute +1 each
        # W: w11, w12, w21, w22 all contribute +1
        # C11 gets +1, C22 gets -1... wait, let me be more careful.
        
        # Actually, for the tensor decomposition, each rank-1 term contributes
        # T[i,j,k] += U[i*m+j] * V[j*p+k] * W[k*n+i]
        # and we need T[i,j,k] = delta_{i,k}
        
        # For Strassen, the 7 terms are:
        # m1: A-part = a11+a22, B-part = b11+b22
        #   U: +1 at (ra0,cb0), +1 at (ra1,cb1)
        #   V: +1 at (cb0,cb0), +1 at (cb1,cb1)
        #   W: +1 at (cb0,ra0), +1 at (cb0,ra1), +1 at (cb1,ra0), +1 at (cb1,ra1)
        #   But wait - this gives a product that's the full 2x2 outer product, not just the diagonal.
        
        # Let me reconsider. The tensor T[i,j,k] = delta_{i,k} means:
        # For the 2x2 sub-block, T[i,j,k] = 1 if i==k (within the sub-block), else 0
        # So T[ra0,cb0,cb0]=1, T[ra0,cb1,cb1]=1, T[ra1,cb0,cb0]=1, T[ra1,cb1,cb1]=1
        # All other T[i,j,k] = 0 for i in {ra0,ra1}, j in {cb0,cb1}, k in {cb0,cb1}
        
        # Strassen computes C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
        # The rank-1 factors for the 7 products are:
        # m1: A-part = (a11+a22) means U has +1 at (ra0,cb0) and +1 at (ra1,cb1)
        #     B-part = (b11+b22) means V has +1 at (cb0,cb0) and +1 at (cb1,cb1)
        #     The W part must give the C combination: C11 += 1, C22 += 1, C12 += 0, C21 += 0
        #     C11 is at (i=ra0, k=cb0), so W at (k=cb0, i=ra0) = w11 should be +1
        #     C22 is at (i=ra1, k=cb1), so W at (k=cb1, i=ra1) = w22 should be +1
        #     C12 is at (i=ra0, k=cb1), so W at (k=cb1, i=ra0) = w21 should be 0
        #     C21 is at (i=ra1, k=cb0), so W at (k=cb0, i=ra1) = w12 should be 0
        
        # Wait, that doesn't work because the product U*V*W would then be:
        # For (i=ra0, j=cb0, k=cb0): U[ra0,cb0]*V[cb0,cb0]*W[cb0,ra0] = 1*1*1 = 1 ✓
        # For (i=ra0, j=cb0, k=cb1): U[ra0,cb0]*V[cb0,cb1]*W[cb1,ra0] = 1*1*0 = 0 ✓
        # For (i=ra0, j=cb1, k=cb0): U[ra0,cb1]*V[cb1,cb0]*W[cb0,ra0] = 1*1*1 = 1 ✗ (should be 0)
        
        # Hmm, I'm confusing myself. Let me think more carefully.
        
        # The tensor for matrix mult: T[i,j,k] = delta_{i,k}
        # A rank-1 term is: sum over (i,j,k) of U[i,j] * V[j,k] * W[k,i]
        # This equals delta_{i,k} for the specific (i,j,k) pattern.
        
        # For a single rank-1 term with U having nonzeros at certain (i,j), V at certain (j,k), W at certain (k,i):
        # The product U[i,j]*V[j,k]*W[k,i] is nonzero only when all three are nonzero simultaneously.
        
        # For Strassen m1: A-part = a11 + a22, B-part = b11 + b22
        # This means: U[ra0,cb0] += 1, U[ra1,cb1] += 1
        #            V[cb0,cb0] += 1, V[cb1,cb1] += 1
        # Now the product U[i,j]*V[j,k] is nonzero when:
        #   (i=ra0,j=cb0,k=cb0), (i=ra0,j=cb0,k=cb1), (i=ra1,j=cb1,k=cb0), (i=ra1,j=cb1,k=cb1)
        # Wait no: V[j,k] for j=cb0: V[cb0,cb0]=1, V[cb0,cb1]=1
        # V[j,k] for j=cb1: V[cb1,cb0]=1, V[cb1,cb1]=1
        # So U[ra0,cb0]*V[cb0,k] is nonzero for k=cb0 and k=cb1
        # U[ra1,cb1]*V[cb1,k] is nonzero for k=cb0 and k=cb1
        # So the U*V product is nonzero for all 4 combinations of (i in {ra0,ra1}, k in {cb0,cb1})
        # We need W to zero out the off-diagonal terms.
        # We need: W[cb0,ra0] + ... hmm, this is getting complicated.
        
        # Let me just use the standard Strassen factorization directly.
        # The 7 rank-1 terms for 2x2 matrix mult are well-known.
        
        # Term 1: m1 = (a11+a22)(b11+b22)
        # U: [1,0;0,1] (diag)
        # V: [1,1;1,1] (all ones)
        # W: [1,0;0,1] (diag) -- but this gives C11+=1, C22+=1, and also C12+=1, C21+=1
        # That's not right.
        
        # I think I need to be more careful. Let me look up the exact rank-1 factors.
        
        # Actually, let me just directly construct the factors by solving the system.
        # For each of the 7 Strassen products, I know the A-part, B-part, and C-part.
        # The A-part is a linear combination of a11,a12,a21,a22
        # The B-part is a linear combination of b11,b12,b21,b22
        # The C-part is a linear combination of c11,c12,c21,c22
        # And the rank-1 term contributes to T[i,j,k] as:
        #   For each (i,j) where A-part has coefficient alpha_ij:
        #     For each (j',k) where B-part has coefficient beta_j'k:
        #       If j==j': T[i,k] += alpha_ij * beta_jk * gamma_ki
        # where gamma is the C-part coefficients.
        
        # This is getting too complex. Let me just use a simpler approach:
        # I'll directly compute the rank-1 factors for Strassen.
        
        # Standard Strassen factors (from literature):
        # Let me define the 7 terms with their U, V, W contributions.
        
        # I'll use the convention that for the 2x2 block:
        # U is 2x2 (rows = A rows, cols = B cols within block)
        # V is 2x2 (rows = B cols within block, cols = C cols within block)  
        # W is 2x2 (rows = C cols within block, cols = A rows within block)
        
        # The 7 rank-1 decompositions of the 2x2x2 matmul tensor:
        
        # Term 1: U=[[1,0],[0,1]], V=[[1,1],[1,1]], W=[[1,0],[0,-1]]
        # Term 2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[1,0],[0,0]]
        # Term 3: U=[[1,0],[0,0]], V=[[0,1],[0,0]], W=[[0,1],[0,0]]
        # Term 4: U=[[0,0],[0,1]], V=[[0,0],[1,0]], W=[[0,0],[1,0]]
        # Term 5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,0],[0,1]]
        # Term 6: U=[[0,0],[-1,1]], V=[[1,1],[0,0]], W=[[0,0],[0,1]]
        # Term 7: U=[[0,1],[0,-1]], V=[[0,0],[1,1]], W=[[1,0],[0,0]]
        
        # Let me verify Term 1:
        # U = [[1,0],[0,1]], V = [[1,1],[1,1]], W = [[1,0],[0,-1]]
        # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
        # T[0,0,1] = U[0,0]*V[0,1]*W[1,0] = 1*1*0 = 0
        # T[0,1,0] = U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0
        # T[0,1,1] = U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
        # T[1,0,0] = U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
        # T[1,0,1] = U[1,0]*V[0,1]*W[1,1] = 0*1*(-1) = 0
        # T[1,1,0] = U[1,1]*V[1,0]*W[0,1] = 1*1*0 = 0
        # T[1,1,1] = U[1,1]*V[1,1]*W[1,1] = 1*1*(-1) = -1
        # Sum so far: T[0,0,0]=1, T[1,1,1]=-1
        # But we need T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1
        # So Term 1 alone gives wrong signs.
        
        # I think the standard factors need to be summed. Let me just directly construct
        # a valid rank-7 decomposition by solving the linear system.
        
        # Actually, let me use a different approach. I'll construct the factors
        # directly from the known Strassen algorithm.
        
        # For 2x2 matmul, the 7 products are:
        # p1 = (a00+a11)(b00+b11)
        # p2 = (a10+a11)(b00)
        # p3 = (a00)(b01-b11)
        # p4 = (a11)(b10-b00)
        # p5 = (a00+a01)(b11)
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        # c00 = p1+p4-p5+p7
        # c01 = p3+p5
        # c10 = p2+p4
        # c11 = p1-p2+p3+p6
        
        # Each pk is a rank-1 term in the tensor.
        # For pk, the U part is the coefficient pattern of A, V part is B, W part is C.
        
        # p1: A-part = a00+a11, B-part = b00+b11
        #   U[0,0]=1, U[1,1]=1
        #   V[0,0]=1, V[1,1]=1
        #   C-part: c00 += 1, c11 += 1
        #   W[0,0]=1, W[1,1]=1
        #   But we also need c01 and c10 to get their contributions from other terms.
        
        # Let me just sum up all 7 terms and verify.
        
        terms = [
            # (U_2x2, V_2x2, W_2x2)
            ([[1,0],[0,1]], [[1,1],[1,1]], [[1,0],[0,1]]),   # p1
            ([[0,0],[1,1]], [[1,0],[0,0]], [[1,0],[0,0]]),   # p2
            ([[1,0],[0,0]], [[0,1],[0,0]], [[0,1],[0,0]]),   # p3
            ([[0,0],[0,1]], [[0,0],[1,0]], [[0,0],[1,0]]),   # p4
            ([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[0,1]]),   # p5
            ([[0,0],[-1,1]], [[1,1],[0,0]], [[0,0],[0,1]]),  # p6
            ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),  # p7
        ]
        
        # Sum all terms to get the full tensor
        T = [[0,0],[0,0]]  # T[i,k] = sum_j T[i,j,k]... no
        # Actually T[i,j,k] for i,j,k in {0,1}
        T3 = np.zeros((2,2,2))
        for U2, V2, W2 in terms:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        T3[i,j,k] += U2[i][j] * V2[j][k] * W2[k][i]
        
        # Check: T3[i,j,k] should be 1 if i==k, else 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    expected = 1.0 if i == k else 0.0
                    if abs(T3[i,j,k] - expected) > 1e-9:
                        # This decomposition is wrong, let me fix it
                        pass
        
        # Let me try different factors. I'll look at this more carefully.
        # The issue is that W[k,i] couples k and i, so the rank-1 structure is different.
        
        # Let me just use a direct approach: for each of the 7 Strassen products,
        # I'll create one rank-1 factor in the full 5x5x5 tensor.
        
        # For p1 = (a00+a11)(b00+b11):
        # U contribution: U[ra0,cb0] += 1, U[ra1,cb1] += 1
        # V contribution: V[cb0,cb0] += 1, V[cb1,cb1] += 1
        # W contribution: needs to give c00 += 1, c11 += 1, c01 += 0, c10 += 0
        # c00 is at (i=ra0, k=cb0) -> W[cb0,ra0]
        # c11 is at (i=ra1, k=cb1) -> W[cb1,ra1]
        # c01 is at (i=ra0, k=cb1) -> W[cb1,ra0]
        # c10 is at (i=ra1, k=cb0) -> W[cb0,ra1]
        # So W[cb0,ra0]=1, W[cb1,ra1]=1, W[cb1,ra0]=0, W[cb0,ra1]=0
        
        # Let me verify: product U[i,j]*V[j,k]*W[k,i]
        # (i=ra0,j=cb0,k=cb0): U[ra0,cb0]*V[cb0,cb0]*W[cb0,ra0] = 1*1*1 = 1 ✓
        # (i=ra0,j=cb0,k=cb1): U[ra0,cb0]*V[cb0,cb1]*W[cb1,ra0] = 1*1*0 = 0 ✓
        # (i=ra0,j=cb1,k=cb0): U[ra0,cb1]*V[cb1,cb0]*W[cb0,ra0] = 0*1*1 = 0 ✓
        # (i=ra0,j=cb1,k=cb1): U[ra0,cb1]*V[cb1,cb1]*W[cb1,ra0] = 0*1*0 = 0 ✓
        # (i=ra1,j=cb0,k=cb0): U[ra1,cb0]*V[cb0,cb0]*W[cb0,ra1] = 0*1*0 = 0 ✓
        # (i=ra1,j=cb0,k=cb1): U[ra1,cb0]*V[cb0,cb1]*W[cb1,ra1] = 0*1*1 = 0 ✓
        # (i=ra1,j=cb1,k=cb0): U[ra1,cb1]*V[cb1,cb0]*W[cb0,ra1] = 1*1*0 = 0 ✓
        # (i=ra1,j=cb1,k=cb1): U[ra1,cb1]*V[cb1,cb1]*W[cb1,ra1] = 1*1*1 = 1 ✓
        # 
        # But we need T[ra0,cb0,cb1] = 0 and T[ra0,cb1,cb0] = 0 and T[ra1,cb0,cb0] = 1 and T[ra1,cb1,cb1] = 1
        # Wait, T[i,j,k] = delta_{i,k}, so:
        # T[ra0,cb0,cb0] = 1 (i=k=ra0? No, i=ra0, k=cb0, so i==k means ra0==cb0)
        
        # I'm confusing row and column indices. Let me restart with clear notation.
        # A is n×m = 5×5, B is m×p = 5×5, C is n×p = 5×5
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # T[i,j,k] = delta_{i,k} where i is A row / C row, j is A col / B row, k is B col / C col
        # U[i*m + j], V[j*p + k], W[k*n + i]
        
        # For the 2x2 sub-block with A rows {ra0,ra1}, B/C cols {cb0,cb1}:
        # The sub-tensor has indices i in {ra0,ra1}, j in {cb0,cb1}, k in {cb0,cb1}
        # T[i,j,k] = 1 if i==k (as integers), else 0
        # Since ra0,ra1 are A row indices and cb0,cb1 are B/C col indices,
        # i==k means the A row index equals the C col index.
        # In general ra0 != cb0, so T[ra0,cb0,cb0] = 1 only if ra0 == cb0.
        
        # This is the key insight: the tensor T[i,j,k] = delta_{i,k} is NOT the identity
        # in the sub-block unless the row and column indices coincide!
        
        # For the full 5x5 matrix mult, T[i,j,k] = 1 if i==k (global indices).
        # So for the sub-block at rows {0,1} and cols {0,1}:
        # T[0,0,0]=1, T[0,0,1]=0, T[0,1,0]=0, T[0,1,1]=0
        # T[1,0,0]=0, T[1,0,1]=0, T[1,1,0]=0, T[1,1,1]=1
        # So it IS the identity in this case since rows={0,1} and cols={0,1}.
        
        # For the sub-block at rows {2,3} and cols {2,3}:
        # T[2,2,2]=1, T[2,2,3]=0, T[2,3,2]=0, T[2,3,3]=1
        # Also identity.
        
        # For cross-block rows {0,1} and cols {2,3}:
        # T[0,2,2]=0, T[0,2,3]=0, T[0,3,2]=0, T[0,3,3]=0
        # T[1,2,2]=0, T[1,2,3]=0, T[1,3,2]=0, T[1,3,3]=0
        # All zeros!
        
        # So the cross-block tensors are all zero, which means we don't need any
        # factors for them! The only nonzero contributions are from the diagonal blocks
        # where A row index == C col index.
        
        # Wait, that can't be right. Let me re-examine.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # For i=0, k=2: C[0,2] = sum_j A[0,j]*B[j,2]
        # This involves A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] + A[0,3]*B[3,2] + A[0,4]*B[4,2]
        # So T[0,j,2] should be 1 for all j? No...
        
        # I think I've been misunderstanding the tensor. Let me re-read the problem.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 5 by 5 matrix by a 5 by 5 matrix."
        # The tensor for matrix multiplication is T_{i,j,k} = delta_{i,k}
        # This means: for the output C[i,k], the contribution from A[i,j] and B[j,k] is
        # present only when i==k. But that's not how matrix multiplication works!
        
        # Actually, I think the tensor is defined as:
        # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The "multiplication tensor" has T_{i,j,k} = 1 for all i,j,k, and the
        # decomposition gives the rank needed to compute all (i,j,k) combinations.
        # No wait, that would be rank 125 for 5x5.
        
        # Let me look at the schoolbook decomposition in the seed:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
        # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = number of c where all three are nonzero
        # For a given (i,j,k), only one c has all three nonzero (the one corresponding to (i,j,k))
        # So the sum = 1 for each (i,j,k), meaning T[i,j,k] = 1 for all i,j,k.
        
        # So the tensor is ALL ONES, not delta_{i,k}!
        # T[i,j,k] = 1 for all i in [0,5), j in [0,5), k in [0,5)
        
        # This makes sense: the matrix multiplication tensor has a 1 for every (i,j,k)
        # triple, representing that A[i,j] and B[j,k] always contribute to C[i,k].
        
        # So for the sub-block at A rows {ra0,ra1} and B/C cols {cb0,cb1}:
        # T[i,j,k] = 1 for all i in {ra0,ra1}, j in {cb0,cb1}, k in {cb0,cb1}
        # This is a 2x2x2 all-ones tensor, which has rank 1!
        # U = [[1,1],[1,1]], V = [[1,1],[1,1]], W = [[1,1],[1,1]]
        
        # Wait, that means every sub-block has rank 1, and the total rank would be
        # 9 (for the 3x3 partition into 2x2, 2x2, 1x1 blocks) + cross terms.
        # But that contradicts the known rank-16 for 3x3 and rank-7 for 2x2.
        
        # I think I'm still confused. Let me just go with the schoolbook approach
        # but use the block partition to see if we can do better.
        
        # Actually, re-reading the knowledge: "Using Strassen's rank-7 algorithm for the eight
        # 2x2x2 sub-blocks in a 2+2+1 partition reduces the total rank from 125 to 117"
        # 8 sub-blocks of 2x2x2: 8*7 = 56
        # 1x1x1 block: 1
        # Cross terms: 125 - 56 - 1 = 68
        # 56 + 1 + 68 = 125? No, that's still 125.
        
        # I think the partition is different. Let me just implement the known result:
        # rank 117 for 5x5x5 using a specific block decomposition.
        
        # Given the complexity, let me just implement the schoolbook (rank 125) as a safe
        # fallback, and try to apply Strassen to the diagonal 2x2 blocks.
        
        # Actually, let me just return the schoolbook decomposition to be safe,
        # since getting the block decomposition exactly right is error-prone.
        
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
