import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin

    # Build a decomposition using block structure:
    # Split m=4 into two blocks of 2: m1=2, m2=2
    # For each block (i in {0,1}), we need to multiply A_block (n x 2) by B_block (2 x p)
    # The tensor for block i: T_i[a, b, c] where a in [0,n), b in [0,2), c in [0,p)
    # T_i[a, b, c] = 1 if a==c, else 0  (since it's just selecting column c of A_block row... wait)
    # Actually for matrix mult C = A * B, C[i,k] = sum_j A[i,j] B[j,k]
    # For block 0 (j=0,1): C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For block 1 (j=2,3): C[i,k] += A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # So each block is a 4x2x7 sub-tensor: multiply (4x2) by (2x7)
    # The 4x2x7 tensor: T[i, j, k] = delta_{i,k} for i in [0,4), j in [0,2), k in [0,7)
    # Wait, that's not right either. Let me reconsider.
    
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # Block 0: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # The sub-tensor for block 0 has indices (i, j_local, k) where j_local in {0,1}
    # T_0[i, j_local, k] = 1 if the term A[i, j_local] * B[j_local, k] contributes to C[i,k]
    # Which is always 1 (it's a linear combination). So T_0[i, j_local, k] = 1 for all i, j_local, k?
    # No! The tensor entry T[i,j,k] = 1 means that the product of the i-th row of A, j-th column... 
    # Actually the matrix multiplication tensor is: M[i,j,k] = delta_{j, j'} * delta_{k,k'}
    # where the three indices are (row_of_C, col_of_A_row_index, col_of_B_col_index)
    # More precisely: M[a, b, c] = 1 if a==c (the output index a matches the B column index c)
    # and b is the shared index. So M[a,b,c] = delta_{a,c}.
    # Wait no. For C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has T[i,j,k] = 1 if this term exists, i.e., T[i,j,k] = 1 for all valid i,j,k.
    # But that's the schoolbook. The tensor is actually:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not a tensor decomposition. The tensor itself is:
    # For each (i,j,k), the coefficient is 1. The rank is the minimum r such that
    # T[i,j,k] = sum_{r=0}^{R-1} U[i,j,r] V[j,k,r] W[k,i,r]
    # where W uses the trace convention W[k*n+i, r].
    
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the "right" sense).
    # Actually no. Let me re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k] = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k]
    # This is always true, so T[i,j,k] = 1 for all i,j,k.
    # But wait, that would make it a trivial rank-1 tensor? No...
    # 
    # I think the tensor is indexed differently. Let me re-read:
    # "W[k*n+i,r]" - this suggests W is indexed by (k, i) pairs where k is the B column and i is the A row/C row.
    # The tensor equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = delta_{j,j'} * delta_{k,k'} ... 
    # 
    # Actually, I think T[i,j,k] = 1 for all i,j,k. The schoolbook decomposition has rank n*m*p
    # with U[i*m+j, i*m*p+j*p+k] = 1, V[j*p+k, i*m*p+j*p+k] = 1, W[k*n+i, i*m*p+j*p+k] = 1.
    # And sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if (i,j,k) matches the term, 0 otherwise.
    # But for matrix multiplication, EVERY term (i,j,k) contributes! So T[i,j,k] = 1 for all i,j,k.
    # 
    # Hmm, but then the rank would be... let me think again.
    # If T[i,j,k] = 1 for all i,j,k, then it's the all-ones tensor which has rank 1!
    # That can't be right.
    #
    # I must be misunderstanding the tensor. Let me look at the check function more carefully:
    # "T[i*m+j, j'*p+k, k'*n+i] = delta_{j,j'} * delta_{k,k'}"
    # So the tensor has three indices: (a, b, c) where a = i*m+j, b = j'*p+k, c = k'*n+i
    # And T[a,b,c] = delta_{j,j'} * delta_{k,k'}
    # So T[i*m+j, j'*p+k, k'*n+i] = 1 if j==j' and k==k', else 0.
    #
    # This means: the tensor is 1 when the "A column index" (j from a) matches the "B row index" (j' from b)
    # AND the "B column index" (k from b) matches the "C row index... wait, k' from c" 
    # Hmm, this is confusing. Let me just work with the check function's definition.
    #
    # The decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i, r] = 1 if j==j2 and k==k2, else 0
    #
    # So for fixed i, the sub-tensor over (j, j2, k, k2) has the structure:
    # S_i[j, j2, k, k2] = delta_{j,j2} * delta_{k,k2}
    #
    # This is equivalent to saying that for each i, we have a 4x4x7x7 tensor that factors as
    # (identity in j,j2) * (identity in k,k2).
    #
    # Now, the key insight for block decomposition:
    # If we split j into two blocks of 2, then for block 0 (j in {0,1}):
    # S_i[j, j2, k, k2] = delta_{j,j2} * delta_{k,k2} for j,j2 in {0,1}
    # This is a 2x2x7x7 tensor = (I_2) ⊗ (I_7)
    #
    # The rank of I_2 ⊗ I_7 as a 4-index tensor... but we need to think of it as a 3-index tensor
    # after fixing i. Actually, for each i, we need a decomposition of the tensor
    # T_i[j, j2, k, k2] = delta_{j,j2} * delta_{k,k2}
    # into rank-r tensors where each rank-1 term is u_j * v_{j2,k} * w_{k2,i}... 
    # 
    # Wait, I'm overcomplicating this. Let me just use the Strassen approach directly.
    #
    # For the 2x2x2 core (n=2, m=2, p=2), Strassen gives rank 7.
    # For our problem, we can think of it as: for each pair of output rows (i, i+2) and 
    # each block of input columns, we have a 2x2x7 sub-problem.
    #
    # Actually, let me try a different structural approach. Since the tensor is:
    # sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i, r] = delta_{j,j2} * delta_{k,k2}
    #
    # This means: for each i, the "slice" over (j,j2,k,k2) must equal delta_{j,j2}*delta_{k,k2}.
    # 
    # A key observation: if we can find a decomposition where U, V, W have a block structure
    # in j (the m-dimension), we can exploit it.
    #
    # Let me try: use Strassen for the 2x2x2 part and schoolbook for the rest.
    # Split m=4 into m1=2, m2=2. Split n=4 into n1=2, n2=2.
    # For each (n_block, m_block) pair, we have a 2x2x7 sub-problem.
    #
    # The 2x2x7 tensor: T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,6}
    # T[i,j,k] = delta_{j,j'} * delta_{k,k'} ... but this is a 3-index tensor with the constraint.
    #
    # Actually, I realize the tensor for 2x2x7 is:
    # sum_r U[2*2+j, r] * V[j'*7+k, r] * W[k'*2+i, r] = delta_{j,j'} * delta_{k,k'}
    #
    # For the 2x2x2 part (k in {0,1}), this is the standard Strassen tensor with rank 7.
    # For the remaining 2x2x5 part (k in {2,6}), we need rank 10 (schoolbook) or less.
    #
    # Total: 4 blocks * (7 + 10) = 4 * 17 = 68? That seems too optimistic.
    # Wait, the blocks share indices. Let me think more carefully.
    #
    # Let me just implement a structured search: start with Strassen-like patterns for the 2x2 cores
    # and optimize the remaining entries.

    # Helper: check decomposition
    def check_decomp(U, V, W):
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Check all non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check zero entries: j != j2
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2:
                        continue
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
                        if abs(val) > 1e-9:
                            return False
        # Check zero entries: k != k2
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for k2 in range(p):
                        if k == k2:
                            continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k2, :] * W[k*n+i, :])
                        if abs(val) > 1e-9:
                            return False
        return True

    # Strassen 2x2x2 terms (rank 7)
    # For 2x2 matrices: C = A * B where A,B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # Strassen:
    # p1 = (a11+a22)*(b11+b22)
    # p2 = (a21+a22)*b11
    # p3 = a11*(b12-b22)
    # p4 = a22*(b21-b11)
    # p5 = (a11+a12)*b22
    # p6 = (a21-a11)*(b11+b12)
    # p7 = (a12-a22)*(b21+b22)
    # c11 = p1+p4-p5+p7
    # c12 = p3+p5
    # c21 = p2+p4
    # c22 = p1-p2+p3+p6

    # For our tensor with n=2, m=2, p=2:
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, r] for i,j in {0,1}
    # V[j*2+k, r] for j,k in {0,1}
    # W[k*2+i, r] for k,i in {0,1}
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4, 7)."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        # p1 = (a11+a22)*(b11+b22): U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
        U[0, 0] = 1; U[1, 0] = 1  # a11, a22
        V[0, 0] = 1; V[1, 0] = 1  # b11, b22
        W[0, 0] = 1; W[1, 0] = 1  # c11, c22 (k=0,i=0 and k=1,i=1)
        # Wait, W[k*2+i, r]. For c11: k=0, i=0 -> W[0,0]. For c22: k=1, i=1 -> W[3,0]? 
        # No: W[k*2+i] where k is B column, i is A row/C row.
        # c11 corresponds to output (i=0, k=0), so W[0*2+0, 0] = W[0, 0]
        # c22 corresponds to output (i=1, k=1), so W[1*2+1, 0] = W[3, 0]
        # c12 corresponds to output (i=0, k=1), so W[1*2+0, 0] = W[2, 0]
        # c21 corresponds to output (i=1, k=0), so W[0*2+1, 0] = W[1, 0]
        
        # Let me redo this carefully.
        # W index: k*n + i = k*2 + i
        # (i=0,k=0): W[0]
        # (i=0,k=1): W[2]
        # (i=1,k=0): W[1]
        # (i=1,k=1): W[3]
        
        # p1 = (a11+a22)*(b11+b22) contributes to c11 and c22
        # U[0*2+0, 0] = U[0,0] = 1 (a11)
        # U[1*2+1, 0] = U[3,0] = 1 (a22)
        # V[0*2+0, 0] = V[0,0] = 1 (b11)
        # V[1*2+1, 0] = V[3,0] = 1 (b22)
        # W[0*2+0, 0] = W[0,0] = 1 (c11)
        # W[1*2+1, 0] = W[3,0] = 1 (c22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p2 = (a21+a22)*b11 contributes to c21
        # U[1*2+0, 1] = U[2,1] = 1 (a21)
        # U[1*2+1, 1] = U[3,1] = 1 (a22)
        # V[0*2+0, 1] = V[0,1] = 1 (b11)
        # W[0*2+1, 1] = W[1,1] = 1 (c21)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # p3 = a11*(b12-b22) contributes to c12
        # U[0*2+0, 2] = U[0,2] = 1 (a11)
        # V[0*2+1, 2] = V[1,2] = 1 (b12)
        # V[1*2+1, 2] = V[3,2] = -1 (b22)
        # W[1*2+0, 2] = W[2,2] = 1 (c12)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # p4 = a22*(b21-b11) contributes to c11 and c21
        # U[1*2+1, 3] = U[3,3] = 1 (a22)
        # V[1*2+0, 3] = V[2,3] = 1 (b21)
        # V[0*2+0, 3] = V[0,3] = -1 (b11)
        # W[0*2+0, 3] = W[0,3] = 1 (c11)
        # W[0*2+1, 3] = W[1,3] = 1 (c21)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # p5 = (a11+a12)*b22 contributes to c11 and c12
        # U[0*2+0, 4] = U[0,4] = 1 (a11)
        # U[0*2+1, 4] = U[1,4] = 1 (a12)
        # V[1*2+1, 4] = V[3,4] = 1 (b22)
        # W[0*2+0, 4] = W[0,4] = -1 (c11, with negative sign)
        # W[1*2+0, 4] = W[2,4] = 1 (c12)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # p6 = (a21-a11)*(b11+b12) contributes to c22
        # U[1*2+0, 5] = U[2,5] = 1 (a21)
        # U[0*2+0, 5] = U[0,5] = -1 (a11)
        # V[0*2+0, 5] = V[0,5] = 1 (b11)
        # V[0*2+1, 5] = V[1,5] = 1 (b12)
        # W[1*2+1, 5] = W[3,5] = 1 (c22)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # p7 = (a12-a22)*(b21+b22) contributes to c11 and c22
        # U[0*2+1, 6] = U[1,6] = 1 (a12)
        # U[1*2+1, 6] = U[3,6] = -1 (a22)
        # V[1*2+0, 6] = V[2,6] = 1 (b21)
        # V[1*2+1, 6] = V[3,6] = 1 (b22)
        # W[0*2+0, 6] = W[0,6] = 1 (c11)
        # W[1*2+1, 6] = W[3,6] = 1 (c22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Now build the full 4x4x7 decomposition using 4 blocks of 2x2x7
    # Each block: n_block=2, m_block=2, p=7
    # For block (nb, mb) where nb in {0,1}, mb in {0,1}:
    #   i in {2*nb, 2*nb+1}, j in {2*mb, 2*mb+1}, k in {0,...,6}
    #   U[(2*nb + i_local)*4 + (2*mb + j_local), r]
    #   V[(2*mb + j_local)*7 + k, r]
    #   W[k*4 + (2*nb + i_local), r]
    #
    # For the 2x2x7 sub-problem, we can use Strassen for the first 2 k-values and schoolbook for the rest.
    # Actually, let's think about the 2x2x7 tensor more carefully.
    #
    # The tensor for 2x2x7: T[i,j,k] with i in {0,1}, j in {0,1}, k in {0,...,6}
    # Must satisfy: sum_r U[i*2+j, r] * V[j2*7+k, r] * W[k2*2+i, r] = delta_{j,j2} * delta_{k,k2}
    #
    # This is equivalent to: for each (i, j2, k2), the vector U_i,j2,k2 = [U[i*2+j, r] for j] 
    # dotted with V_{j2,k2} = [V[j2*7+k, r] for k] dotted with W_{k2,i} = [W[k2*2+i, r] for i]
    # Hmm, this is getting complicated. Let me just use a direct approach.
    #
    # For 2x2x7, the schoolbook rank is 2*2*7 = 28.
    # Using Strassen for the 2x2x2 part (rank 7) and schoolbook for 2x2x5 (rank 10), total = 17.
    # But we need to be careful about how they combine.
    #
    # Actually, for the 2x2x7 tensor, we can think of it as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # The tensor is: T[i,j,k] = 1 for all i,j,k (each term contributes).
    # Wait, no. The tensor equation is:
    # sum_r U[i*2+j, r] * V[j2*7+k, r] * W[k2*2+i, r] = delta_{j,j2} * delta_{k,k2}
    #
    # For the 2x2x2 part (k, k2 in {0,1}), this is the Strassen tensor.
    # For the 2x2x5 part (k, k2 in {2,...,6}), this is a 2x2x5 tensor.
    #
    # But these parts are NOT independent! The same U, V, W entries are used for both.
    # 
    # Let me try a different approach: use the known rank-23 decomposition for 4x4x4
    # and extend it to 4x4x7.
    #
    # Actually, let me just do a structured random search with a good initialization.
    # I'll initialize with a block-diagonal Strassen-like pattern and then refine.

    # Let me try: for each of the 4 (n_block, m_block) pairs, use a 2x2x7 decomposition.
    # For 2x2x7, I'll use: Strassen for k=0,1 (rank 7) + schoolbook for k=2..6 (rank 10) = rank 17 per block.
    # Total: 4 * 17 = 68. But this double-counts because the W matrix is shared across blocks.
    #
    # Actually, the blocks are independent in the U and V indices but share W.
    # Let me think about this differently.
    
    # Direct approach: build U, V, W with block structure and verify.
    
    # For now, let me just do a smarter random search with structured initialization.
    
    # Initialize with a pattern based on Strassen blocks
    def structured_init(rank):
        """Create U, V, W with Strassen-like block structure."""
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # For each (n_block, m_block) pair, use Strassen for the 2x2x2 core
        # and schoolbook for the remaining 5 k-values
        col = 0
        for nb in range(2):
            for mb in range(2):
                # Strassen 2x2x2 part (7 terms)
                sU, sV, sW = strassen_2x2x2()
                for r in range(7):
                    if col + r >= rank:
                        break
                    for i_loc in range(2):
                        for j_loc in range(2):
                            U[(nb*2 + i_loc)*m + (mb*2 + j_loc), col + r] = sU[i_loc*2 + j_loc, r]
                    for j_loc in range(2):
                        for k in range(2):
                            V[(mb*2 + j_loc)*p + k, col + r] = sV[j_loc*2 + k, r]
                    for k in range(2):
                        for i_loc in range(2):
                            W[k*n + (nb*2 + i_loc), col + r] = sW[k*2 + i_loc, r]
                col += 7
                
                # Schoolbook for k=2..6 (10 terms: 2 i * 1 j... no, 2 j * 5 k = 10)
                for j_loc in range(2):
                    for k in range(2, 7):
                        if col >= rank:
                            break
                        for i_loc in range(2):
                            U[(nb*2 + i_loc)*m + (mb*2 + j_loc), col] = 1
                        V[(mb*2 + j_loc)*p + k, col] = 1
                        W[k*n + (nb*2 + 0), col] = 1  # only for i_loc=0? No...
                        # Actually for schoolbook, each term is one (i, j, k) triple
                        # But here we're sharing j across i's. That won't work for schoolbook.
                        # Let me redo: schoolbook has separate terms for each (i, j, k)
                        pass
                # This approach is getting messy. Let me just use pure schoolbook for the k>=2 part.
                for i_loc in range(2):
                    for j_loc in range(2):
                        for k in range(2, 7):
                            if col >= rank:
                                break
                            U[(nb*2 + i_loc)*m + (mb*2 + j_loc), col] = 1
                            V[(mb*2 + j_loc)*p + k, col] = 1
                            W[k*n + (nb*2 + i_loc), col] = 1
                            col += 1
        
        return U, V, W

    # The structured init above gives rank = 4*(7 + 2*2*5) = 4*(7+20) = 108. Not great.
    # Let me just do a random search with a better strategy.
    
    # Key insight: the tensor is "separable" in the i-index. For each i, the sub-tensor
    # over (j, j2, k, k2) is delta_{j,j2} * delta_{k,k2}. This means we can potentially
    # use different decompositions for different i's, but they must be compatible.
    
    # Let me try: use rank-7 Strassen for each of the 4 (i_pair, j_pair) blocks for the
    # k=0,1 part, and then optimize the k=2..6 part.
    
    # For now, let me do a random search with half-integer entries, but with a smarter
    # initialization and more attempts.
    
    best_rank = n * m * p  # 112
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as initial best
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Random search with structured patterns
    np.random.seed(42)
    
    # Try various rank targets
    for target_rank in [95, 90, 85, 80, 75, 70]:
        if time.time() > deadline - 20:
            break
        
        max_attempts = 200
        attempts = 0
        while attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1
            
            # Use a mix of structured and random entries
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            W = np.zeros((p*n, target_rank), dtype=float)
            
            # First, try to set up Strassen-like patterns for the 2x2x2 cores
            col = 0
            for nb in range(2):
                for mb in range(2):
                    sU, sV, sW = strassen_2x2x2()
                    for r in range(7):
                        if col + r >= target_rank:
                            break
                        for i_loc in range(2):
                            for j_loc in range(2):
                                U[(nb*2 + i_loc)*m + (mb*2 + j_loc), col + r] = sU[i_loc*2 + j_loc, r]
                        for j_loc in range(2):
                            for k in range(2):
                                V[(mb*2 + j_loc)*p + k, col + r] = sV[j_loc*2 + k, r]
                        for k in range(2):
                            for i_loc in range(2):
                                W[k*n + (nb*2 + i_loc), col + r] = sW[k*2 + i_loc, r]
                    col += 7
            
            # Fill remaining columns with random half-integers
            for c in range(col, target_rank):
                U[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n*m)
                V[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m*p)
                W[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=p*n)
            
            if check_decomp(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank < rank_sb:
            break  # Found improvement, stop
    
    # Convert to list format
    def to_list(M):
        return [[float(M[i][j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
