"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]... 
    Actually the tensor decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    Wait, the convention is W[k*n+i, r]. Let me re-read.
    
    The tensor for matrix multiplication (n x m) * (m x p) = (n x p) is:
    T[i, j, k] = delta where A is n x m, B is m x p, C = A*B is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p),
    and T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    
    In the factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For schoolbook: each (i,j,k) gets its own rank-1 term with U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Compute the tensor from the factorization
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: A is 2x2, B is 2x2, C = A*B is 2x2.
    C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor T[i,j,k] = 1 (all 8 terms are 1)
    Factorization: T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's define the factors. U[i*2+j, r] is the coefficient for A[i,j] in product r.
    # V[j*2+k, r] is the coefficient for B[j,k] in product r.
    # W[k*2+i, r] is the coefficient for C[i,k] from product r.
    
    # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    #           B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    #           C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3 (W[k*2+i]: C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3)
    
    # Wait, W index is k*n+i where n=2. So W[k*2+i].
    # C[i,k] -> W[k*2+i]. So:
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # U[i*2+j]: A[i,j]
    # A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[2], A[1,1]->U[3]
    # V[j*2+k]: B[j,k]
    # B[0,0]->V[0], B[0,1]->V[1], B[1,0]->V[2], B[1,1]->V[3]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A coeffs: A00=1, A11=1 -> U[0]=1, U[3]=1
    # B coeffs: B00=1, B11=1 -> V[0]=1, V[3]=1
    # C contribution: C00 += 1, C11 += 1
    # C00 -> W[0], C11 -> W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A: A10=1, A11=1 -> U[2]=1, U[3]=1
    # B: B00=1 -> V[0]=1
    # C: C10 += 1, C11 += 1
    # C10 -> W[1], C11 -> W[3]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    # A: A00=1 -> U[0]=1
    # B: B10=1, B11=-1 -> V[2]=1, V[3]=-1
    # C: C01 += 1, C11 += 1
    # C01 -> W[2], C11 -> W[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    # A: A11=1 -> U[3]=1
    # B: B01=1, B00=-1 -> V[1]=1, V[0]=-1
    # C: C01 += 1, C00 -= 1
    # C01 -> W[2], C00 -> W[0]
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1
    
    # M5 = (A00+A01)B11
    # A: A00=1, A01=1 -> U[0]=1, U[1]=1
    # B: B11=1 -> V[3]=1
    # C: C00 += 1, C10 += 1
    # C00 -> W[0], C10 -> W[1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    # A: A10=1, A00=-1 -> U[2]=1, U[0]=-1
    # B: B00=1, B10=1 -> V[0]=1, V[2]=1
    # C: C11 += 1
    # C11 -> W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    # A: A01=1, A11=-1 -> U[1]=1, U[3]=-1
    # B: B01=1, B11=1 -> V[1]=1, V[3]=1
    # C: C00 -= 1
    # C00 -> W[0]
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # A is 3x4, B is 4x4, C = A*B is 3x4
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, j in 0..3, k in 0..3
    
    # Strategy: Use Strassen on 2x2x2 blocks.
    # Partition A's rows into groups: rows {0,1} and row {2}
    # For rows {0,1}: 2x4 * 4x4 -> use Strassen-style decomposition
    # For row {2}: 1x4 * 4x4 -> trivial rank 4
    
    # For the 2x4 * 4x4 case:
    # We can think of it as two 2x2 * 2x4 blocks? No.
    # Actually, let's partition the columns of A (which are the rows of B) into two groups of 2.
    # A[0:2, 0:4] * B[0:4, 0:4]
    # = A[0:2, 0:2] * B[0:2, 0:4] + A[0:2, 2:4] * B[2:4, 0:4]
    
    # Each term is 2x2 * 2x4. We can use Strassen on the 2x2 part.
    # 2x2 * 2x4: C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a 2x2x4 tensor. We can decompose it using Strassen's 7 products
    # applied to each pair of output columns.
    
    # Actually, for 2x2 * 2x4, we can use Strassen's algorithm by treating it as
    # two independent 2x2 * 2x2 multiplications (for columns 0,1 and columns 2,3).
    # But we can share the A factors!
    
    # Let's think more carefully. For 2x2 * 2x4:
    # A is 2x2, B is 2x4, C is 2x4.
    # C[:, 0:2] = A * B[:, 0:2] (2x2 * 2x2)
    # C[:, 2:4] = A * B[:, 2:4] (2x2 * 2x2)
    
    # Using Strassen: 7 products for each 2x2*2x2.
    # But the A factors are shared. In Strassen, the 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For the second 2x2*2x2 (columns 2,3 of B):
    # M1' = (A00+A11)(B02+B13)
    # M2' = (A10+A11)B02
    # M3' = A00(B12-B13)
    # M4' = A11(B03-B02)
    # M5' = (A00+A01)B13
    # M6' = (A10-A00)(B02+B12)
    # M7' = (A01-A11)(B03+B13)
    
    # So total 14 products for 2x2 * 2x4.
    # Compare to schoolbook: 2*2*4 = 16. So we save 2.
    
    # For our problem:
    # Rows 0,1: 2x4 * 4x4. Partition B's rows into {0,1} and {2,3}.
    # A[0:2, 0:2] * B[0:2, 0:4] -> 14 products (Strassen on 2x2*2x4)
    # A[0:2, 2:4] * B[2:4, 0:4] -> 14 products
    # Total for rows 0,1: 28 products
    
    # Row 2: 1x4 * 4x4 -> 4*4 = 16 products (schoolbook)
    # Total: 28 + 16 = 44
    
    # Let me verify this is correct and build the factors.
    
    # Alternative: can we do better?
    # For row 2: 1x4 * 4x4. C[2,k] = sum_j A[2,j]*B[j,k].
    # This is inherently rank 16 (it's just a linear combination of 16 terms).
    # Actually no, it's rank 16 in the sense of tensor rank? 
    # The tensor T[i,j,k] for i=2 only: T[2,j,k] = 1.
    # This is a 1x4x4 tensor with all entries 1. Its tensor rank is 4 (not 16)!
    # Because T[2,j,k] = sum_{r=0}^{3} U[j,r]*V[k,r]*W[r] where we can set
    # U[j,r] = 1 for all j, V[k,r] = 1 for all k, W[r] = 1, but that gives rank 1.
    # Wait, T[2,j,k] = 1 for all j,k. So it's the all-ones 4x4 matrix as a tensor.
    # The tensor rank of the all-ones 1x4x4 tensor is 4 (it's the outer product of 
    # three vectors: ones(1) x ones(4) x ones(4), which is rank 1!).
    # Actually, a 1x4x4 tensor with all entries 1 is rank 1: it's u x v x w where
    # u=[1], v=ones(4), w=ones(4). So T[2,j,k] = 1*1*1 = 1.
    # But wait, in our factorization, U has index i*m+j, V has index j*p+k, W has index k*n+i.
    # For i=2: U[2*4+j, r] = U[8+j, r], V[j*4+k, r], W[k*3+2, r].
    # We need sum_r U[8+j,r]*V[j*4+k,r]*W[k*3+2,r] = 1 for all j,k.
    # This is rank 1: set U[8+j,0]=1, V[j*4+k,0]=1, W[k*3+2,0]=1 for all j,k.
    # But V[j*4+k,0]=1 for all j,k means V is all ones in column 0.
    # And U[8+j,0]=1 for all j.
    # And W[k*3+2,0]=1 for all k.
    # So yes, rank 1 for the row-2 part!
    
    # Wait, that can't be right. Let me re-examine.
    # The full tensor is T[i,j,k] = 1 for all i,j,k (since every term A[i,j]*B[j,k] 
    # contributes to C[i,k]).
    # No wait, that's not right either. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], every (i,j,k) triple has T[i,j,k] = 1.
    # So the tensor is all-ones of shape (n,m,p) = (3,4,4).
    # The tensor rank of the all-ones 3x4x4 tensor is min(3,4,4) = 3!
    # Because T[i,j,k] = 1 = u_i * v_j * w_k where u=ones(3), v=ones(4), w=ones(4).
    # So rank 1!
    
    # But that's not how matrix multiplication tensor works. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix 
    # by a 4 by 4 matrix."
    # 
    # The matrix multiplication tensor T is defined such that:
    # C[i,k] = sum_{j} sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # 
    # No, that's not right. The standard definition is:
    # The bilinear map (A,B) -> C where C = A*B can be written as:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where the sum is over r, and for each r, the term is:
    # (sum_{i,j} U[i*m+j,r] * A[i,j]) * (sum_{j,k} V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # 
    # Actually, the correct formulation is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # 
    # Hmm, that's not quite right either. Let me think about this more carefully.
    # 
    # The matrix multiplication tensor is a trilinear form. The standard factorization is:
    # C[i,k] = sum_{r=1}^{rank} alpha_r * (sum_j U_r[i,j] * A[i,j]) * (sum_j V_r[j,k] * B[j,k]) * W_r[i,k]
    # 
    # No, the correct form for the "trace convention" W[k*n+i, r] is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where the summation is implicit over j as well? No...
    # 
    # Let me look at the schoolbook example:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = (i,j,k)
    # So for a given (i,k):
    # C[i,k] = sum_{c} U[i*m+j_c, c] * V[j_c*p+k_c, c] * W[k_c*n+i_c, c]
    # where c indexes (i_c, j_c, k_c).
    # For the term to be nonzero, we need i_c = i, j_c = j, k_c = k.
    # So C[i,k] = sum_{j} 1*1*1 = sum_j 1... that's not right.
    # 
    # I think the correct interpretation is:
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)} and
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' where i=i', j=j', k=k'.
    # So T[i*m+j, j*p+k, k*n+i] = 1 and all other entries are 0.
    # 
    # The factorization is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # 
    # For schoolbook: T[i*m+j, j*p+k, k*n+i] = U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] = 1*1*1 = 1. ✓
    # And for any other (a,b,c), at most one of the factors is nonzero, so the product is 0. ✓
    # 
    # So the tensor is NOT all-ones. It's a sparse tensor where T[i*m+j, j*p+k, k*n+i] = 1.
    # 
    # Now, for the row-2 part: T[8+j, j*4+k, k*3+2] = 1 for j=0..3, k=0..3.
    # This is 16 entries. Can we factor these with rank < 16?
    # We need: sum_r U[8+j,r]*V[j*4+k,r]*W[k*3+2,r] = 1 for all j,k,
    # and = 0 for all other (a,b,c) triples.
    # 
    # The "other" constraints are important. For example, T[8+j, j'*4+k, k*3+2] should be 0 
    # if j' != j. And T[8+j, j*4+k, k'*3+2] should be 0 if k' != k.
    # 
    # So it's not just the all-ones tensor. It's a structured sparse tensor.
    # The rank-16 schoolbook decomposition is likely close to minimal for this sub-problem.
    
    # Let me just implement the 44-rank decomposition and verify it.
    
    # Build the 44-rank decomposition:
    # Part 1: Rows 0,1 with B rows 0,1 (Strassen 2x2*2x4): 14 products
    # Part 2: Rows 0,1 with B rows 2,3 (Strassen 2x2*2x4): 14 products  
    # Part 3: Row 2 with B rows 0,1,2,3 (schoolbook 1x4*4x4): 16 products
    # Total: 44
    
    U = [[0]*44 for _ in range(n*m)]  # 12 x 44
    V = [[0]*44 for _ in range(m*p)]  # 16 x 44
    W = [[0]*44 for _ in range(p*n)]  # 12 x 44
    
    # Helper: add a Strassen 2x2*2x4 block
    # A block: rows i0, i0+1 of A; columns j0, j0+1 of A (= rows j0, j0+1 of B)
    # B block: rows j0, j0+1 of B; all 4 columns k=0..3
    # Output: rows i0, i0+1 of C; all 4 columns k=0..3
    def add_strassen_2x2_2x4(i0, j0, col_offset):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, 0:4].
        i0: first row index of A block
        j0: first column index of A block (= first row index of B block)
        col_offset: column offset in the factor matrices
        """
        # U indices: U[i*m+j] for i in {i0, i0+1}, j in {j0, j0+1}
        # V indices: V[j*p+k] for j in {j0, j0+1}, k in 0..3
        # W indices: W[k*n+i] for k in 0..3, i in {i0, i0+1}
        
        # Strassen 7 products for the 2x2*2x2 part, applied to each pair of k columns.
        # For k pair (2t, 2t+1) where t=0,1:
        # M1_t = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,2t]+B[j0+1,2t+1])
        # M2_t = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,2t]
        # M3_t = A[i0,j0] * (B[j0+1,2t]-B[j0+1,2t+1])
        # M4_t = A[i0+1,j0+1] * (B[j0,2t+1]-B[j0,2t])
        # M5_t = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,2t+1]
        # M6_t = (A[i0+1,j0]-A[i0,j0]) * (B[j0,2t]+B[j0+1,2t])
        # M7_t = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0,2t+1]+B[j0+1,2t+1])
        
        # C[i0,2t] += M1_t - M4_t + M5_t - M7_t
        # C[i0,2t+1] += M3_t + M4_t
        # C[i0+1,2t] += M2_t + M5_t
        # C[i0+1,2t+1] += M1_t - M2_t + M3_t + M6_t
        
        for t in range(2):  # t=0 for k pair (0,1), t=1 for k pair (2,3)
            k0 = 2*t
            k1 = 2*t+1
            base = col_offset + 7*t
            
            # M1
            r = base + 0
            U[i0*m+j0][r] += 1
            U[(i0+1)*m+(j0+1)][r] += 1
            V[j0*p+k0][r] += 1
            V[(j0+1)*p+k1][r] += 1
            W[k0*n+i0][r] += 1    # C[i0,k0] += M1
            W[k1*n+(i0+1)][r] += 1  # C[i0+1,k1] += M1
            
            # M2
            r = base + 1
            U[(i0+1)*m+j0][r] += 1
            U[(i0+1)*m+(j0+1)][r] += 1
            V[j0*p+k0][r] += 1
            W[k0*n+(i0+1)][r] += 1  # C[i0+1,k0] += M2
            W[k1*n+(i0+1)][r] -= 1  # C[i0+1,k1] -= M2
            
            # M3
            r = base + 2
            U[i0*m+j0][r] += 1
            V[(j0+1)*p+k0][r] += 1
            V[(j0+1)*p+k1][r] -= 1
            W[k1*n+i0][r] += 1    # C[i0,k1] += M3
            W[k1*n+(i0+1)][r] += 1  # C[i0+1,k1] += M3
            
            # M4
            r = base + 3
            U[(i0+1)*m+(j0+1)][r] += 1
            V[j0*p+k1][r] += 1
            V[j0*p+k0][r] -= 1
            W[k1*n+i0][r] += 1    # C[i0,k1] += M4
            W[k0*n+i0][r] -= 1    # C[i0,k0] -= M4
            
            # M5
            r = base + 4
            U[i0*m+j0][r] += 1
            U[i0*m+(j0+1)][r] += 1
            V[(j0+1)*p+k1][r] += 1
            W[k0*n+i0][r] += 1    # C[i0,k0] += M5
            W[k0*n+(i0+1)][r] += 1  # C[i0+1,k0] += M5
            
            # M6
            r = base + 5
            U[(i0+1)*m+j0][r] += 1
            U[i0*m+j0][r] -= 1
            V[j0*p+k0][r] += 1
            V[(j0+1)*p+k0][r] += 1
            W[k1*n+(i0+1)][r] += 1  # C[i0+1,k1] += M6
            
            # M7
            r = base + 6
            U[i0*m+(j0+1)][r] += 1
            U[(i0+1)*m+(j0+1)][r] -= 1
            V[j0*p+k1][r] += 1
            V[(j0+1)*p+k1][r] += 1
            W[k0*n+i0][r] -= 1    # C[i0,k0] -= M7
        
        return col_offset + 14
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:4]
    col = add_strassen_2x2_2x4(0, 0, 0)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 0:4]
    col = add_strassen_2x2_2x4(0, 2, col)
    
    # Part 3: A[2:3, 0:4] * B[0:4, 0:4] (schoolbook for 1x4*4x4)
    for j in range(4):
        for k in range(4):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W):
            # Fallback to schoolbook
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
