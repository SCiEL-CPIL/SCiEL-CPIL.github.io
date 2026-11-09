"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    We use a structured approach:
    1. Start with Strassen-like decomposition for the 2x2 blocks (treating 3x3 and 3x6 appropriately)
    2. Try to reduce rank by exploiting the rectangular structure
    
    The key insight: for 2x3 * 3x6, we can view this as:
    - Split the 3 columns of A into 2+1 (or use other splits)
    - Split the 3 rows of B into 2+1
    - Use Strassen for the 2x2 * 2x2 core (rank 7 instead of 4)
    - Handle the remaining 1x2 * 2x6 and 2x1 * 1x6 parts
    
    Actually, let me think more carefully.
    
    A is 2x3, B is 3x6, C is 2x6.
    
    Let's partition:
    A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    B = [B1; B2] where B1 is 2x6, B2 is 1x6
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x6):
    We can use Strassen's algorithm concept. A1*B1 = [a11 a12; a21 a22] * [b1; b2] where b1, b2 are 1x6 rows.
    
    Using Strassen for 2x2 * 2x1 (then extended to 2x6):
    M1 = (a11+a22)*(b1+b2)
    M2 = (a21+a22)*b2
    M3 = a11*(b1-b2)
    M4 = a22*(b2-b1)
    M5 = (a11+a12)*b2
    M6 = (a21-a11)*(b1+b2)
    M7 = (a12-a22)*(b1+b2)
    
    Wait, standard Strassen for 2x2*2x2:
    C11 = M1 - M4
    C12 = M5 + M4  
    C21 = M2 - M3
    C22 = M1 - M2 + M3 + M5
    
    where:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B22
    M3 = A11(B11-B22)
    M4 = A22(B22-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B22)
    M7 = (A12-A22)(B11+B22)
    
    For 2x2 * 2x6, we can apply this row-wise. Each "element" of B is a row vector.
    
    So for A1*B1 where A1 is 2x2 and B1 is 2x6:
    Let B1 = [b1_row; b2_row] where b1_row, b2_row are 1x6.
    
    The 7 multiplications become 7 scalar-vector multiplications (each producing a 1x6 result).
    So A1*B1 has rank at most 7 (in terms of the tensor).
    
    For A2*B2 where A2 is 2x1 and B2 is 1x6:
    This is outer product: A2 * B2. Rank 2 (one for each row of the result, or we can think of it as 2 rank-1 terms).
    Actually, A2*B2 = [a1; a2] * [b2_row] = [a1*b2_row; a2*b2_row]. This has rank 2 (two independent rank-1 tensors).
    
    So total rank: 7 + 2 = 9? But wait, we need to be careful about the tensor structure.
    
    Actually, let me reconsider. The matrix multiplication tensor for n x m * m x p has rank at most n*m*p (schoolbook). We want to minimize this.
    
    For 2x3 * 3x6: schoolbook rank = 2*3*6 = 36.
    
    Using the partition above:
    - A1*B1: 2x2 * 2x6. The tensor rank for this subproblem... 
      If we use Strassen, we get 7 "multiplications" but each involves a scalar from A and a vector from B.
      In tensor terms, this contributes rank 7 to the overall decomposition? Not exactly.
      
    Let me think about this differently. The tensor T has indices (i, j, k) where i in {0,1}, j in {0,1,2}, k in {0,...,5}.
    T[i,j,k] = delta (it's the identity tensor for matrix multiplication).
    
    A rank-r decomposition means: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    (with appropriate index mapping)
    
    Actually, the convention is:
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (18, rank)  
    W has shape (p*n, rank) = (12, rank)
    
    And the decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... 
    
    Wait, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor is T[i,j,k] = 1 if we're computing the (i,k) entry using (j,j) intermediate.
    
    Actually the standard tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all valid i,j,k, and the multiplication is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The rank is the minimum r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c where:
    u_c[i,j] = U[i*m+j, c]
    v_c[j,k] = V[j*p+k, c]  
    w_c[k,i] = W[k*n+i, c]
    
    And T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    For the identity (matrix multiplication), T[i,j,k] = 1 for all i,j,k in range.
    
    So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,1], j in [0,2], k in [0,5].
    
    Schoolbook: one term per (i,j,k) triple, rank = 2*3*6 = 36.
    
    Now, can we do better? Let's try the Strassen approach more carefully.
    
    Partition j into {0,1} and {2}:
    A = [A[:,0:2] | A[:,2:3]]
    B = [B[0:2,:]; B[2:3,:]]
    
    C = A[:,0:2] * B[0:2,:] + A[:,2:3] * B[2:3,:]
    
    Let C1 = A[:,0:2] * B[0:2,:] (2x2 * 2x6 -> 2x6)
    Let C2 = A[:,2:3] * B[2:3,:] (2x1 * 1x6 -> 2x6)
    
    For C1: Using Strassen for 2x2 * 2x6:
    The 7 Strassen products each involve a linear combination of A entries times a linear combination of B rows.
    In the tensor decomposition, each Strassen "product" corresponds to one term in the sum.
    
    For C1, the 7 terms give rank 7 contribution.
    For C2: A[:,2:3] is 2x1, B[2:3,:] is 1x6.
    C2[i,k] = A[i,2] * B[2,k]
    This is an outer product. In tensor terms, we need:
    sum_c U[i*3+2, c] * V[2*6+k, c] * W[k*2+i, c] = A[i,2]*B[2,k] for the C2 part.
    
    This can be done with rank 2:
    Term 1: U[0*3+2, c1]=1, V[12+k, c1]=1 for all k, W[k*2+0, c1]=1 for all k, others 0.
    Term 2: U[1*3+2, c2]=1, V[12+k, c2]=1 for all k, W[k*2+1, c2]=1 for all k, others 0.
    
    Wait, that's not quite right. Let me be more careful.
    
    For C2: we need for each i in {0,1}, k in {0,...,5}:
    Contribution to T[i,2,k] from C2 terms = 1 (since T[i,2,k]=1 and C1 doesn't contribute to j=2).
    
    Actually no, T[i,j,k]=1 for ALL i,j,k. The decomposition must satisfy:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    So for j=2 (the third column of A / third row of B), we need:
    sum_c U[i*3+2,c] * V[2*6+k,c] * W[k*2+i,c] = 1 for all i in {0,1}, k in {0,...,5}.
    
    This is a 2x6 matrix of 1s. We can decompose this as sum of 2 rank-1 tensors:
    For i=0: U[2,c] * V[12+k,c] * W[2k,c] = 1
    For i=1: U[5,c] * V[12+k,c] * W[2k+1,c] = 1
    
    Using 2 terms:
    c_a: U[2,c_a]=1, V[12+k,c_a]=1 for all k, W[2k,c_a]=1 for all k, W[2k+1,c_a]=0
    c_b: U[5,c_b]=1, V[12+k,c_b]=1 for all k, W[2k,c_b]=0, W[2k+1,c_b]=1 for all k
    
    This gives rank 2 for the j=2 part.
    
    For j in {0,1} (the first two columns/rows), we use Strassen. The Strassen decomposition for 2x2*2x2 gives 7 terms. For 2x2*2x6, each term involves multiplying a scalar (linear combo of A entries) with a vector (linear combo of B rows). In the tensor, this means:
    
    For each Strassen term c:
    U[i*3+j, c] = coefficient of A[i,j] in the linear combo
    V[j*6+k, c] = coefficient of B[j,k] in the linear combo  
    W[k*2+i, c] = coefficient in the output
    
    The 7 Strassen terms give rank 7 for the j in {0,1} part.
    
    Total rank: 7 + 2 = 9.
    
    Let me verify this is correct by constructing it explicitly.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # We'll use rank 9: 7 from Strassen for j in {0,1}, 2 for j=2
    rank = 9
    
    U = np.zeros((n*m, rank), dtype=float)  # (6, 9)
    V = np.zeros((m*p, rank), dtype=float)  # (18, 9)
    W = np.zeros((p*n, rank), dtype=float)  # (12, 9)
    
    # Strassen for 2x2 * 2x6 (j in {0,1}, k in {0,...,5})
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B1 = B[0,:], B2 = B[1,:] (each 1x6)
    
    # M1 = (A11+A22)(B1+B2)
    # M2 = (A21+A22)B2
    # M3 = A11(B1-B2)
    # M4 = A22(B2-B1)
    # M5 = (A11+A12)B2
    # M6 = (A21-A11)(B1+B2)
    # M7 = (A12-A22)(B1+B2)
    
    # C11 = M1 - M4
    # C12 = M5 + M4
    # C21 = M2 - M3
    # C22 = M1 - M2 + M3 + M5
    
    # For each term, we set U, V, W accordingly.
    
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,:]+B[1,:])
    # U[0*3+0, 0] = 1 (A[0,0])
    # U[1*3+1, 0] = 1 (A[1,1])
    # V[0*6+k, 0] = 1 for all k (B[0,:])
    # V[1*6+k, 0] = 1 for all k (B[1,:])
    # W[k*2+0, 0] = 1 (C[0,k] gets +1)
    # W[k*2+1, 0] = 1 (C[1,k] gets +1)
    U[0, 0] = 1  # A[0,0]
    U[4, 0] = 1  # A[1,1]
    V[0:6, 0] = 1  # B[0,:]
    V[6:12, 0] = 1  # B[1,:]
    W[0:12:2, 0] = 1  # C[0,k] for k=0..5
    W[1:12:2, 0] = 1  # C[1,k] for k=0..5
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[1,:]
    U[3, 1] = 1  # A[1,0]
    U[4, 1] = 1  # A[1,1]
    V[6:12, 1] = 1  # B[1,:]
    W[1:12:2, 1] = 1  # C[1,k]
    
    # Term 2: M3 = A[0,0] * (B[0,:]-B[1,:])
    U[0, 2] = 1  # A[0,0]
    V[0:6, 2] = 1  # B[0,:]
    V[6:12, 2] = -1  # -B[1,:]
    W[0:12:2, 2] = 1  # C[0,k]
    W[1:12:2, 2] = 1  # C[1,k]
    
    # Term 3: M4 = A[1,1] * (B[1,:]-B[0,:])
    U[4, 3] = 1  # A[1,1]
    V[6:12, 3] = 1  # B[1,:]
    V[0:6, 3] = -1  # -B[0,:]
    W[0:12:2, 3] = 1  # C[0,k]
    W[1:12:2, 3] = 1  # C[1,k]
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,:]
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[6:12, 4] = 1  # B[1,:]
    W[0:12:2, 4] = 1  # C[0,k]
    W[1:12:2, 4] = 1  # C[1,k]
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,:]+B[1,:])
    U[3, 5] = 1  # A[1,0]
    U[0, 5] = -1  # -A[0,0]
    V[0:6, 5] = 1  # B[0,:]
    V[6:12, 5] = 1  # B[1,:]
    W[0:12:2, 5] = 1  # C[0,k]
    W[1:12:2, 5] = 1  # C[1,k]
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[0,:]+B[1,:])
    U[1, 6] = 1  # A[0,1]
    U[4, 6] = -1  # -A[1,1]
    V[0:6, 6] = 1  # B[0,:]
    V[6:12, 6] = 1  # B[1,:]
    W[0:12:2, 6] = 1  # C[0,k]
    W[1:12:2, 6] = 1  # C[1,k]
    
    # Now set the W coefficients to implement the Strassen combination:
    # C[0,k] = M1 - M4
    # C[1,k] = M5 + M4  (wait, standard Strassen: C12 = M5+M4, C21=M2-M3, C22=M1-M2+M3+M5)
    
    # Let me redo this properly. Standard Strassen:
    # C11 = M1 - M4
    # C12 = M5 + M4
    # C21 = M2 - M3
    # C22 = M1 - M2 + M3 + M5
    
    # So:
    # C[0,k] (i=0) = C11 if k corresponds to col 0, C12 if k corresponds to col 1
    # But we have 6 columns, so we need to extend.
    
    # Actually, for 2x2 * 2x6, we treat each column of B independently but share the A coefficients.
    # The 7 M values are each 1x6 vectors.
    # C11 is 1x6: M1 - M4
    # C12 is 1x6: M5 + M4
    # C21 is 1x6: M2 - M3
    # C22 is 1x6: M1 - M2 + M3 + M5
    
    # So C[0,k] = (M1-M4)[k] for all k
    # C[1,k] = (M2-M3)[k] for all k
    # Wait, that's not right either. C is 2x6:
    # C[0,0:2] corresponds to C11 (first 2 cols), C[0,2:4] to C12, etc? No.
    
    # Actually, A1 is 2x2, B1 is 2x6. The product is 2x6.
    # C1[i,k] = sum_j A1[i,j] * B1[j,k]
    # This is just standard matrix multiplication of 2x2 and 2x6.
    # Strassen gives us a way to compute this with 7 scalar-vector multiplications.
    
    # The 7 M's are:
    # M1 = (A11+A22) * (B1+B2) : 1x6
    # M2 = (A21+A22) * B2 : 1x6
    # M3 = A11 * (B1-B2) : 1x6
    # M4 = A22 * (B2-B1) : 1x6
    # M5 = (A11+A12) * B2 : 1x6
    # M6 = (A21-A11) * (B1+B2) : 1x6
    # M7 = (A12-A22) * (B1+B2) : 1x6
    
    # And:
    # C[0,:] = M1 - M4 + M6 - M7  (hmm, let me look up the exact formula)
    
    # Standard Strassen for 2x2*2x2:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b22
    # p3 = a11(b11-b22)
    # p4 = a22(b22-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b22)
    # p7 = (a12-a22)(b11+b22)
    # c11 = p1 - p4 + p6 - p7
    # c12 = p5 + p4
    # c21 = p2 - p3 + p7
    # c22 = p1 - p2 + p3 + p5
    
    # So for 2x2*2x6, each p is 1x6, and:
    # C[0,:] = p1 - p4 + p6 - p7  (this gives the first row of C, but it's 1x6, and we need it to be the first 2 columns? No, it's the entire first row of the 2x6 result)
    
    # Wait, I'm confusing myself. For 2x2 * 2x6:
    # A1 = [[a11, a12], [a21, a22]]
    # B1 = [[b11, b12, ..., b16], [b21, b22, ..., b26]]
    # C1 = A1 * B1 = [[c11, c12, ..., c16], [c21, c22, ..., c26]]
    # where c1k = a11*b1k + a12*b2k, c2k = a21*b1k + a22*b2k
    
    # Using Strassen, we compute 7 products p1..p7, each 1x6, and then:
    # c1k = p1_k - p4_k + p6_k - p7_k
    # c2k = p2_k - p3_k + p7_k
    
    # Wait, that's not symmetric. Let me re-derive.
    
    # For each column k of B1, we have a 2x1 vector [b1k, b2k].
    # The product for column k is:
    # c1k = a11*b1k + a12*b2k
    # c2k = a21*b1k + a22*b2k
    
    # This is a 2x2 * 2x1 multiplication. Strassen for 2x2*2x2 doesn't directly apply to 2x2*2x1.
    
    # Actually, we can embed 2x1 into 2x2 by padding with zeros, but that's wasteful.
    
    # Alternative: just do the 2x2*2x6 multiplication directly. It has 4*6 = 24 scalar multiplications, but in tensor rank terms, it's rank 4 (one for each of the 4 entries of A1, each multiplied by the corresponding row of B1).
    
    # So the rank for A1*B1 is 4, not 7. Strassen doesn't help here because the B dimension is 6, not 2.
    
    # Let me reconsider. The tensor rank for 2x2*2x6 is at most 4 (schoolbook for the 2x2 part, extended to 6 columns).
    
    # So my earlier analysis was wrong. Let me restart.
    
    # For 2x3*3x6:
    # Schoolbook rank = 36.
    # 
    # Can we do better? Let's think about it as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The tensor T[i,j,k] = 1. We want to write this as sum_c u_c(i,j) v_c(j,k) w_c(k,i).
    # 
    # One approach: fix j. For each j, we have a 2x6 sub-tensor T[i,j,k] = 1.
    # This sub-tensor has rank 2 (it's the outer product of [1,1]^T and [1,1,...,1]^T, but with the i and k indices coupled through the output).
    # 
    # Actually, for fixed j, T[i,j,k] = 1 for all i,k. This is a 2x6 all-ones matrix. Its tensor rank (with the i,k coupling) is 2:
    # u_c1(i,j) = 1 if i=0, v_c1(j,k) = 1, w_c1(k,i) = 1 if i=0
    # u_c2(i,j) = 1 if i=1, v_c2(j,k) = 1, w_c2(k,i) = 1 if i=1
    # 
    # So for each j, we need rank 2. Total rank = 3*2 = 6.
    # 
    # Wait, that's much better than 36! Let me verify.
    
    # For j=0:
    # Term c0a: U[0*3+0, c0a]=1, V[0*6+k, c0a]=1 for all k, W[k*2+0, c0a]=1 for all k, else 0
    # Term c0b: U[1*3+0, c0b]=1, V[0*6+k, c0b]=1 for all k, W[k*2+1, c0b]=1 for all k, else 0
    # 
    # Contribution to T[i,0,k]:
    # i=0: U[0,c0a]*V[k,c0a]*W[2k,c0a] + U[0,c0b]*V[k,c0b]*W[2k,c0b] = 1*1*1 + 0 = 1 ✓
    # i=1: U[3,c0a]*V[k,c0a]*W[2k+1,c0a] + U[3,c0b]*V[k,c0b]*W[2k+1,c0b] = 0 + 1*1*1 = 1 ✓
    # 
    # For j=1:
    # Term c1a: U[0*3+1, c1a]=1, V[1*6+k, c1a]=1, W[k*2+0, c1a]=1
    # Term c1b: U[1*3+1, c1b]=1, V[1*6+k, c1b]=1, W[k*2+1, c1b]=1
    # 
    # For j=2:
    # Term c2a: U[0*3+2, c2a]=1, V[2*6+k, c2a]=1, W[k*2+0, c2a]=1
    # Term c2b: U[1*3+2, c2b]=1, V[2*6+k, c2b]=1, W[k*2+1, c2b]=1
    
    # Total rank = 6. Let me verify the full decomposition.
    
    # T[i,j,k] = sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*2+i, c]
    
    # For i=0, j=0, k=0:
    # Only c0a contributes: U[0, c0a]=1, V[0, c0a]=1, W[0, c0a]=1 -> 1
    # Others: 0
    # Total: 1 ✓
    
    # For i=1, j=0, k=0:
    # Only c0b contributes: U[3, c0b]=1, V[0, c0b]=1, W[1, c0b]=1 -> 1
    # Total: 1 ✓
    
    # For i=0, j=1, k=5:
    # Only c1a contributes: U[1, c1a]=1, V[11, c1a]=1, W[10, c1a]=1 -> 1
    # Total: 1 ✓
    
    # For i=1, j=2, k=3:
    # Only c2b contributes: U[5, c2b]=1, V[15, c2b]=1, W[7, c2b]=1 -> 1
    # Total: 1 ✓
    
    # Great! Rank 6 works. Can we do even better?
    
    # The lower bound for matrix multiplication rank is at least max(n, m, p) = 6.
    # So rank 6 is optimal!
    
    # Let me construct this.
    
    U = np.zeros((n*m, 6), dtype=float)
    V = np.zeros((m*p, 6), dtype=float)
    W = np.zeros((p*n, 6), dtype=float)
    
    for j in range(m):  # j = 0, 1, 2
        for i in range(n):  # i = 0, 1
            c = j * n + i  # column index in U, V, W
            U[i*m + j, c] = 1
            V[j*p + np.arange(p), c] = 1
            W[np.arange(p)*n + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
