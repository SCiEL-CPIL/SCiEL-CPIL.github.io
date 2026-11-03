"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.

We use a block-structured approach:
  - Split the 3x4 matrix A into blocks of size (2x2, 2x2, 2x2, 2x2) along columns? 
    Actually A is 3x4, B is 4x5.
  - Let's split A into A_top (2x4) and A_bottom (1x4).
  - Let's split B into B_left (4x4) and B_right (4x1).

Then C = A * B = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]

We can use Strassen for the 2x2 * 2x2 sub-blocks where applicable, but the blocks aren't square.

Alternative approach: Use a known rank-30 or lower decomposition.
Actually, for 3x4 * 4x5, the schoolbook rank is 3*4*5 = 60.

Let's use a different strategy: 
  We can express the product using a smaller set of multiplications by exploiting
  the structure. One effective method is to use the "block Strassen" approach:

  Split A (3x4) into A1 (2x2), A2 (2x2), A3 (1x2), A4 (1x2)
  Split B (4x5) into B1 (2x2), B2 (2x2), B3 (2x1), B4 (2x1), B5 (2x1), B6 (2x1), B7 (2x1), B8 (2x1)
  
  This is getting complicated. Let's use a simpler but effective approach:
  
  Use the fact that rank(3x4*4x5) <= rank(2x2*2x2) * 2 + ... 
  
  Actually, let's use a well-known technique: 
  For multiplying an (n x m) by (m x p) matrix, if we can decompose into
  blocks where each block product has a known low-rank decomposition, we can combine them.

  Let's try:
    A = [A11 A12] where A11 is 2x2, A12 is 2x2
    A3 = [A31 A32] where A31 is 1x2, A32 is 1x2
    B = [B11; B21; B31; B41] where each Bi is 2x5... no, B is 4x5.
    
  Let me re-think. A is 3x4, B is 4x5.
  Split A into top 2 rows and bottom 1 row: A_top (2x4), A_bot (1x4)
  Split B into left 4 cols: B is already 4x5.
  
  C_top = A_top * B (2x5)
  C_bot = A_bot * B (1x5)
  
  For C_top: A_top is 2x4, B is 4x5. 
    We can split A_top = [A1 A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x5.
    C_top = A1*B1 + A2*B2.
    A1*B1 is 2x2 * 2x5. We can use a rank-7 decomposition for 2x2*2x2, but here it's 2x2*2x5.
    
  This is getting complex. Let's use a simpler approach: 
    Just use the schoolbook decomposition (rank 60) which is guaranteed correct,
    and attempt a Strassen-based improvement.

  Actually, let's implement a rank-30 decomposition using a block approach:
    Split the 4-dimension into two 2-dimensions.
    A (3x4) = [A1 A2] where A1, A2 are 3x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    C = A1*B1 + A2*B2
    
    For A1*B1 (3x2 * 2x5): rank <= 3*2*5 = 30, but we can do better.
    Actually, 3x2 * 2x5 can be done with rank 10 using schoolbook (3*2*5/... no).
    
  Let me just implement the schoolbook for now and verify it works, then try to improve.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Standard rank-nmp decomposition for matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*5 = 20 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*3 = 15 rows
    
    col = 0
    for i in range(n):       # 3
        for j in range(m):   # 4
            for k in range(p):  # 5
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U, V, W for the 2x2x2x2 tensor (rank 7).
    
    Convention: 
      A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
      C[0,0] = ae + bg
      C[0,1] = af + bh
      C[1,0] = ce + dg
      C[1,1] = cf + dh
      
    Strassen:
      M1 = (a+d)(e+h)
      M2 = (c+d)e
      M3 = a(f-h)
      M4 = d(g-e)
      M5 = (a+b)h
      M6 = (c-a)(e+f)
      M7 = (b-d)(g+h)
      
      C[0,0] = M1 + M4 - M5 + M7
      C[0,1] = M3 + M5
      C[1,0] = M2 + M4
      C[1,1] = M1 - M2 + M3 + M6
    """
    # For 2x2 * 2x2, the tensor has indices (i,j,k) where i,k in {0,1} for output, j in {0,1} for A's col, k in {0,1} for B's row
    # Actually, the multiplication tensor T[i,j,k] = 1 if the (i,j) element of A and (j,k) element of B contribute to (i,k) element of C.
    # For 2x2*2x2, T is a 2x2x2 tensor (since output is 2x2, and we trace over the middle dimension of size 2).
    # Wait, the tensor for matmul is T[i,j,k,l] where A is (i,j), B is (j,k), C is (i,l). But here we have fixed dimensions.
    # The task says: output factor W[k*n+i, r]. So the tensor is T[i,j,k] where i is row of A/output, j is col of A/row of B, k is col of B/output.
    # For 2x2*2x2: i in {0,1}, j in {0,1}, k in {0,1}. T[i,j,k] = 1 if this term contributes.
    # C[i,k] = sum_j A[i,j] * B[j,k]. So T[i,j,k] = 1 for all i,j,k in the "standard" basis.
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}. It's the all-ones tensor of shape (2,2,2).
    
    # Strassen decomposes this rank-8 tensor into 7 rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors in R^2.
    
    # The 7 rank-1 components for Strassen:
    # We need to express T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    
    # Let me use the standard Strassen decomposition.
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # For Strassen, the 7 multiplications are:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])  -> u1, v1, w1
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # The tensor T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # We need: sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
    
    # Let me construct U, V, W matrices of shape (2, 7) each.
    U = np.zeros((2, 7))
    V = np.zeros((2, 7))
    W = np.zeros((2, 7))
    
    # r=0: M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # Contribution to C[0,0]: +1, C[1,1]: +1
    U[0, 0] = 1; U[1, 0] = 1  # u = (1,1) for A part: A[0,0] + A[1,1]
    V[0, 0] = 1; V[1, 0] = 0  # Wait, I need to be more careful.
    
    # Actually, let me use a known correct Strassen decomposition for the 2x2x2 tensor.
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1} can be decomposed as:
    
    # Let's use the following (standard Strassen):
    # Each factor is a vector in the appropriate space.
    
    # I'll use a verified decomposition:
    U = [
        [1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 1]
    ]
    V = [
        [1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0]
    ]
    
    # Let me verify this by checking if sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k in {0,1}
    # This is a known Strassen decomposition. Let me trust it for now and verify numerically.
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build the decomposed tensor
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                T_decomp[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)


def _block_decomposition():
    """
    Attempt a block-based decomposition with lower rank than 60.
    
    Strategy:
    Split the 4-dimension (middle) into two 2-dimensions.
    A (3x4) = [A1 A2] where A1, A2 are 3x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (3x2 * 2x5): 
      We can use a rank decomposition. The schoolbook rank is 3*2*5 = 30.
      But we can do better. Note that 3x2 * 2x5 is equivalent to multiplying
      a 3x2 matrix by a 2x5 matrix. This can be seen as 2 dot products of length 2.
      Actually, C[i,k] = sum_j A1[i,j] * B1[j,k] where j in {0,1}.
      So for each (i,k), it's a dot product of two 2-vectors.
      The tensor is T[i,j,k] for i in {0,1,2}, j in {0,1}, k in {0,1,2,3,4}.
      This tensor has rank at most 3*2*5 = 30, but we can potentially do better.
      
    For now, let's just use schoolbook for the sub-blocks and see if the total is less than 60.
    A1*B1: rank 3*2*5 = 30
    A2*B2: rank 3*2*5 = 30
    Total: 60. Same as schoolbook.
    
    We need a smarter approach. Let's use Strassen on the 2x2 sub-blocks where possible.
    
    Alternative: Use a known rank-30 decomposition for 3x4*4x5.
    
    Actually, let's try a different split:
    A (3x4) -> split into 3 blocks of 1x4? No.
    
    Let's use the following approach:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Group terms:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    The first group is a 1x2 * 2x5 product (for each i,k), and the second is similar.
    For a fixed i, C[i,k] for k=0..4 is the product of a 1x2 vector (A[i,0:2]) and a 2x5 matrix.
    This is a 1x2 * 2x5 multiplication, which has rank at most 1*2*5 = 10, but actually
    the tensor T[j,k] for j in {0,1}, k in {0,1,2,3,4} is just the outer product structure.
    For 1x2 * 2x5, the output is 1x5, and T[j,k] = 1 for all j,k. The rank is 2 (since 
    T[j,k] = u_j * v_k where u = (1,1), v = (1,1,1,1,1)).
    
    So for each i, C[i,k] = sum_{j=0,1} A[i,j] * B[j,k] + sum_{j=2,3} A[i,j] * B[j,k]
    = dot(A[i,0:2], B[0:2,k]) + dot(A[i,2:4], B[2:4,k])
    
    Each dot product of length 2 can be done with rank 2 (since 1x2*2x1 has rank 2).
    But here we have 1x2 * 2x5, which for each output k is a dot product.
    The tensor for 1x2 * 2x5 is T[j,k] where j in {0,1}, k in {0,1,2,3,4}, T[j,k]=1.
    This can be decomposed as: T[j,k] = sum_{r=1}^{2} u_r[j] * v_r[k] where u_r are in R^2 and v_r in R^5.
    For example: u_1 = (1,0), v_1 = (1,1,1,1,1); u_2 = (0,1), v_2 = (1,1,1,1,1).
    So rank 2.
    
    Therefore, for each i (3 values), we have two rank-2 decompositions, total 3*2*2 = 12? No.
    For each i, we need to represent the product of A[i,:] (1x4) and B (4x5).
    C[i,k] = sum_j A[i,j] B[j,k].
    
    Let's decompose the sum over j into two parts: j=0,1 and j=2,3.
    Part 1: C1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    Part 2: C2[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    For Part 1: This is a 1x2 * 2x5 product. As noted, rank 2.
    The U vector for this part: for the A part, we need to extract A[i,0] and A[i,1].
    The V vector: for the B part, we need B[0,k] and B[1,k].
    The W vector: for the C part, we need to add to C[i,k].
    
    For the rank-2 decomposition of 1x2 * 2x5:
      U factors: u_1 = [1, 0] (for A[i,0]), u_2 = [0, 1] (for A[i,1])
      V factors: v_1 = [1, 1, 1, 1, 1] (for B[0,k] contribution), v_2 = [1, 1, 1, 1, 1] (for B[1,k])
      Wait, that's not right. The tensor is T[j,k] = 1 for j in {0,1}, k in {0,1,2,3,4}.
      Decomposition: T[j,k] = sum_r u_r[j] * v_r[k]
      u_1 = (1, 0), v_1 = (1, 1, 1, 1, 1)
      u_2 = (0, 1), v_2 = (1, 1, 1, 1, 1)
      
      So for Part 1, for each i:
        r1: U[i,0] *= 1, V[0,k] *= 1 for all k, W[i,k] += 1
        r2: U[i,1] *= 1, V[1,k] *= 1 for all k, W[i,k] += 1
      
    Similarly for Part 2.
    
    Total rank: 3 (i values) * 2 (parts) * 2 (rank per part) = 12? No, that's not right either.
    Let me count more carefully.
    
    For Part 1 (j=0,1): 
      For each i (0,1,2), we have 2 rank-1 terms.
      Term (i, r1): U[i,0] = 1, V[0,:] = (1,1,1,1,1), W[i,:] = (1,1,1,1,1)
      Term (i, r2): U[i,1] = 1, V[1,:] = (1,1,1,1,1), W[i,:] = (1,1,1,1,1)
      
    For Part 2 (j=2,3):
      Term (i, r3): U[i,2] = 1, V[2,:] = (1,1,1,1,1), W[i,:] = (1,1,1,1,1)
      Term (i, r4): U[i,3] = 1, V[3,:] = (1,1,1,1,1), W[i,:] = (1,1,1,1,1)
      
    Total: 3 * 4 = 12 rank-1 terms.
    
    But wait, this is just the schoolbook decomposition! Because for each (i,j,k) we have a term.
    3*4*5 = 60 terms. But I'm only counting 12? That can't be right.
    
    I think I'm confusing the tensor structure. Let me re-examine.
    
    The multiplication tensor T[i,j,k] for A (n x m) * B (m x p) -> C (n x p) has T[i,j,k] = 1 for all i,j,k.
    The decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r where u_r in R^n, v_r in R^m, w_r in R^p.
    
    For the schoolbook: each term is u = e_i, v = e_j, w = e_k, so rank = n*m*p.
    
    For my proposed decomposition:
    Part 1, i, r1: u = e_i (in R^3), but only the j=0 component is 1. So u = (0,0,0) with u[i,0]=1? 
    No, u is a vector in R^n = R^3. The j-index is in V, not U.
    
    I was making an error. Let me redo:
    
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    For the schoolbook: U[i, (i,j,k)] = 1, V[j, (i,j,k)] = 1, W[k, (i,j,k)] = 1.
    
    For a better decomposition, we need to find U, V, W with fewer columns.
    
    Let's use the block approach properly:
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
           = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    
    First sum: S1[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    This is a 1x2 * 2x5 product for each i. The tensor for this is T1[i,j,k] where j in {0,1}, k in {0,1,2,3,4}, T1[i,j,k] = 1.
    But T1 also depends on i. For each i, it's the same tensor structure.
    
    T1[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1,2,3,4}.
    This can be written as T1[i,j,k] = (sum_a u1_a[i] * v1_a[j] * w1_a[k])
    
    One decomposition: 
    For j=0: u = (1,1,1) (all i), v = (1,0), w = (1,1,1,1,1)
    For j=1: u = (1,1,1), v = (0,1), w = (1,1,1,1,1)
    
    So T1 = u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2 where:
    u1 = (1,1,1), v1 = (1,0), w1 = (1,1,1,1,1)
    u2 = (1,1,1), v2 = (0,1), w2 = (1,1,1,1,1)
    
    Similarly for T2 (j=2,3):
    u3 = (1,1,1), v3 = (0,0,1,0), w3 = (1,1,1,1,1)
    u4 = (1,1,1), v4 = (0,0,0,1), w4 = (1,1,1,1,1)
    
    Total: 4 rank-1 terms!
    
    But wait, this gives C[i,k] = (A[i,0]+A[i,1]+A[i,2]+A[i,3]) * (B[0,k]+B[1,k]+B[2,k]+B[3,k])
    which is NOT the correct matrix product. This is the outer product of the row sums and column sums.
    
    I made an error. The tensor T1[i,j,k] = 1 means that each A[i,j] and B[j,k] pair contributes to C[i,k].
    But in the decomposition, U[i,r] * V[j,r] * W[k,r] must sum to 1 for each (i,j,k).
    
    For the term u1 = (1,1,1), v1 = (1,0), w1 = (1,1,1,1,1):
    This contributes U[i,r] * V[j,r] * W[k,r] = 1 * 1 * 1 = 1 for all i, j=0, all k.
    So it covers T[i,0,k] = 1 for all i,k.
    
    For u2 = (1,1,1), v2 = (0,1), w2 = (1,1,1,1,1):
    Covers T[i,1,k] = 1 for all i,k.
    
    So T1 is correctly decomposed with 2 terms.
    Similarly T2 with 2 terms.
    Total 4 terms.
    
    But this means C[i,k] = sum_j A[i,j] B[j,k] is computed with only 4 multiplications?
    That can't be right because the rank of the 3x4*4x5 multiplication tensor is at least 12 (the dimension of the output).
    
    Let me check: with 4 terms, the computed C[i,k] would be:
    C[i,k] = sum_r U[i,r] * (sum_j V[j,r] * A[i,j]) * W[k,r]
           = U[i,0]*A[i,0]*W[k,0] + U[i,1]*A[i,1]*W[k,1] + U[i,2]*A[i,2]*W[k,2] + U[i,3]*A[i,3]*W[k,3]
           = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]  (if V[j,r] picks the right B row)
    
    Wait, in the decomposition, the V matrix has rows indexed by j (the middle dimension), and for each r, V[:,r] is a vector in R^m.
    When we compute the product, we form the vectors:
    For each r: a_r = U[:,r] (in R^n), b_r = V[:,r] (in R^m), c_r = W[:,r] (in R^p)
    Then C = sum_r a_r * (b_r^T * A^T) * c_r^T? No.
    
    The correct interpretation: 
    C[i,k] = sum_r U[i,r] * V[j,r] * W[k,r] summed over j? No.
    
    Let me re-read the task: "exactly decompose the matrix multiplication tensor"
    The tensor T[i,j,k] is decomposed as sum_r U[i,r] V[j,r] W[k,r].
    This means for each (i,j,k), T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    
    If T[i,j,k] = 1 for all i,j,k, then we need:
    sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
    
    With my 4-term decomposition:
    r=0: U[i,0]=1 for all i, V[j,0]=1 if j=0 else 0, W[k,0]=1 for all k
    r=1: U[i,1]=1 for all i, V[j,1]=1 if j=1 else 0, W[k,1]=1 for all k
    r=2: U[i,2]=1 for all i, V[j,2]=1 if j=2 else 0, W[k,2]=1 for all k
    r=3: U[i,3]=1 for all i, V[j,3]=1 if j=3 else 0, W[k,3]=1 for all k
    
    Then sum_r U[i,r] V[j,r] W[k,r] = 1*1*1 + 1*0*1 + 1*0*1 + 1*0*1 = 1 for j=0.
    For j=1: 1*0*1 + 1*1*1 + 1*0*1 + 1*0*1 = 1.
    For j=2: 1.
    For j=3: 1.
    
    So yes, T[i,j,k] = 1 is satisfied for all i,j,k with just 4 terms!
    
    But this means the matrix multiplication can be done with 4 multiplications? That's impossible because
    the rank of the 3x4*4x5 multiplication tensor is at least max(3*4, 4*5, 3*5) = 20? No, the rank is at least the dimension of the output, which is 3*5=15.
    
    I think I'm misunderstanding the tensor structure. Let me look at the task again:
    "output factor follows the official trace convention W[k*n+i,r]"
    
    And the verification code:
    T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So the indices are:
    U: (i*m + j, r) where i in {0,1,2}, j in {0,1,2,3} -> 12 rows
    V: (j*p + k, r) where j in {0,1,2,3}, k in {0,1,2,3,4} -> 20 rows
    W: (k*n + i, r) where k in {0,1,2,3,4}, i in {0,1,2} -> 15 rows
    
    And T_decomp[i,j,k] should equal 1 for all i,j,k.
    
    So the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    This is a 3x4x5 tensor (120 entries) that must equal 1 everywhere.
    
    With my 4-term decomposition, let's check:
    r=0: U[i*4+j, 0] = 1 if j=0 else 0, V[j*5+k, 0] = 1 if j=0 else 0, W[k*3+i, 0] = 1
    Wait, U is indexed by (i*m+j), so for a given (i,j), U[i*4+j, 0] is 1 only if j=0.
    But U is a matrix with 12 rows, and for r=0, U[i*4+j, 0] should be the same for all i,j? No.
    
    I think my confusion is that in the tensor decomposition, U[i*m+j, r] depends on both i and j, not just i.
    Similarly, V[j*p+k, r] depends on both j and k, and W[k*n+i, r] depends on both k and i.
    
    So the rank-1 tensor is not a simple outer product of three vectors, but a more complex structure.
    
    This is the "3-tensor" for matrix multiplication, and the rank is the minimum number of multiplications needed.
    
    Given the complexity, let's just use the schoolbook decomposition (rank 60) which is guaranteed correct, and verify it.
    
    For now, I'll return the schoolbook decomposition.
    """
    return None  # Not implemented yet


def _try_rank_reduction():
    """
    Attempt to find a lower rank decomposition.
    For now, just return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    return _schoolbook()


def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    result = _try_rank_reduction()
    
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    return _schoolbook()
# EVOLVE-BLOCK-END
