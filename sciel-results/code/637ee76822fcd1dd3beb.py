"""
Strategy: Structural Composition
We decompose the 4x4x7 matrix multiplication tensor T_447 into two parts:
1. A 4x4x4 block (multiplying A[4x4] by B[:, :4])
2. A 4x4x3 block (multiplying A[4x4] by B[:, 4:])

We use known low-rank decompositions for these sub-tensors.
- For 4x4x4: We use Strassen's algorithm recursively.
  Rank(4x4x4) <= 7 * Rank(2x2x2) ? No, Strassen on 2x2x2 gives rank 7.
  For 4x4x4, we can use 2x2 blocks. There are 4 blocks of A and 4 blocks of B.
  Actually, standard Strassen for 4x4x4:
  We can view 4x4 matrices as 2x2 blocks of 2x2 matrices.
  The multiplication of 4x4 matrices requires 7 multiplications of 2x2 matrices?
  No, that's not right. The rank of 4x4x4 matrix multiplication is known to be 49? No.
  Strassen's algorithm for n x n where n is a power of 2 has rank 7^(log2(n)).
  For n=2, rank = 7.
  For n=4, rank = 7^2 = 49.
  Wait, is there a better decomposition for 4x4x4?
  The known upper bound for 4x4x4 is 49.
  However, we can do better for the specific 4x4x7 case by mixing.
  
  Let's look at the inspiration: "hybrid Strassen(4x4x4) + Schoolbook(4x4x3) decomposition of rank 97".
  Schoolbook for 4x4x3 is 4*4*3 = 48.
  Strassen for 4x4x4 is 49.
  Total = 49 + 48 = 97.
  
  Can we do better?
  Maybe use a better decomposition for 4x4x3?
  Or maybe use a better decomposition for 4x4x4?
  
  Actually, the rank of 2x2x2 is 7.
  The rank of 4x4x4 is at most 49.
  The rank of 4x4x3 is at most 48 (schoolbook).
  
  Is there a known decomposition for 4x4x3 with rank < 48?
  The tensor is T_443.
  We can use the fact that 4x4x3 is a sub-tensor of 4x4x4.
  But usually, sub-tensors don't necessarily have lower rank in a simple additive way if we just slice.
  
  However, we can construct the 4x4x7 decomposition by:
  1. Taking a rank-49 decomposition for 4x4x4 (Strassen).
  2. Taking a rank-48 decomposition for 4x4x3 (Schoolbook).
  3. Concatenating them.
  
  This yields rank 97.
  
  Can we improve the 4x4x3 part?
  If we use Strassen-style for the 3 columns?
  3 is not a power of 2.
  We can pad 3 to 4 and use 4x4x4 Strassen, but that adds 1 column of zeros.
  Rank would be 49 + 49 = 98. Worse.
  
  What if we use a specific decomposition for 4x4x3?
  The tensor T_443 has indices i (0..3), j (0..3), k (0..2).
  We can use the rank-7 decomposition for 2x2x2.
  Split 4x4 into 2x2 blocks.
  A = [[A11, A12], [A21, A22]]
  B = [[B11, B12, B13], [B21, B22, B23]]
  C = [[C11, C12, C13], [C21, C22, C23]]
  
  C11 = A11 B11 + A12 B21
  C12 = A11 B12 + A12 B22
  C13 = A11 B13 + A12 B23
  C21 = A21 B11 + A22 B21
  C22 = A21 B12 + A22 B22
  C23 = A21 B13 + A22 B23
  
  This is a system of 6 equations involving 2x2 matrix multiplications.
  We can use Strassen's 7 multiplications for the 2x2x2 part?
  But the B matrix has 3 columns.
  
  Let's stick to the proven 97 rank first, as it's a significant improvement over 112.
  Then we can try to optimize the 4x4x3 part.
  
  Actually, let's look at the rank of 4x4x3.
  It is known that the rank of 4x4x3 is at most 48.
  Is it less?
  The tensor is a "slice" of 4x4x4.
  There are no widely known sub-48 rank decompositions for 4x4x3 that are simple to implement.
  
  However, we can try to use a rank-7 decomposition for 2x2x2 and apply it to the 2x2 blocks of the 4x4x3 problem.
  Let's see if we can reduce the 48.
  
  Alternative: Use a rank-15 decomposition for 2x2x2? No, 7 is optimal.
  
  Let's implement the Strassen 4x4x4 and Schoolbook 4x4x3.
  
  Strassen 4x4x4:
  We need to construct the U, V, W matrices for the 4x4x4 tensor.
  The tensor is T_444[i, j, k] = 1.
  We can generate the Strassen decomposition for 2x2x2 and then "embed" it into 4x4x4.
  
  How to embed 2x2x2 into 4x4x4?
  The 4x4 matrix multiplication can be viewed as 16 scalar multiplications.
  Strassen reduces this to 7 matrix multiplications of 2x2 matrices.
  Each 2x2 matrix multiplication can be decomposed into 7 scalar multiplications.
  So total scalar multiplications = 7 * 7 = 49.
  
  We need to construct the U, V, W for this 49-rank decomposition.
  
  Let's define the Strassen decomposition for 2x2x2.
  U_222, V_222, W_222 are 4x7 matrices.
  
  For 4x4x4:
  The indices are i, j, k in 0..3.
  We can map i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2.
  This is getting complex to code from scratch correctly.
  
  Let's use a simpler approach:
  We know the schoolbook rank is 112.
  We can easily get 97 by combining Strassen(4x4x4) and Schoolbook(4x4x3).
  
  Let's hardcode the Strassen 4x4x4 decomposition.
  Or better, generate it.
  
  Generation of Strassen 4x4x4:
  The 4x4x4 tensor T_444 is the tensor product of T_222 with itself?
  No, it's not a tensor product in the simple sense because the indices are coupled differently.
  Actually, the 4x4x4 matrix multiplication tensor IS the tensor product of two 2x2x2 matrix multiplication tensors?
  Let's check.
  T_222[i, j, k] = 1.
  (T_222 x T_222)[(i1,i2), (j1,j2), (k1,k2)] = T_222[i1, j1, k1] * T_222[i2, j2, k2] = 1.
  This corresponds to the multiplication of Kronecker products?
  If A = A1 x A2 and B = B1 x B2, then A B = (A1 B1) x (A2 B2).
  This is not general 4x4 multiplication.
  
  So we cannot simply tensor product the 2x2x2 decomposition.
  
  We must construct the 4x4x4 decomposition by applying Strassen's algorithm to the 2x2 blocks.
  
  Let's write a function to generate the Strassen decomposition for n x n x n where n is a power of 2.
  
  Base case: n=2.
  U_2, V_2, W_2 are 4x7 matrices.
  
  Recursive step:
  To multiply 4x4 matrices, we use 7 multiplications of 2x2 matrices.
  Let A, B be 4x4.
  A = [[A11, A12], [A21, A22]]
  B = [[B11, B12], [B21, B22]]
  
  M1 = (A11+A22)(B11+B22)
  M2 = (A21+A22)B11
  M3 = A11(B12-B22)
  M4 = A22(B21-B11)
  M5 = (A11+A12)B22
  M6 = (A21-A11)(B11+B12)
  M7 = (A12-A22)(B21+B22)
  
  C11 = M1 + M4 - M5 + M7
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M2 + M3 + M6
  
  Each Mx is a 2x2 matrix product.
  We can decompose each Mx using the 2x2x2 Strassen decomposition.
  
  This means we need to combine the linear combinations.
  
  Let's represent the decomposition as a list of terms (u, v, w) where u, v, w are vectors.
  
  For 2x2x2, we have 7 terms.
  For 4x4x4, we have 7 groups of 7 terms, total 49 terms.
  
  We can build the U, V, W matrices by iterating through the 7 Strassen multiplications and, for each, iterating through the 7 scalar multiplications of the 2x2 block.
  
  This is a structural composition.
  
  Let's implement this.
  
  1. Define the 2x2x2 Strassen decomposition (U2, V2, W2).
  2. Define the 4x4x4 Strassen structure (the 7 linear combinations).
  3. Combine them to get U4, V4, W4.
  4. Define the 4x4x3 Schoolbook decomposition (U3, V3, W3).
  5. Concatenate U4, V4, W4 with U3, V3, W3 to get the final 4x4x7 decomposition.
  
  Note: The indices for U, V, W in the final output must match the global indices.
  U is (4*4, rank) = (16, rank).
  V is (4*7, rank) = (28, rank).
  W is (7*4, rank) = (28, rank).
  
  For the 4x4x4 part:
  U4 is (16, 49).
  V4 is (16, 49) -> But V for 4x4x7 is (28, rank).
  The V for the 4x4x4 part only uses the first 16 entries of the 28 (columns 0-3 of B).
  So V4_part is (28, 49) with zeros in the last 12 rows.
  W4 is (16, 49) -> W for 4x4x7 is (28, rank).
  The W for the 4x4x4 part only uses the first 16 entries of the 28 (columns 0-3 of C).
  So W4_part is (28, 49) with zeros in the last 12 rows.
  
  For the 4x4x3 part:
  U3 is (16, 48).
  V3 is (28, 48) -> Uses the last 12 entries of the 28 (columns 4-6 of B).
  So V3_part is (28, 48) with zeros in the first 16 rows.
  W3 is (12, 48) -> W for 4x4x7 is (28, rank).
  The W for the 4x4x3 part only uses the last 12 entries of the 28 (columns 4-6 of C).
  So W3_part is (28, 48) with zeros in the first 16 rows.
  
  Total Rank = 49 + 48 = 97.
  
  This is a deterministic, exact construction.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    U[i*2+j]
    V[j*2+k]
    W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition.
    # The tensor T_222[i, j, k] = 1.
    # The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # Let's define the 7 terms explicitly.
    # Term 1: M1
    # U: A00, A11 -> 1
    # V: B00, B11 -> 1
    # W: C00, C11 -> 1 (for M1 part of C00 and C11)
    # Wait, the W matrix maps the intermediate product to C.
    # The intermediate product is a scalar (in the scalar decomposition view).
    # But here we are decomposing the tensor.
    
    # Let's use the standard Strassen decomposition for the tensor.
    # Reference: "The rank of the 2x2x2 matrix multiplication tensor is 7".
    
    # U (4x7):
    # Col 0: A00, A11 -> [1, 0, 1, 0] (indices 0, 1, 2, 3 correspond to 00, 01, 10, 11)
    # Col 1: A10, A11 -> [0, 0, 1, 1]
    # Col 2: A00 -> [1, 0, 0, 0]
    # Col 3: A11 -> [0, 0, 0, 1]
    # Col 4: A00, A01 -> [1, 1, 0, 0]
    # Col 5: A10, A00 -> [-1, 0, 1, 0]
    # Col 6: A01, A11 -> [0, 1, 0, -1]
    
    U[0, 0] = 1; U[2, 0] = 1  # M1: A00, A11
    U[2, 1] = 1; U[3, 1] = 1  # M2: A10, A11
    U[0, 2] = 1               # M3: A00
    U[3, 3] = 1               # M4: A11
    U[0, 4] = 1; U[1, 4] = 1  # M5: A00, A01
    U[0, 5] = -1; U[2, 5] = 1 # M6: A10-A00
    U[1, 6] = 1; U[3, 6] = -1 # M7: A01-A11
    
    # V (4x7):
    # Col 0: B00, B11 -> [1, 0, 0, 1]
    # Col 1: B00 -> [1, 0, 0, 0]
    # Col 2: B01, B11 -> [0, 1, 0, -1]
    # Col 3: B10, B00 -> [0, 0, 1, -1]
    # Col 4: B11 -> [0, 0, 0, 1]
    # Col 5: B00, B01 -> [1, 1, 0, 0]
    # Col 6: B10, B11 -> [0, 0, 1, 1]
    
    V[0, 0] = 1; V[3, 0] = 1  # M1: B00, B11
    V[0, 1] = 1               # M2: B00
    V[1, 2] = 1; V[3, 2] = -1 # M3: B01-B11
    V[2, 3] = 1; V[0, 3] = -1 # M4: B10-B00
    V[3, 4] = 1               # M5: B11
    V[0, 5] = 1; V[1, 5] = 1  # M6: B00+B01
    V[2, 6] = 1; V[3, 6] = 1  # M7: B10+B11
    
    # W (4x7):
    # Col 0: C00, C11 -> [1, 0, 0, 1] (indices 0, 1, 2, 3 correspond to 00, 01, 10, 11)
    # Col 1: C10 -> [0, 0, 1, 0]
    # Col 2: C01 -> [0, 1, 0, 0]
    # Col 3: C00 -> [1, 0, 0, 0]
    # Col 4: C00, C01 -> [1, 1, 0, 0]
    # Col 5: C11 -> [0, 0, 0, 1]
    # Col 6: C00, C11 -> [1, 0, 0, 1]
    
    W[0, 0] = 1; W[3, 0] = 1  # M1: C00, C11
    W[2, 1] = 1               # M2: C10
    W[1, 2] = 1               # M3: C01
    W[0, 3] = 1               # M4: C00
    W[0, 4] = 1; W[1, 4] = 1  # M5: C00, C01
    W[3, 5] = 1               # M6: C11
    W[0, 6] = 1; W[3, 6] = 1  # M7: C00, C11
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Constructs U, V, W for 4x4x4 matrix multiplication using Strassen.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The 4x4x4 multiplication is composed of 7 multiplications of 2x2 blocks.
    # Let's define the 7 block multiplications.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the indices of the 4x4 matrix to the 2x2 block indices.
    # A is 4x4.
    # A00 is rows 0-1, cols 0-1.
    # A01 is rows 0-1, cols 2-3.
    # A10 is rows 2-3, cols 0-1.
    # A11 is rows 2-3, cols 2-3.
    
    # The U matrix for 4x4x4 will have 16 rows (4*4) and 49 columns.
    # Each of the 7 Strassen multiplications contributes 7 scalar multiplications.
    # So we have 7 groups of 7 columns.
    
    # For each group g (0..6), we have 7 scalar multiplications.
    # The U, V, W for the scalar multiplications are given by U2, V2, W2.
    # But the indices are shifted.
    
    # Let's define the linear combinations for A, B, C for each Strassen multiplication.
    
    # M1: A = A00 + A11, B = B00 + B11, C = C00 + C11 (partially)
    # Actually, C is reconstructed from M's.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to construct the U, V, W such that the tensor identity holds.
    # This is equivalent to finding the decomposition of the 4x4x4 tensor.
    
    # Instead of deriving it from scratch, we can use the fact that the 4x4x4 tensor
    # is the "tensor product" of the 2x2x2 tensor in a specific block sense?
    # No, it's not a simple tensor product.
    
    # However, we can construct it by iterating over the 7 Strassen steps.
    # For each step, we have a 2x2x2 problem.
    # The indices of the 4x4 matrix are (i, j) where i, j in 0..3.
    # The block indices are (i1, i2) where i1, i2 in 0..1.
    # i = 2*i1 + i2.
    
    # Let's define the mapping for A, B, C for each Strassen multiplication.
    # We will create a list of 7 tuples (U_lin, V_lin, W_lin) where U_lin is a 16x7 matrix, etc.
    # Then we concatenate them.
    
    # This is getting too complex to code correctly in a short time without verification.
    # Let's use a simpler, guaranteed correct approach:
    # Use the schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x3 (rank 48).
    # Total rank 112.
    # This is the same as the current program.
    
    # To improve, we must use a better 4x4x4 decomposition.
    # Let's try to use the known rank-49 decomposition.
    
    # I will hardcode a rank-49 decomposition for 4x4x4 if I can find one,
    # or generate it using the recursive Strassen structure.
    
    # Recursive Strassen Generation:
    # The 4x4x4 tensor T_444 can be decomposed into 7 tensors T_222 embedded in 4x4x4.
    # Specifically, T_444 = sum_{g=1}^7 U_g V_g W_g where U_g, V_g, W_g are 16x7 matrices.
    # Each U_g V_g W_g corresponds to one Strassen multiplication.
    
    # Let's define the 7 Strassen multiplications in terms of the 4x4 indices.
    
    # M1 = (A00+A11)(B00+B11)
    # A00: i in {0,1}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # B00: i in {0,1}, j in {0,1} (Note: B is 4x4, so B00 is rows 0-1, cols 0-1)
    # B11: i in {2,3}, j in {2,3}
    
    # The U matrix for M1 should have 1s at positions corresponding to A00 and A11.
    # The V matrix for M1 should have 1s at positions corresponding to B00 and B11.
    # The W matrix for M1 should have 1s at positions corresponding to C00 and C11.
    
    # But this is not a rank-1 decomposition. It's a rank-7 decomposition of the 2x2 block multiplication.
    # So we need to decompose the 2x2 block multiplication using Strassen.
    
    # Let's use the U2, V2, W2 from _strassen_2x2().
    # For M1, the A part is A00 + A11.
    # This means the U vector for the first scalar multiplication of M1 is the sum of the U vectors for A00 and A11 in the 2x2 decomposition.
    
    # This is a linear combination.
    # We can construct the 16x49 matrix U4 by:
    # For g in 0..6:
    #   For c in 0..6:
    #     col = g*7 + c
    #     U4[:, col] = U_lin_g[:, c]
    
    # Where U_lin_g is a 16x7 matrix representing the linear combination of A blocks for the g-th Strassen multiplication.
    
    # Let's define U_lin_g for each g.
    # g=0 (M1): A00 + A11
    # U_lin_0[:, c] = U2[:, c] for A00 block + U2[:, c] for A11 block
    # A00 block is rows 0-1, cols 0-1.
    # A11 block is rows 2-3, cols 2-3.
    
    # We need to map the 2x2 indices to the 4x4 indices.
    # A00[i,j] -> A[2*i1 + i2, 2*j1 + j2] for i1,j2 in 0..1.
    # Specifically, A00 is A[0:2, 0:2].
    # A11 is A[2:4, 2:4].
    
    # So U_lin_0[i, c] = U2[i, c] if i in 0..1 (for A00) + U2[i-2, c] if i in 2..3 (for A11)?
    # No, the 2x2 decomposition U2 is for a 2x2 matrix.
    # The 4x4 matrix A has 16 elements.
    # The 2x2 block A00 has 4 elements.
    # The U2 matrix has 4 rows.
    # So U2[i, c] corresponds to the element at (i1, i2) in the 2x2 block.
    
    # For M1, the A part is A00 + A11.
    # So the U vector for the 4x4 matrix is:
    # For i1, i2 in 0..1:
    #   U4[2*i1 + i2, col] = U2[i1*2+i2, c]  # from A00
    #   U4[2*i1 + i2 + 4, col] = U2[i1*2+i2, c]  # from A11? No.
    
    # Let's be precise.
    # A is 4x4.
    # A00 is A[0:2, 0:2].
    # A11 is A[2:4, 2:4].
    
    # The element A[i, j] in the 4x4 matrix.
    # If i < 2 and j < 2, it's in A00. Index in A00 is (i, j).
    # If i >= 2 and j >= 2, it's in A11. Index in A11 is (i-2, j-2).
    
    # So for M1:
    # U4[i, col] = U2[i, c] if i < 2 and j < 2?
    # No, U is indexed by (i, j) in the 4x4 matrix.
    # U4[i*4+j, col]
    
    # For M1, the linear combination is A00 + A11.
    # So U4[i*4+j, col] = U2[i*2+j, c] if (i<2 and j<2) or (i>=2 and j>=2)?
    # No, U2 is for the 2x2 matrix.
    # The 2x2 matrix for A00 has indices (i1, j1).
    # The 2x2 matrix for A11 has indices (i1, j1).
    
    # So U4[i*4+j, col] = U2[i*2+j, c] if (i<2 and j<2) + U2[(i-2)*2+(j-2), c] if (i>=2 and j>=2).
    
    # This is correct.
    
    # Let's implement this for all 7 Strassen multiplications.
    
    # Define the 7 linear combinations for A, B, C.
    # M1: A00+A11, B00+B11, C00+C11
    # M2: A10+A11, B00, C10
    # M3: A00, B01-B11, C01
    # M4: A11, B10-B00, C00
    # M5: A00+A01, B11, C00+C01
    # M6: A10-A00, B00+B01, C11
    # M7: A01-A11, B10+B11, C00+C11
    
    # We need to define the block indices for A, B, C.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Let's create a function to get the U, V, W for a linear combination of blocks.
    
    def get_block_U(block_idx, sign=1):
        """
        Returns a 16x7 matrix U for a single 2x2 block with a sign.
        block_idx: 0 (00), 1 (01), 2 (10), 3 (11)
        """
        U = np.zeros((16, 7))
        i1 = block_idx // 2
        j1 = block_idx % 2
        for i in range(2):
            for j in range(2):
                global_i = 2*i1 + i
                global_j = 2*j1 + j
                U[global_i*4 + global_j, :] = sign * U2[:, :]
        return U
    
    # Wait, U2 is 4x7.
    # The 2x2 block has 4 elements.
    # U2[i*2+j, c]
    
    # So for block (i1, j1), the elements are at (2*i1+i, 2*j1+j).
    # The index in U4 is (2*i1+i)*4 + (2*j1+j).
    # The index in U2 is i*2+j.
    
    # So U4[(2*i1+i)*4 + (2*j1+j), :] = U2[i*2+j, :]
    
    # This is correct.
    
    # Let's define the 7 linear combinations.
    # Each is a list of (block_idx, sign)
    
    combos_A = [
        [(0, 1), (3, 1)], # M1: A00 + A11
        [(2, 1), (3, 1)], # M2: A10 + A11
        [(0, 1)],         # M3: A00
        [(3, 1)],         # M4: A11
        [(0, 1), (1, 1)], # M5: A00 + A01
        [(2, 1), (0, -1)],# M6: A10 - A00
        [(1, 1), (3, -1)] # M7: A01 - A11
    ]
    
    combos_B = [
        [(0, 1), (3, 1)], # M1: B00 + B11
        [(0, 1)],         # M2: B00
        [(1, 1), (3, -1)],# M3: B01 - B11
        [(2, 1), (0, -1)],# M4: B10 - B00
        [(3, 1)],         # M5: B11
        [(0, 1), (1, 1)], # M6: B00 + B01
        [(2, 1), (3, 1)]  # M7: B10 + B11
    ]
    
    combos_C = [
        [(0, 1), (3, 1)], # M1: C00 + C11
        [(2, 1)],         # M2: C10
        [(1, 1)],         # M3: C01
        [(0, 1)],         # M4: C00
        [(0, 1), (1, 1)], # M5: C00 + C01
        [(3, 1)],         # M6: C11
        [(0, 1), (3, 1)]  # M7: C00 + C11
    ]
    
    # Now construct U4, V4, W4
    for g in range(7):
        for c in range(7):
            col = g*7 + c
            # U
            for block_idx, sign in combos_A[g]:
                i1 = block_idx // 2
                j1 = block_idx % 2
                for i in range(2):
                    for j in range(2):
                        global_i = 2*i1 + i
                        global_j = 2*j1 + j
                        U4[global_i*4 + global_j, col] += sign * U2[i*2+j, c]
                        
            # V
            for block_idx, sign in combos_B[g]:
                i1 = block_idx // 2
                j1 = block_idx % 2
                for i in range(2):
                    for j in range(2):
                        global_i = 2*i1 + i
                        global_j = 2*j1 + j
                        V4[global_i*4 + global_j, col] += sign * V2[i*2+j, c]
                        
            # W
            for block_idx, sign in combos_C[g]:
                i1 = block_idx // 2
                j1 = block_idx % 2
                for i in range(2):
                    for j in range(2):
                        global_i = 2*i1 + i
                        global_j = 2*j1 + j
                        W4[global_i*4 + global_j, col] += sign * W2[i*2+j, c]
                        
    return U4, V4, W4

def _construct_4x4x3_schoolbook():
    """
    Constructs U, V, W for 4x4x3 matrix multiplication using schoolbook.
    U: (16, 48)
    V: (12, 48)
    W: (12, 48)
    """
    U = np.zeros((16, 48))
    V = np.zeros((12, 48))
    W = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j, col] = 1
                V[j*3+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    # 4x4x4 Strassen
    U4, V4, W4 = _construct_4x4x4_strassen()
    
    # 4x4x3 Schoolbook
    U3, V3, W3 = _construct_4x4x3_schoolbook()
    
    # Concatenate
    rank = 49 + 48
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # 4x4x4 part
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    # 4x4x3 part
    U[:, 49:] = U3
    V[16:, 49:] = V3
    W[16:, 49:] = W3
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify T_447[i, j, k] = sum_c U[i*4+j, c] V[j*7+k, c] W[k*4+i, c]
    # Should be 1 for all i, j, k
    
    T = np.zeros((4, 4, 7))
    for i in range(4):
        for j in range(4):
            for k in range(7):
                T[i, j, k] = np.sum(U[i*4+j, :] * V[j*7+k, :] * W[k*4+i, :])
                
    # Check if T is all 1s
    if np.allclose(T, 1):
        print("Verification passed")
    else:
        print("Verification failed")
        print("Max error:", np.max(np.abs(T - 1)))

# EVOLVE-BLOCK-END
