"""
Matrix multiplication 3x4x4 tensor decomposition.

Strategy:
1. Start with the schoolbook decomposition (rank 48).
2. Apply Strassen's algorithm to 2x2 blocks to reduce rank.
   - The 3x4 matrix can be viewed as a 2x2 block matrix with blocks of sizes (2x2, 2x2, 1x4)
     Actually, let's think more carefully.
   
   A 3x4 matrix times a 4x4 matrix.
   
   Let's partition the 3x4 matrix A into blocks:
   A = [A11 A12; A21 A22; A31 A32] where A11, A21, A31 are 1x2 and A12, A22, A32 are 1x2.
   So A is 3 rows, each with 4 columns. Let's group columns into two groups of 2.
   
   A = [a11 a12 a13 a14]
       [a21 a22 a23 a24]
       [a31 a32 a33 a34]
   
   B = [b11 b12 b13 b14]
       [b21 b22 b23 b24]
       [b31 b32 b33 b34]
       [b41 b42 b43 b44]
   
   Let's partition B into 2x2 blocks:
   B = [B11 B12; B21 B22] where each block is 2x2.
   B11 = [b11 b12; b21 b22], B12 = [b13 b14; b23 b24]
   B21 = [b31 b32; b41 b42], B22 = [b33 b34; b43 b44]
   
   For A, let's also partition into 2x2 blocks where possible. But A is 3x4.
   Let's split A's columns into two groups of 2:
   A = [A1 A2] where A1 is 3x2 and A2 is 3x2.
   
   Then C = A*B = A1*B1 + A2*B2 where B1 = [B11; B21] (4x2) and B2 = [B12; B22] (4x2).
   
   Hmm, this doesn't directly give 2x2 block structure for Strassen.
   
   Alternative approach: Use the known result that the rank of 3x4x4 matrix multiplication
   is at most 36 (or lower). Let me think about what's achievable.
   
   Actually, let me use a more systematic approach. I'll use the fact that we can apply
   Strassen to 2x2 sub-problems.
   
   Let's view the 3x4x4 multiplication as follows:
   - Split the 4 columns of A (and rows of B) into two groups of 2.
   - A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
   - B = [B_top; B_bottom] where B_top is 2x4, B_bottom is 2x4
   
   Then C = A_left * B_top + A_right * B_bottom
   
   Each of these is a 3x2 * 2x4 multiplication, which has rank at most 3*2*4 = 24,
   but we can do better.
   
   For 3x2 * 2x4: We can further split. Let's say we have X (3x2) and Y (2x4).
   We can write this as a sum of rank-1 terms, but we can use Strassen-like tricks.
   
   Actually, let me just implement a known decomposition. The rank of m x n x p matrix
   multiplication is known for small cases. For 3x4x4, the known upper bound is 36
   (using block Strassen) or possibly lower.
   
   Let me try a different approach: use the fact that we can embed 3x4x4 into 4x4x4
   and use known decompositions of 4x4x4.
   
   The rank of 4x4x4 is known to be at most 49 (using Winograd's algorithm) or 48
   (schoolbook). Actually, the best known upper bound for 4x4x4 is 49.
   
   For 3x4x4, we can use a 4x4x4 decomposition and zero out one row/column.
   
   Let me try yet another approach: directly construct a decomposition using linear
   algebra. I'll use the fact that the multiplication tensor for 3x4x4 can be decomposed
   using a combination of Strassen steps.
   
   Let me partition:
   - A is 3x4: split into 2 blocks of rows: top 2 rows, bottom 1 row.
     A_top = A[0:2, :] (2x4), A_bottom = A[2:3, :] (1x4)
   - B is 4x4: split into 2 blocks of columns: left 2 cols, right 2 cols.
     B_left = B[:, 0:2] (4x2), B_right = B[:, 2:4] (4x2)
   
   C[0:2, 0:2] = A_top * B_left  (2x4 * 4x2 = 2x2)
   C[0:2, 2:4] = A_top * B_right (2x4 * 4x2 = 2x2)
   C[2:3, 0:2] = A_bottom * B_left (1x4 * 4x2 = 1x2)
   C[2:3, 2:4] = A_bottom * B_right (1x4 * 4x2 = 1x2)
   
   For the 2x2 * 4x2 * 2x2 parts, we can use Strassen on the 4x2 matrix.
   
   This is getting complex. Let me just implement a known rank-36 decomposition
   for 3x4x4 based on block Strassen.
   
   Actually, let me try a simpler approach: use the schoolbook but with Strassen
   applied to 2x2 sub-blocks where possible.
   
   For 3x4 * 4x4:
   - Split the 4 middle dimension into 2+2.
   - This gives us 4 sub-problems: (3x2)*(2x4) for each combination of row-block and col-block.
   
   Wait, I think the cleanest approach is:
   
   Let A be 3x4, B be 4x4, C = A*B be 3x4.
   
   Partition the 4-dimensional index (j) into two groups of 2: j in {0,1} and j in {2,3}.
   
   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
   
   Let A1 = A[:, 0:2] (3x2), A2 = A[:, 2:4] (3x2)
   Let B1 = B[0:2, :] (2x4), B2 = B[2:4, :] (2x4)
   
   C = A1*B1 + A2*B2
   
   Each term A1*B1 is 3x2 * 2x4 = 3x4. The rank of 3x2*2x4 is at most 3*2*4=24,
   but we can do better using Strassen on the 2x2 parts.
   
   For A1 (3x2) * B1 (2x4):
   Split B1 into two 2x2 blocks: B1 = [B11 | B12] where B11, B12 are 2x2.
   A1 is 3x2.
   
   C1 = A1 * B1 = [A1*B11 | A1*B12]
   
   A1*B11 is 3x2 * 2x2 = 3x2.
   A1*B12 is 3x2 * 2x2 = 3x2.
   
   For 3x2 * 2x2: We can use Strassen by embedding into 2x2*2x2*2x2.
   Actually, 3x2 * 2x2: Let A1 = [a1; a2; a3] where each ai is 1x2.
   Then A1*B = [a1*B; a2*B; a3*B] where each ai*B is 1x2.
   Each 1x2 * 2x2 = 1x2 multiplication has rank 2 (since it's just a linear combination).
   So total rank for 3x2*2x2 is 3*2 = 6.
   
   Hmm, that's not helping much.
   
   Let me just go with a known result. The rank of 3x4x4 matrix multiplication is
   at most 36. I'll construct a rank-36 decomposition.
   
   Actually, let me try to be more clever. I'll use the following approach:
   
   The multiplication (3,4,4) can be decomposed by:
   1. Using Strassen's 7-term decomposition for 2x2*2x2*2x2 where applicable.
   
   Let me partition:
   - Rows of A: 3 = 2 + 1
   - Cols of A / Rows of B: 4 = 2 + 2  
   - Cols of B / Cols of C: 4 = 2 + 2
   
   So we have a 2x2x2 block structure with some blocks being 1x2 or 2x1.
   
   Block (i,j,k) of the tensor corresponds to A_block[i,j] * B_block[j,k] contributing to C_block[i,k].
   
   i in {0,1} (2x2 and 1x2 row blocks of A)
   j in {0,1} (2x2 column blocks of A / row blocks of B)
   k in {0,1} (2x2 column blocks of B / C)
   
   A blocks:
   A[0,0] = A[0:2, 0:2] (2x2)
   A[0,1] = A[0:2, 2:4] (2x2)
   A[1,0] = A[2:3, 0:2] (1x2)
   A[1,1] = A[2:3, 2:4] (1x2)
   
   B blocks:
   B[0,0] = B[0:2, 0:2] (2x2)
   B[0,1] = B[0:2, 2:4] (2x2)
   B[1,0] = B[2:4, 0:2] (2x2)
   B[1,1] = B[2:4, 2:4] (2x2)
   
   C blocks:
   C[0,0] = C[0:2, 0:2] (2x2)
   C[0,1] = C[0:2, 2:4] (2x2)
   C[1,0] = C[2:3, 0:2] (1x2)
   C[1,1] = C[2:3, 2:4] (1x2)
   
   Now, C[i,k] = sum_j A[i,j] * B[j,k]
   
   For i=0, k=0: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] (2x2 = 2x2*2x2 + 2x2*2x2)
   For i=0, k=1: C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] (2x2 = 2x2*2x2 + 2x2*2x2)
   For i=1, k=0: C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] (1x2 = 1x2*2x2 + 1x2*2x2)
   For i=1, k=1: C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] (1x2 = 1x2*2x2 + 1x2*2x2)
   
   For the 2x2*2x2*2x2 parts, we can use Strassen (rank 7 instead of 8).
   For the 1x2*2x2*1x2 parts, the rank is 2 (just a linear combination).
   
   So total rank: 4*7 + 4*2 = 28 + 8 = 36.
   
   That gives us rank 36! Let me implement this.
   
   Strassen's algorithm for 2x2*2x2*2x2:
   M1 = (A11 + A22)(B11 + B22)
   M2 = (A21 + A22)B11
   M3 = A11(B12 - B22)
   M4 = A22(B21 - B11)
   M5 = (A11 + A12)B22
   M6 = (A21 - A11)(B11 + B12)
   M7 = (A12 - A22)(B21 + B22)
   
   C11 = M1 + M4 - M5 + M7
   C12 = M3 + M5
   C21 = M2 + M4
   C22 = M1 - M2 + M3 + M6
   
   For 1x2 * 2x2 * 1x2:
   Let a = [a1, a2] (1x2), B = [[b11,b12],[b21,b22]] (2x2)
   c = a*B = [a1*b11 + a2*b21, a1*b12 + a2*b22] (1x2)
   
   This is rank 2:
   c1 = a1*b11 + a2*b21
   c2 = a1*b12 + a2*b22
   
   We can write this as 2 rank-1 terms:
   Term 1: a1 * [b11, b12] -> contributes to c1 via b11, c2 via b12... no, that's not right.
   
   Actually, for 1x2 * 2x2 * 1x2:
   c1 = a1*b11 + a2*b21
   c2 = a1*b12 + a2*b22
   
   This is a bilinear map from (a1,a2) x (b11,b12,b21,b22) -> (c1,c2).
   The rank is 2 since c1 and c2 are independent linear combinations.
   
   Rank-1 decomposition:
   Term 1: U: a1, V: b11, W: c1 -> a1*b11 contributes to c1
   Term 2: U: a1, V: b12, W: c2 -> a1*b12 contributes to c2
   Term 3: U: a2, V: b21, W: c1 -> a2*b21 contributes to c1
   Term 4: U: a2, V: b22, W: c2 -> a2*b22 contributes to c2
   
   Wait, that's rank 4, not 2. Let me reconsider.
   
   For 1x2 * 2x2 * 1x2, the tensor is:
   T[a1,b11,c1] = 1
   T[a1,b12,c2] = 1
   T[a2,b21,c1] = 1
   T[a2,b22,c2] = 1
   
   This is actually rank 4 in general (since it's just the schoolbook for this small case).
   But wait, can we do better?
   
   c1 = a1*b11 + a2*b21
   c2 = a1*b12 + a2*b22
   
   This is equivalent to:
   [c1; c2] = [[b11, b21]; [b12, b22]] * [a1; a2]
   
   So it's a 2x2 matrix (from B) times a 2x1 vector (from A). The rank of this
   multiplication tensor is 2 (since it's just a 2x2 matrix-vector product, which
   has rank 2).
   
   Rank-1 decomposition for 2x2 * 1x2 -> 2x1:
   We need 2 terms:
   Term 1: U: a1, V: [b11, b12] (row 1 of B), W: [1, 0] (contributes to c1)
   Term 2: U: a2, V: [b21, b22] (row 2 of B), W: [1, 0]... no.
   
   Let me think again. The output is 2-dimensional (c1, c2).
   c1 = a1*b11 + a2*b21
   c2 = a1*b12 + a2*b22
   
   Rank-1 terms (each term is U_factor * V_factor * W_factor):
   Term 1: U: a1, V: b11, W: c1
   Term 2: U: a1, V: b12, W: c2
   Term 3: U: a2, V: b21, W: c1
   Term 4: U: a2, V: b22, W: c2
   
   That's 4 terms. Can we do it in 2?
   
   If we use:
   Term 1: U: a1, V: [b11, b12], W: [1, 0]
   Term 2: U: a2, V: [b21, b22], W: [1, 0]
   Then c1 = a1*b11 + a2*b21, c2 = 0. That's wrong.
   
   I think the rank is indeed 4 for 1x2*2x2*1x2 in the general case. But wait,
   that can't be right because the dimension of the output is only 2.
   
   Actually, the rank of the matrix multiplication tensor [m,n,p] is at most m*n*p,
   and for [1,2,2] it should be 2 (since it's just a 2x2 matrix times a scalar... no,
   it's a 1x2 vector times a 2x2 matrix giving a 1x2 vector).
   
   Hmm, let me just use the schoolbook for the 1x2*2x2*1x2 case (rank 4) and see
   what total rank we get: 4*7 + 4*4 = 28 + 16 = 44.
   
   That's worse than the schoolbook rank of 48. Let me reconsider.
   
   Actually, I realize I should just implement the Strassen-based decomposition
   and see what rank I get. Let me be more careful.
   
   For the full 3x4x4 problem, let me use a different partitioning:
   
   Partition the middle dimension (4) into 2+2.
   A = [A1 | A2] where A1 is 3x2, A2 is 3x2
   B = [B1; B2] where B1 is 2x4, B2 is 2x4
   
   C = A1*B1 + A2*B2
   
   Now, A1*B1 is 3x2 * 2x4 = 3x4.
   A2*B2 is 3x2 * 2x4 = 3x4.
   
   For 3x2 * 2x4:
   Partition the 4 columns into 2+2.
   A1*B1 = [A1*B1_left | A1*B1_right] where B1_left is 2x2, B1_right is 2x2.
   A1*B1_left is 3x2 * 2x2 = 3x2.
   A1*B1_right is 3x2 * 2x2 = 3x2.
   
   For 3x2 * 2x2 * 3x2:
   The rank is at most 3*2*2 = 12, but can we do better?
   
   3x2 * 2x2 = 3x2. Let A be 3x2, B be 2x2, C = A*B be 3x2.
   C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
   C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
   C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:]
   
   Each row of C is a linear combination of the rows of B with coefficients from A.
   The rank of this tensor is 3*2 = 6 (since each of the 3 output rows requires 2
   multiplications, and they're independent).
   
   Actually, the rank of [3,2,2] matrix multiplication is 6 (it's just 3 independent
   1x2*2x2*1x2 problems, each of rank 2).
   
   So 3x2*2x2 has rank 6.
   
   Then 3x2*2x4 = 3x2*2x2 + 3x2*2x2 has rank 6+6 = 12.
   
   And the full 3x4x4 = 3x2*2x4 + 3x2*2x4 has rank 12+12 = 24.
   
   Wait, that can't be right. The schoolbook rank is 48, and we're claiming rank 24?
   
   Let me re-examine. The issue is that when we write C = A1*B1 + A2*B2, the two
   terms contribute to the same output C. So we can't just add the ranks.
   
   Actually, we can: if A1*B1 and A2*B2 use disjoint sets of rank-1 terms, then
   the total rank is the sum. So rank(3x4x4) <= rank(3x2*2x4) + rank(3x2*2x4) = 24.
   
   But that seems too good. Let me verify: the rank of 4x4x4 is known to be at
   least 25 (and at most 49). If 3x4x4 had rank 24, that would be very surprising.
   
   I think my error is in the rank of 3x2*2x2. Let me reconsider.
   
   For 3x2 * 2x2 * 3x2:
   A is 3x2, B is 2x2, C = A*B is 3x2.
   
   C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
   C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
   
   For each i (0,1,2), we have:
   [C[i,0]; C[i,1]] = [[B[0,0], B[1,0]]; [B[0,1], B[1,1]]] * [A[i,0]; A[i,1]]
   
   This is a 2x2 matrix times a 2x1 vector. The rank of this operation is 2
   (it's just a matrix-vector product).
   
   So for 3 independent such operations, the rank is 3*2 = 6.
   
   Hmm, but that would mean rank(3x2*2x2) = 6, which seems low. Let me check:
   the schoolbook rank is 3*2*2 = 12. Can we really do it in 6?
   
   I think the issue is that the "rank" of a matrix-vector product is not the same
   as the tensor rank of the multiplication. Let me be more careful.
   
   The tensor T for 3x2*2x2*3x2 has indices (i,j,k) where i is row of A/C (0,1,2),
   j is col of A / row of B (0,1), k is col of B/C (0,1).
   
   T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
   
   So T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}.
   
   The rank of this tensor: we need to find the minimum number of rank-1 tensors
   that sum to T.
   
   A rank-1 tensor is U[i]*V[j]*W[k] for some vectors U, V, W.
   
   T[i,j,k] = 1 for all i,j,k. This is the all-ones tensor of size 3x2x2.
   The rank of the all-ones tensor of size m x n x p is min(m,n,p) = 2 in this case.
   
   Wait, that's not right either. The all-ones tensor [1,1,1,...,1] of size m x n x p
   has rank min(m,n,p). For 3x2x2, the rank is 2.
   
   But that's the rank of the tensor where every entry is 1. In our case, the tensor
   is indeed all-ones (since every combination of (i,j,k) appears exactly once).
   
   So rank(3x2*2x2) = 2? That seems too low.
   
   I think I'm confusing the tensor rank with something else. Let me look at this
   more carefully.
   
   The matrix multiplication tensor [m,n,p] is defined as:
   T = sum_{i=0}^{m-1} sum_{j=0}^{n-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C * delta_{j,k}
   
   Wait, no. For matrix multiplication C = A*B where A is m x n, B is n x p, C is m x p:
   C[i,k] = sum_j A[i,j] * B[j,k]
   
   The tensor is:
   T = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
   
   As a tensor in the space (R^m x R^n) * (R^n x R^p) * (R^m x R^p), the
   multiplication tensor is:
   T = sum_{i=0}^{m-1} sum_{j=0}^{n-1} sum_{k=0}^{p-1} (e_i * e_j) * (e_j * e_k) * (e_i * e_k)
   
   Where e_i * e_j denotes the tensor product.
   
   The rank of this tensor is what we're trying to minimize.
   
   For m=3, n=2, p=2:
   T = sum_{i=0}^{2} sum_{j=0}^{1} sum_{k=0}^{1} (e_i^A * e_j) * (e_j^B * e_k) * (e_i^C * e_k)
   
   This has 3*2*2 = 12 terms in the schoolbook decomposition.
   
   The rank is the minimum number of rank-1 tensors needed to represent T.
   
   I believe the rank of [3,2,2] is 6, not 2. My earlier reasoning was flawed.
   
   Let me just implement the Strassen-based approach and compute the actual rank.
   
   Here's my plan:
   1. Implement the schoolbook decomposition (rank 48).
   2. Implement a Strassen-based decomposition for 2x2*2x2*2x2 sub-problems.
   3. Combine them to get a lower-rank decomposition for 3x4x4.
   
   Let me just go with a known decomposition. I'll use the following approach:
   
   Partition 4 = 2 + 2 for the middle dimension.
   This gives us 4 sub-problems of type 3x2*2x4.
   
   For each 3x2*2x4 sub-problem, partition 4 = 2 + 2 for the output dimension.
   This gives us 2 sub-problems of type 3x2*2x2.
   
   For each 3x2*2x2 sub-problem, use the schoolbook (rank 12) or try to optimize.
   
   Total rank: 4 * 2 * 12 = 96? That's worse than schoolbook.
   
   I think I'm overcomplicating this. Let me just use the schoolbook and see if
   I can find a better decomposition through optimization.
   
   Actually, let me try a completely different approach: use the known result that
   the rank of 4x4x4 is at most 49 (using Winograd's algorithm), and then restrict
   to the 3x4x4 sub-problem.
   
   But implementing Winograd's algorithm for 4x4x4 is complex.
   
   Let me just go with the Strassen-based approach for 2x2 blocks and see what
   rank I get.
   
   Here's my revised plan:
   1. Partition A (3x4) into row blocks: top 2 rows, bottom 1 row.
   2. Partition B (4x4) into column blocks: left 2 cols, right 2 cols.
   3. This gives 4 sub-problems:
      - Top-left: 2x4 * 4x2 = 2x2 (use Strassen, rank 7)
      - Top-right: 2x4 * 4x2 = 2x2 (use Strassen, rank 7)
      - Bottom-left: 1x4 * 4x2 = 1x2 (schoolbook, rank 8)
      - Bottom-right: 1x4 * 4x2 = 1x2 (schoolbook, rank 8)
   
   Total rank: 7 + 7 + 8 + 8 = 30.
   
   Wait, but the middle dimension is 4, not 2. Let me re-partition.
   
   A is 3x4, B is 4x4.
   
   Let me partition the middle dimension (4) into 2+2:
   A = [A1 | A2] where A1 is 3x2, A2 is 3x2
   B = [B1; B2] where B1 is 2x4, B2 is 2x4
   
   C = A1*B1 + A2*B2
   
   Now, A1*B1 is 3x2 * 2x4 = 3x4.
   
   For 3x2 * 2x4:
   Partition the output dimension (4) into 2+2:
   A1*B1 = [A1*B1_left | A1*B1_right]
   where B1_left is 2x2, B1_right is 2x2.
   
   A1*B1_left is 3x2 * 2x2 = 3x2.
   A1*B1_right is 3x2 * 2x2 = 3x2.
   
   For 3x2 * 2x2 * 3x2, use schoolbook: rank 12.
   
   So A1*B1 has rank 12 + 12 = 24.
   Similarly, A2*B2 has rank 24.
   
   Total rank: 24 + 24 = 48. That's the same as schoolbook!
   
   I need to use Strassen for the 2x2*2x2*2x2 parts to get a reduction.
   
   Let me try:
   A1*B1_left is 3x2 * 2x2 = 3x2.
   This can be viewed as 3 independent 1x2 * 2x2 = 1x2 multiplications.
   Each 1x2 * 2x2 = 1x2 has rank 2 (since it's a 2x2 matrix times a 2x1 vector).
   
   Wait, I keep going back and forth. Let me just implement it and count the terms.
   
   For 1x2 * 2x2 * 1x2:
   A = [a0, a1] (1x2)
   B = [[b00, b01], [b10, b11]] (2x2)
   C = [c0, c1] = [a0*b00 + a1*b10, a0*b01 + a1*b11] (1x2)
   
   Rank-1 decomposition:
   Term 1: U: a0, V: b00, W: c0
   Term 2: U: a0, V: b01, W: c1
   Term 3: U: a1, V: b10, W: c0
   Term 4: U: a1, V: b11, W: c1
   
   Rank = 4.
   
   So 3x2 * 2x2 * 3x2 has rank 3*4 = 12.
   
   And 3x2 * 2x4 * 3x4 has rank 12 + 12 = 24.
   
   And 3x4 * 4x4 * 3x4 has rank 24 + 24 = 48.
   
   So I can't beat the schoolbook with this partitioning. I need a different approach.
   
   Let me try using Strassen directly on 2x2 blocks within the 3x4x4 problem.
   
   Here's an idea: embed the 3x4x4 problem into a 4x4x4 problem, use a rank-49
   decomposition for 4x4x4, and then restrict to the 3x4x4 sub-problem.
   
   But implementing a rank-49 decomposition for 4x4x4 is complex.
   
   Let me just go with the schoolbook rank of 48 for now, and see if I can find
   a better decomposition through optimization.
   
   Actually, let me try one more thing: use the fact that 3 = 2 + 1 and apply
   Strassen to the 2x2 part.
   
   A (3x4) = [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4.
   B (4x4) = [B_left | B_right] where B_left is 4x2, B_right is 4x2.
   
   C (3x4) = [C_top_left C_top_right; C_bottom_left C_bottom_right]
   
   C_top_left = A_top * B_left (2x4 * 4x2 = 2x2)
   C_top_right = A_top * B_right (2x4 * 4x2 = 2x2)
   C_bottom_left = A_bottom * B_left (1x4 * 4x2 = 1x2)
   C_bottom_right = A_bottom * B_right (1x4 * 4x2 = 1x2)
   
   For 2x4 * 4x2 * 2x2:
   Partition the middle dimension (4) into 2+2:
   A_top = [A_top1 | A_top2] where A_top1 is 2x2, A_top2 is 2x2.
   B_left = [B_left1; B_left2] where B_left1 is 2x2, B_left2 is 2x2.
   
   C_top_left = A_top1 * B_left1 + A_top2 * B_left2 (2x2 = 2x2*2x2 + 2x2*2x2)
   
   Use Strassen for each 2x2*2x2*2x2: rank 7.
   Total for C_top_left: 7 + 7 = 14.
   
   Similarly, C_top_right: 14.
   
   For 1x4 * 4x2 * 1x2:
   Partition the middle dimension (4) into 2+2:
   A_bottom = [A_bottom1 | A_bottom2] where A_bottom1 is 1x2, A_bottom2 is 1x2.
   B_left = [B_left1; B_left2] where B_left1 is 2x2, B_left2 is 2x2.
   
   C_bottom_left = A_bottom1 * B_left1 + A_bottom2 * B_left2 (1x2 = 1x2*2x2 + 1x2*2x2)
   
   For 1x2 * 2x2 * 1x2: rank 4 (as computed above).
   Total for C_bottom_left: 4 + 4 = 8.
   
   Similarly, C_bottom_right: 8.
   
   Total rank: 14 + 14 + 8 + 8 = 44.
   
   That's better than 48! Let me implement this.
   
   Strassen's algorithm for 2x2*2x2*2x2:
   M1 = (A11 + A22)(B11 + B22)
   M2 = (A21 + A22)B11
   M3 = A11(B12 - B22)
   M4 = A22(B21 - B11)
   M5 = (A11 + A12)B22
   M6 = (A21 - A11)(B11 + B12)
   M7 = (A12 - A22)(B21 + B22)
   
   C11 = M1 + M4 - M5 + M7
   C12 = M3 + M5
   C21 = M2 + M4
   C22 = M1 - M2 + M3 + M6
   
   Each M is a 2x2 matrix product, which we decompose into rank-1 terms.
   
   For M1 = (A11 + A22)(B11 + B22):
   Let P = A11 + A22, Q = B11 + B22.
   M1 = P*Q, which is 2x2*2x2*2x2.
   
   The rank-1 decomposition of P*Q:
   M1[0,0] = P[0,0]*Q[0,0] + P[0,1]*Q[1,0]
   M1[0,1] = P[0,0]*Q[0,1] + P[0,1]*Q[1,1]
   M1[1,0] = P[1,0]*Q[0,0] + P[1,1]*Q[1,0]
   M1[1,1] = P[1,0]*Q[0,1] + P[1,1]*Q[1,1]
   
   This is 8 rank-1 terms in the schoolbook, but Strassen reduces it to 7.
   
   The 7 Strassen terms for P*Q:
   S1 = (P[0,0] + P[1,1]) * (Q[0,0] + Q[1,1])
   S2 = (P[1,0] + P[1,1]) * Q[0,0]
   S3 = P[0,0] * (Q[0,1] - Q[1,1])
   S4 = P[1,1] * (Q[1,0] - Q[0,0])
   S5 = (P[0,0] + P[0,1]) * Q[1,1]
   S6 = (P[1,0] - P[0,0]) * (Q[0,0] + Q[0,1])
   S7 = (P[0,1] - P[1,1]) * (Q[1,0] + Q[1,1])
   
   Each Si is a scalar, and the result is:
   M1[0,0] = S1 + S4 - S5 + S7
   M1[0,1] = S3 + S5
   M1[1,0] = S2 + S4
   M1[1,1] = S1 - S2 + S3 + S6
   
   Now, each Si involves products of linear combinations of A and B entries.
   To convert this to a rank-1 decomposition of the original tensor, I need to
   express each Si as a product of a linear combination of A entries, a linear
   combination of B entries, and a linear combination of C entries.
   
   This is getting very complex. Let me just implement it programmatically.
   
   I'll create a function that takes a Strassen decomposition and converts it to
   rank-1 terms.
   
   Actually, let me just hardcode the rank-44 decomposition.
   
   Here's the structure:
   
   For C_top_left (2x2) = A_top1 * B_left1 + A_top2 * B_left2:
   Use Strassen for A_top1 * B_left1 (7 terms) and A_top2 * B_left2 (7 terms).
   Total: 14 terms.
   
   For C_top_right (2x2) = A_top1 * B_right1 + A_top2 * B_right2:
   Use Strassen for A_top1 * B_right1 (7 terms) and A_top2 * B_right2 (7 terms).
   Total: 14 terms.
   
   For C_bottom_left (1x2) = A_bottom1 * B_left1 + A_bottom2 * B_left2:
   Use schoolbook for A_bottom1 * B_left1 (4 terms) and A_bottom2 * B_left2 (4 terms).
   Total: 8 terms.
   
   For C_bottom_right (1x2) = A_bottom1 * B_right1 + A_bottom2 * B_right2:
   Use schoolbook for A_bottom1 * B_right1 (4 terms) and A_bottom2 * B_right2 (4 terms).
   Total: 8 terms.
   
   Grand total: 14 + 14 + 8 + 8 = 44 terms.
   
   Now I need to implement this. Let me define the indices carefully.
   
   A is 3x4: A[i,j] where i in {0,1,2}, j in {0,1,2,3}.
   B is 4x4: B[j,k] where j in {0,1,2,3}, k in {0,1,2,3}.
   C is 3x4: C[i,k] where i in {0,1,2}, k in {0,1,2,3}.
   
   Partition:
   A_top = A[0:2, :] (2x4): rows 0,1
   A_bottom = A[2:3, :] (1x4): row 2
   B_left = B[:, 0:2] (4x2): cols 0,1
   B_right = B[:, 2:4] (4x2): cols 2,3
   
   A_top1 = A[0:2, 0:2] (2x2)
   A_top2 = A[0:2, 2:4] (2x2)
   A_bottom1 = A[2:3, 0:2] (1x2)
   A_bottom2 = A[2:3, 2:4] (1x2)
   
   B_left1 = B[0:2, 0:2] (2x2)
   B_left2 = B[2:4, 0:2] (2x2)
   B_right1 = B[0:2, 2:4] (2x2)
   B_right2 = B[2:4, 2:4] (2x2)
   
   C_top_left = C[0:2, 0:2] (2x2) = A_top1 * B_left1 + A_top2 * B_left2
   C_top_right = C[0:2, 2:4] (2x2) = A_top1 * B_right1 + A_top2 * B_right2
   C_bottom_left = C[2:3, 0:2] (1x2) = A_bottom1 * B_left1 + A_bottom2 * B_left2
   C_bottom_right = C[2:3, 2:4] (1x2) = A_bottom1 * B_right1 + A_bottom2 * B_right2
   
   For the Strassen parts, I need to implement the 7-term decomposition for each
   2x2*2x2*2x2 sub-problem.
   
   Let me write a helper function for Strassen.
   
   Actually, I'll just implement the whole thing directly.
   
   For a 2x2*2x2*2x2 multiplication with Strassen:
   Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A*B.
   
   The 7 Strassen terms are:
   M1 = (a00 + a11)(b00 + b11)
   M2 = (a10 + a11)b00
   M3 = a00(b01 - b11)
   M4 = a11(b10 - b00)
   M5 = (a00 + a01)b11
   M6 = (a10 - a00)(b00 + b01)
   M7 = (a01 - a11)(b10 + b11)
   
   C00 = M1 + M4 - M5 + M7
   C01 = M3 + M5
   C10 = M2 + M4
   C11 = M1 - M2 + M3 + M6
   
   Each Mi is a scalar that is a product of a linear combination of A entries and
   a linear combination of B entries.
   
   To convert to rank-1 terms of the original tensor, I need to express each Mi
   as U_factor * V_factor * W_factor where U_factor is a linear combination of A
   entries, V_factor is a linear combination of B entries, and W_factor is a linear
   combination of C entries.
   
   This is complex because the Mi terms are intermediate results, not directly
   output entries.
   
   Let me think about this differently. The Strassen algorithm computes the
   product using 7 multiplications of linear combinations. Each multiplication
   is a scalar product, which is a rank-1 term in the tensor.
   
   So for each 2x2*2x2*2x2 sub-problem, I add 7 rank-1 terms.
   
   The U factor for each term is a linear combination of the 4 A entries.
   The V factor is a linear combination of the 4 B entries.
   The W factor is a linear combination of the 4 C entries.
   
   But wait, in the original problem, the A, B, C entries are specific elements,
   not linear combinations. So I need to map the Strassen terms back to the
   original tensor.
   
   This is getting too complex to do by hand. Let me just implement it programmatically.
   
   I'll create a list of rank-1 terms, where each term is (U_vector, V_vector, W_vector)
   with U_vector in R^(n*m), V_vector in R^(m*p), W_vector in R^(p*n).
   
   For the Strassen part, I'll add 7 terms per 2x2*2x2*2x2 sub-problem.
   
   Let me code this up.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    
    # We'll build a list of rank-1 terms.
    # Each term is (u, v, w) where u is a dict mapping (i,j) -> coeff,
    # v is a dict mapping (j,k) -> coeff, w is a dict mapping (k,i) -> coeff.
    # Wait, the indices are:
    # U index: i*m + j where i in [0,n), j in [0,m)
    # V index: j*p + k where j in [0,m), k in [0,p)
    # W index: k*n + i where k in [0,p), i in [0,n)
    
    # Let me use a different representation. Each rank-1 term is:
    # u: vector in R^(n*m)
    # v: vector in R^(m*p)
    # w: vector in R^(p*n)
    # such that for all i,j,k: sum_c u[i*m+j] * v[j*p+k] * w[k*n+i] = delta_{j,k} (for the multiplication tensor)
    
    # Actually, the multiplication tensor is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # Wait, no. The tensor is:
    # T = sum_{i,j,k} e_i^A * e_j^B * e_k^C * (delta_{j,k} if we're doing matrix mult)
    
    # Hmm, I'm getting confused. Let me just use the standard definition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (since every combination appears).
    
    # No, that's not right either. The tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} (e_i * e_j) * (e_j * e_k) * (e_i * e_k)
    
    # Where e_i * e_j is the tensor product.
    
    # The rank-1 decomposition is:
    # T = sum_c U_c * V_c * W_c
    # where U_c is in R^(n*m), V_c is in R^(m*p), W_c is in R^(p*n).
    
    # For the schoolbook:
    # For each (i,j,k): U_c[i*m+j] = 1, V_c[j*p+k] = 1, W_c[k*n+i] = 1.
    
    # For Strassen, the U, V, W vectors have multiple non-zero entries.
    
    # Let me just implement the rank-44 decomposition.
    
    terms = []  # Each term is (u, v, w) where u, v, w are dicts
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Helper to create a dict with a single entry
    def single(i, val=1):
        return {i: val}
    
    # Strassen for 2x2*2x2*2x2
    # A indices: (i1, j1) where i1 in {0,1}, j1 in {0,1}
    # B indices: (j1, k1) where j1 in {0,1}, k1 in {0,1}
    # C indices: (i1, k1) where i1 in {0,1}, k1 in {0,1}
    
    def strassen_2x2(i_offset, j_offset, k_offset):
        """Add Strassen terms for a 2x2*2x2*2x2 sub-problem.
        i_offset: offset for i index in A/C
        j_offset: offset for j index in A/B
        k_offset: offset for k index in B/C
        """
        # A[i_offset + i1, j_offset + j1] for i1 in {0,1}, j1 in {0,1}
        # B[j_offset + j1, k_offset + k1] for j1 in {0,1}, k1 in {0,1}
        # C[i_offset + i1, k_offset + k1] for i1 in {0,1}, k1 in {0,1}
        
        a00 = (i_offset, j_offset)
        a01 = (i_offset, j_offset + 1)
        a10 = (i_offset + 1, j_offset)
        a11 = (i_offset + 1, j_offset + 1)
        
        b00 = (j_offset, k_offset)
        b01 = (j_offset, k_offset + 1)
        b10 = (j_offset + 1, k_offset)
        b11 = (j_offset + 1, k_offset + 1)
        
        c00 = (i_offset, k_offset)
        c01 = (i_offset, k_offset + 1)
        c10 = (i_offset + 1, k_offset)
        c11 = (i_offset + 1, k_offset + 1)
        
        # M1 = (a00 + a11)(b00 + b11)
        u = {a00: 1, a11: 1}
        v = {b00: 1, b11: 1}
        w = {c00: 1, c01: 1, c10: 1, c11: 1}  # This is wrong, w should be specific
        # Actually, w is determined by how M1 contributes to C.
        # M1 contributes to C00 and C11.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C00 and +1 to C11.
        # But w is a vector in R^(p*n), not a scalar.
        
        # I think I need to reconsider. Each rank-1 term is:
        # u: R^(n*m), v: R^(m*p), w: R^(p*n)
        # The contribution to the tensor is u * v * w (outer product).
        
        # For Strassen, each of the 7 terms is a scalar multiplication:
        # M1 = (linear combo of A) * (linear combo of B)
        # This contributes to multiple C entries.
        
        # The issue is that a single rank-1 term u*v*w contributes to ALL
        # (i,j,k) combinations where u[i*m+j] != 0, v[j*p+k] != 0, w[k*n+i] != 0.
        
        # So for M1 = (a00 + a11)(b00 + b11):
        # u has non-zero entries at a00 and a11.
        # v has non-zero entries at b00 and b11.
        # w should have non-zero entries at the C positions that M1 contributes to.
        
        # M1 contributes to C00 (with coeff +1) and C11 (with coeff +1).
        # But w is a vector, and the contribution to C[i,k] is:
        # sum_j u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # This is getting too complex. Let me just use the schoolbook for now
        # and accept rank 48.
        
        pass
    
    # Let me just use the schoolbook decomposition.
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
