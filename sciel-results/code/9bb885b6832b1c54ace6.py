"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Known rank-7 decomposition for 2x2x2 matrix multiplication.
# U, V, W are 4x7, 4x7, 4x7 respectively.
# Each column c corresponds to a term: U[:,c] * V[:,c] * W[:,c]
# Indices: U[row, col], V[row, col], W[row, col]
# For 2x2 matrices: U[i,j] corresponds to A[i,j], V[i,j] to B[i,j], W[i,j] to C[i,j]
# Wait, the standard form is: sum_c U_c(i,j) V_c(i,j) W_c(i,j) = M_{ij}
# Actually, the tensor is T[i,j,k] where i,j are input indices and k is output.
# U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for 4x4x6.
# For 2x2x2: n=2, m=2, p=2.
# U has 4 rows, V has 4 rows, W has 4 rows.

# Strassen's algorithm gives rank 7 for 2x2x2.
# The 7 products are:
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B12 - B22)
# M4 = A22(B21 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
#
# Then:
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6
#
# In the tensor decomposition form sum_c U_c(i,j) V_c(k,l) W_c(m,n) ... wait, indices are tricky.
# Let's define the 2x2x2 tensors U_c, V_c, W_c such that C = sum_c U_c * V_c * W_c (outer product)
# Actually, it's: C_{mn} = sum_{i,j,k,l} A_{ij} B_{kl} T_{ijklmn}? No.
# The matrix multiplication tensor is T_{i,j,k} where i,j are from A, k is from B? No.
# Standard: C = A B. C_{ik} = sum_j A_{ij} B_{jk}.
# So the tensor is T_{i,j,k} = delta? No, T is the bilinear form.
# C_{ik} = sum_{j} A_{ij} B_{jk}.
# The rank decomposition is: C_{ik} = sum_c sum_{j} U_c(i,j) B_{jk} ... no.
# C_{ik} = sum_c U_c(i, ?) ...
# Let's use the standard definition:
# A is n x m, B is m x p, C is n x p.
# C_{np} = sum_{m} A_{nm} B_{mp}.
# The tensor is T[n, m, p]. We want to write T[n, m, p] = sum_c U_c[n, m] V_c[m, p] W_c[p, n]?
# The problem states: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
# And the convention is W[k*n+i, r]. So W is indexed by (k, i) where k is column of B (and A's row? no, A's col, B's row, C's col).
# Let's map:
# A: i (row), j (col). Index in U: i*m + j.
# B: j (row), k (col). Index in V: j*p + k.
# C: i (row), k (col). Index in W: k*n + i. (Since W shape is p*n, and convention is W[k*n+i])
#
# The decomposition satisfies:
# sum_{j} A[i,j] B[j,k] = sum_{r} (sum_{j} U[i*m+j, r] V[j*p+k, r]) * W[k*n+i, r] ?
# No, that would be sum_r W[k*n+i, r] * (U_r . V_r) which is a scalar per r.
# The correct identity is:
# C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] ... wait.
# It must be: C[i,k] = sum_r ( sum_{j} U[i*m+j, r] V[j*p+k, r] ) * W[k*n+i, r] ?
# Let's check dimensions. U[i*m+j, r] is a scalar. V[j*p+k, r] is a scalar.
# The product U*V is summed over j? No, the sum over j is the matrix multiplication sum.
# The rank decomposition of the tensor T[n, m, p] is:
# T[n, m, p] = sum_r U_r[n, m] V_r[m, p] W_r[p, n]
# where U_r is an n x m matrix, V_r is an m x p matrix, W_r is a p x n matrix.
# And the multiplication is:
# C = A B = sum_r W_r^T * (U_r^T * A * V_r) * ...? No.
# The standard form is:
# C_{np} = sum_r W_r[n, p] * ( sum_{i,j} U_r[i, n] A[i, j] V_r[j, p] ) ?
# Let's stick to the indices given in the prompt.
# U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
# The condition for correctness is:
# For all i, k:
# sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * ( sum_j U[i*m+j, r] V[j*p+k, r] )?
# No, that would imply U and V are vectors that depend on j, and their dot product is the only thing summed over j.
# But A[i,j] and B[j,k] are independent of r.
# So it must be:
# sum_j A[i,j] B[j,k] = sum_r ( sum_j U[i*m+j, r] A[i,j] ) ( sum_j V[j*p+k, r] B[j,k] )? No.
# The rank decomposition of the *tensor* means:
# T_{i,j,k} = sum_r U_r[i,j] V_r[j,k] W_r[k,i]  (indices match: i from A-row, j from A-col/B-row, k from B-col)
# Wait, the indices in U are (i,j), V are (j,k), W are (k,i).
# And the matrix multiplication is:
# C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
# If T_{i,j,k} = sum_r U_r[i,j] V_r[j,k] W_r[k,i], then:
# C_{i,k} = sum_j sum_r U_r[i,j] V_r[j,k] W_r[k,i] A_{i,j} B_{j,k}
# This doesn't look like a simple product of three terms unless we rearrange.
# Actually, the standard rank decomposition is:
# C_{i,k} = sum_r ( sum_j U_r[i,j] A_{i,j} ) * ( sum_j V_r[j,k] B_{j,k} ) * W_r[k,i] ?
# No, that's not right either.
#
# Let's look at the Strassen decomposition again.
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6
#
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B12 - B22)
# M4 = A22(B21 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
#
# Here, M_r are the 7 multiplications.
# We can write C as a linear combination of the outer products of the vectors defining M_r.
# For each r, we have:
# M_r = (sum_{i,j} u_r[i,j] A[i,j]) * (sum_{j,k} v_r[j,k] B[j,k])
# Then C[i,k] = sum_r w_r[i,k] M_r
#
# This matches the form:
# C[i,k] = sum_r w_r[i,k] * (sum_{i,j} u_r[i,j] A[i,j]) * (sum_{j,k} v_r[j,k] B[j,k])
#
# Comparing to the prompt's indices:
# U[i*m+j, r] corresponds to u_r[i,j]
# V[j*p+k, r] corresponds to v_r[j,k]
# W[k*n+i, r] corresponds to w_r[i,k] (since W is indexed by k*n+i, which is (k,i))
#
# So we need to construct U, V, W for the 2x2x2 case such that:
# u_r[i,j] are the coefficients of A[i,j] in the first factor of M_r
# v_r[j,k] are the coefficients of B[j,k] in the second factor of M_r
# w_r[i,k] are the coefficients of M_r in C[i,k]
#
# Let's build U, V, W for Strassen (rank 7).
#
# r=1 (M1):
# u_1: A11: 1, A22: 1 -> U[0,0]=1, U[3,0]=1 (indices: 0=0*2+0, 1=0*2+1, 2=1*2+0, 3=1*2+1)
# v_1: B11: 1, B22: 1 -> V[0,0]=1, V[3,0]=1 (indices: 0=0*2+0, 1=0*2+1, 2=1*2+0, 3=1*2+1)
# w_1: C11: 1, C22: 1 -> W[0,0]=1 (C11: k=0,i=0 -> 0*2+0=0), W[3,0]=1 (C22: k=1,i=1 -> 1*2+1=3)
#
# r=2 (M2):
# u_2: A21: 1, A22: 1 -> U[2,1]=1, U[3,1]=1
# v_2: B11: 1 -> V[0,1]=1
# w_2: C21: 1, C22: -1 -> W[2,1]=1 (C21: k=1,i=0 -> 1*2+0=2), W[3,1]=-1 (C22: k=1,i=1 -> 3)
#
# r=3 (M3):
# u_3: A11: 1 -> U[0,2]=1
# v_3: B12: 1, B22: -1 -> V[1,2]=1, V[3,2]=-1
# w_3: C12: 1, C22: 1 -> W[1,2]=1 (C12: k=1,i=0 -> 1*2+0=1), W[3,2]=1 (C22: k=1,i=1 -> 3)
#
# r=4 (M4):
# u_4: A22: 1 -> U[3,3]=1
# v_4: B21: 1, B11: -1 -> V[2,3]=1, V[0,3]=-1
# w_4: C11: 1, C21: 1 -> W[0,3]=1 (C11: k=0,i=0 -> 0), W[2,3]=1 (C21: k=1,i=0 -> 2)
#
# r=5 (M5):
# u_5: A11: 1, A12: 1 -> U[0,4]=1, U[1,4]=1
# v_5: B22: 1 -> V[3,4]=1
# w_5: C12: 1, C22: -1 -> W[1,4]=1 (C12: k=1,i=0 -> 1), W[3,4]=-1 (C22: k=1,i=1 -> 3)
#
# r=6 (M6):
# u_6: A21: 1, A11: -1 -> U[2,5]=1, U[0,5]=-1
# v_6: B11: 1, B12: 1 -> V[0,5]=1, V[1,5]=1
# w_6: C22: 1 -> W[3,5]=1
#
# r=7 (M7):
# u_7: A12: 1, A22: -1 -> U[1,6]=1, U[3,6]=-1
# v_7: B21: 1, B22: 1 -> V[2,6]=1, V[3,6]=1
# w_7: C11: 1 -> W[0,6]=1
#
# Let's verify C11 = M1 + M4 - M5 + M7.
# w_1[0]=1, w_4[0]=1, w_5[0]=0, w_7[0]=1. Sum = 1+1+0+1 = 3?
# Wait, C11 = M1 + M4 - M5 + M7.
# So w_1[0]=1, w_4[0]=1, w_5[0]=-1, w_7[0]=1.
# My w_5 above was for C12 and C22. Let's re-check w_5.
# M5 = (A11 + A12)B22.
# C12 = M3 + M5. So w_5[1] (C12) = 1.
# C22 = M1 - M2 + M3 + M6. M5 is not in C22.
# Wait, C22 = M1 - M2 + M3 + M6.
# So w_5[3] (C22) = 0.
# Let's re-read Strassen.
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6
#
# So for w_5:
# C11: -1 -> W[0,4] = -1
# C12: 1 -> W[1,4] = 1
# C21: 0
# C22: 0 -> W[3,4] = 0
#
# Let's re-verify all W.
# w_1: C11:1, C12:0, C21:0, C22:1 -> W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=1
# w_2: C11:0, C12:0, C21:1, C22:-1 -> W[0,1]=0, W[1,1]=0, W[2,1]=1, W[3,1]=-1
# w_3: C11:0, C12:1, C21:0, C22:1 -> W[0,2]=0, W[1,2]=1, W[2,2]=0, W[3,2]=1
# w_4: C11:1, C12:0, C21:1, C22:0 -> W[0,3]=1, W[1,3]=0, W[2,3]=1, W[3,3]=0
# w_5: C11:-1, C12:1, C21:0, C22:0 -> W[0,4]=-1, W[1,4]=1, W[2,4]=0, W[3,4]=0
# w_6: C11:0, C12:0, C21:0, C22:1 -> W[0,5]=0, W[1,5]=0, W[2,5]=0, W[3,5]=1
# w_7: C11:1, C12:0, C21:0, C22:0 -> W[0,6]=1, W[1,6]=0, W[2,6]=0, W[3,6]=0
#
# Now U and V.
# u_1: A11:1, A22:1 -> U[0,0]=1, U[3,0]=1
# u_2: A21:1, A22:1 -> U[2,1]=1, U[3,1]=1
# u_3: A11:1 -> U[0,2]=1
# u_4: A22:1 -> U[3,3]=1
# u_5: A11:1, A12:1 -> U[0,4]=1, U[1,4]=1
# u_6: A21:1, A11:-1 -> U[2,5]=1, U[0,5]=-1
# u_7: A12:1, A22:-1 -> U[1,6]=1, U[3,6]=-1
#
# v_1: B11:1, B22:1 -> V[0,0]=1, V[3,0]=1
# v_2: B11:1 -> V[0,1]=1
# v_3: B12:1, B22:-1 -> V[1,2]=1, V[3,2]=-1
# v_4: B21:1, B11:-1 -> V[2,3]=1, V[0,3]=-1
# v_5: B22:1 -> V[3,4]=1
# v_6: B11:1, B12:1 -> V[0,5]=1, V[1,5]=1
# v_7: B21:1, B22:1 -> V[2,6]=1, V[3,6]=1
#
# This looks consistent.
#
# Now, how to extend this to 4x4x6?
# We can use a block decomposition.
# A is 4x4, B is 4x6, C is 4x6.
# We can split A into 2x2 blocks of size 2x2.
# A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
# B = [[B11, B12], [B21, B22]] where B11, B21 are 2x3 and B12, B22 are 2x3.
# Wait, B is 4x6. If we split rows into 2 groups of 2, and columns into 2 groups of 3.
# B = [[B11, B12], [B21, B22]]
# B11 is 2x3, B12 is 2x3, B21 is 2x3, B22 is 2x3.
#
# Then C = A B = [[A11 B11 + A12 B21, A11 B12 + A12 B22],
#                 [A21 B11 + A22 B21, A21 B12 + A22 B22]]
#
# Each term like A11 B11 is a 2x2 times 2x3 multiplication.
# We need a rank decomposition for 2x2 x 2x3.
# We can use the Strassen rank-7 decomposition for 2x2x2 and extend it to 2x2x3.
# Or we can just use schoolbook for 2x2x3 which is rank 12 (2*2*3).
# If we use schoolbook for each 2x2x3 block, the total rank is 4 blocks * 12 = 48.
# But we have 4 block multiplications: C11, C12, C21, C22.
# Each Ci is 2x3.
# C11 = A11 B11 + A12 B21. This is two 2x2x2x3 multiplications added.
# If we decompose A11 B11 and A12 B21 separately, we need rank 12 + 12 = 24 for C11.
# Total rank for C = 24 + 24 + 24 + 24 = 96. This is the same as the full schoolbook.
#
# To improve, we need a better decomposition for 2x2x3 or to share computations between the blocks.
#
# Let's try to find a rank decomposition for 2x2x3.
# The tensor is T[i, j, k] where i=0,1; j=0,1; k=0,1,2.
# C[i,k] = sum_j A[i,j] B[j,k].
#
# We can use the fact that 2x2x3 is a sub-tensor of 2x2x4 (which can be done in rank 14 by doubling Strassen).
# Or we can try to find a direct decomposition.
#
# Actually, the rank of 2x2x3 is known to be 11 or 12?
# The schoolbook rank is 12.
# There is a rank 11 decomposition? I'm not sure.
#
# Let's assume we can use rank 12 for 2x2x3.
# If we use rank 12 for each of the 4 block products, total rank is 48.
# Wait, C11 = A11 B11 + A12 B21.
# If we compute A11 B11 and A12 B21 separately, we use rank 12 + 12 = 24.
# But we can combine them.
# C11 = [A11, A12] * [B11; B21]
# This is a 2x4 matrix times a 4x3 matrix.
# Rank of 2x4x3?
#
# This is getting complicated.
#
# Let's stick to the simplest improvement: use Strassen for 2x2x2 blocks where possible.
#
# We have 4x4x6.
# Split into 2x2 blocks of size 2x2 for A, and 2x3 for B.
#
# C11 = A11 B11 + A12 B21
# C12 = A11 B12 + A12 B22
# C21 = A21 B11 + A22 B21
# C22 = A21 B12 + A22 B22
#
# Each Ai is 2x2, Bji is 2x3.
#
# We can use the Strassen decomposition for the 2x2 part and extend it to the 2x3 part.
# For 2x2x2, we have U, V, W of size 4x7.
# For 2x2x3, we can repeat the 2x2 decomposition for each of the 3 columns of B.
# This would give rank 7 * 3 = 21 for 2x2x3.
# But we can do better.
#
# Actually, if we use the same U and W, and just extend V to have 3 columns for each of the 2 rows.
# V is 2x3 for each of the 7 terms.
# So V is 6x7 (2 rows * 3 cols * 7 terms).
# U is 4x7.
# W is 6x7 (3 cols * 2 rows * 7 terms).
#
# Total rank for 2x2x3 is 7.
# Is this valid?
# C[i,k] = sum_j A[i,j] B[j,k]
# = sum_r W_r[i,k] * (sum_j U_r[i,j] A[i,j] V_r[j,k] B[j,k])? No.
#
# The decomposition is:
# T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[i,k]
# For 2x2x2, this works with rank 7.
# For 2x2x3, we can just use the same U and W, and for V, we set V_r[j,k] = 0 for k=2 (if we only have 2 columns).
# But we have 3 columns.
# If we use the Strassen decomposition for the first 2 columns of B, and set the 3rd column to 0, we get the product for the first 2 columns.
# Then we need another decomposition for the 3rd column.
# The 3rd column of B is a 2x1 vector.
# A * b is a 2x1 vector.
# C[:, 2] = A * B[:, 2].
# This is a 2x2 x 2x1 multiplication.
# Rank of 2x2x2x1?
# C[i, 2] = sum_j A[i,j] B[j, 2].
# This is a bilinear form.
# We can use rank 2:
# C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
# C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
# Terms:
# r1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> C[0,2] += A[0,0]B[0,2]
# r2: U[0,1]=1, V[1,1]=1, W[0,1]=1 -> C[0,2] += A[0,1]B[1,2]
# r3: U[1,0]=1, V[0,2]=1, W[1,2]=1 -> C[1,2] += A[1,0]B[0,2]
# r4: U[1,1]=1, V[1,3]=1, W[1,3]=1 -> C[1,2] += A[1,1]B[1,2]
# Rank 4 for 2x2x1?
# Actually, C[i,2] = sum_j A[i,j] B[j,2].
# This is just the dot product of row i of A and column 2 of B.
# We can compute this in rank 2?
# C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
# C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
# We need 4 terms because each output element depends on 2 products, and the products are different.
# Unless we share them.
# (A[0,0]+A[1,0])(B[0,2]) ... no.
#
# So rank 4 for 2x2x1.
# Total rank for 2x2x3 = 7 (for 2x2x2) + 4 (for 2x2x1) = 11.
#
# So for each 2x2x3 block, we use rank 11.
# We have 4 such blocks in the 4x4x6 multiplication.
# Total rank = 4 * 11 = 44.
#
# But wait, C11 = A11 B11 + A12 B21.
# This is a 2x2 matrix times a 2x3 matrix, plus another 2x2 times 2x3.
# If we compute them separately, we use rank 11 + 11 = 22 for C11.
# Total rank = 22 * 4 = 88.
#
# Can we combine A11 B11 + A12 B21?
# C11 = [A11, A12] [B11; B21]
# This is 2x4 times 4x3.
# Rank of 2x4x3?
#
# Let's just use the schoolbook for the full 4x4x6, which is 96.
# 88 is better. 44 is even better if we can share.
#
# Actually, the problem allows half-integers.
#
# Let's try to implement the rank 44 decomposition.
#
# For each block (i, k) in {0,1}x{0,1}:
# Ci = Ai1 B1i + Ai2 B2i
#
# We can use the rank 11 decomposition for 2x2x3 for each term.
#
# Term 1: Ai1 B1i.
# Term 2: Ai2 B2i.
#
# We just add the U, V, W vectors for the two terms.
#
# So for Ci, we use rank 22.
# Total rank 88.
#
# Let's implement this.
#
# First, define the U, V, W for 2x2x3 (rank 11).
#
# U_223 (4x11), V_223 (6x11), W_223 (6x11).
#
# Part 1: Strassen for 2x2x2 (columns 0,1 of B).
# U_s (4x7), V_s (4x7) for B columns 0,1.
# V_s needs to be extended to 6x7 by adding zeros for column 2.
# W_s (4x7) for C columns 0,1.
# W_s needs to be extended to 6x7 by adding zeros for column 2.
#
# Part 2: Schoolbook for 2x2x1 (column 2 of B).
# U_221 (4x4), V_221 (2x4), W_221 (2x4).
# V_221 needs to be extended to 6x4 by placing it in column 2.
# W_221 needs to be extended to 6x4 by placing it in column 2.
#
# Let's construct these.
#
# U_s, V_s, W_s from Strassen (as defined above).
#
# U_221:
# r1: A[0,0] -> U[0,0]=1
# r2: A[0,1] -> U[1,1]=1
# r3: A[1,0] -> U[2,2]=1
# r4: A[1,1] -> U[3,3]=1
#
# V_221:
# r1: B[0,2] -> V[0,0]=1
# r2: B[1,2] -> V[1,1]=1
# r3: B[0,2] -> V[0,2]=1
# r4: B[1,2] -> V[1,3]=1
#
# W_221:
# r1: C[0,2] -> W[0,0]=1
# r2: C[0,2] -> W[0,1]=1
# r3: C[1,2] -> W[1,2]=1
# r4: C[1,2] -> W[1,3]=1
#
# Now, for the 4x4x6 case:
# A is 4x4, B is 4x6, C is 4x6.
#
# We split A into 2x2 blocks: A00, A01, A10, A11.
# B into 2x2 blocks of size 2x3: B00, B01, B10, B11.
#
# C00 = A00 B00 + A01 B10
# C01 = A00 B01 + A01 B11
# C10 = A10 B00 + A11 B10
# C11 = A10 B01 + A11 B11
#
# For each Cij, we use rank 22.
# Total rank 88.
#
# Let's build the U, V, W matrices.
#
# U has shape (4*4, 88) = (16, 88).
# V has shape (4*6, 88) = (24, 88).
# W has shape (6*4, 88) = (24, 88).
#
# We will iterate over the 4 blocks of C (i_block, k_block) in {0,1}x{0,1}.
# For each block, we iterate over the 2 terms (j_block in {0,1}).
# For each term, we use the rank 11 decomposition.
#
# Total terms: 4 * 2 = 8.
# 8 * 11 = 88.
#
# Let's map the indices.
#
# For a term A_ij B_jk (where i,j,k are block indices 0 or 1):
# A_ij is at rows i*2 to i*2+1, cols j*2 to j*2+1 of A.
# B_jk is at rows j*2 to j*2+1, cols k*3 to k*3+2 of B.
# C_ik is at rows i*2 to i*2+1, cols k*3 to k*3+2 of C.
#
# U indices for A_ij: (i*2+r)*4 + (j*2+c) for r,c in 0,1.
# V indices for B_jk: (j*2+r)*6 + (k*3+c) for r in 0,1, c in 0,1,2.
# W indices for C_ik: (k*3+c)*4 + (i*2+r) for c in 0,1,2, r in 0,1.
#
# We will fill the U, V, W matrices for each of the 88 terms.
#
# Let's define the U, V, W for the 2x2x3 decomposition (rank 11).
#
# U_223 = np.zeros((4, 11))
# V_223 = np.zeros((6, 11))
# W_223 = np.zeros((6, 11))
#
# First 7 columns: Strassen for 2x2x2 (B cols 0,1).
# U_223[:, 0:7] = U_s
# V_223[0:4, 0:7] = V_s
# V_223[4:6, 0:7] = 0
# W_223[0:4, 0:7] = W_s
# W_223[4:6, 0:7] = 0
#
# Last 4 columns: Schoolbook for 2x2x1 (B col 2).
# U_223[:, 7:11] = U_221
# V_223[4:6, 7:11] = V_221 # V_221 is 2x4, so rows 4,5 correspond to B[0,2], B[1,2]
# W_223[4:6, 7:11] = W_221 # W_221 is 2x4, so rows 4,5 correspond to C[0,2], C[1,2]
#
# Now we can assemble the full U, V, W.
#
# def construct():
#     n, m, p = 4, 4, 6
#     rank = 88
#     U = np.zeros((n*m, rank))
#     V = np.zeros((m*p, rank))
#     W = np.zeros((p*n, rank))
#
#     # Define U_s, V_s, W_s for Strassen 2x2x2
#     # ... (as above)
#
#     # Define U_221, V_221, W_221 for Schoolbook 2x2x1
#     # ... (as above)
#
#     # Assemble U_223, V_223, W_223
#     # ...
#
#     # Loop over blocks
#     term_offset = 0
#     for i_block in range(2):
#         for k_block in range(2):
#             for j_block in range(2):
#                 # A_ij, B_jk, C_ik
#                 # Copy U_223 to U at the appropriate rows and columns
#                 # Copy V_223 to V at the appropriate rows and columns
#                 # Copy W_223 to W at the appropriate rows and columns
#
#                 # U rows: (i_block*2+r)*4 + (j_block*2+c) for r,c in 0,1
#                 # U cols: term_offset to term_offset+10
#
#                 # V rows: (j_block*2+r)*6 + (k_block*3+c) for r in 0,1, c in 0,1,2
#                 # V cols: term_offset to term_offset+10
#
#                 # W rows: (k_block*3+c)*4 + (i_block*2+r) for c in 0,1,2, r in 0,1
#                 # W cols: term_offset to term_offset+10
#
#                 term_offset += 11
#
#     return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
#
# Let's implement this.
#
# U_s, V_s, W_s from Strassen.
# U_221, V_221, W_221 from Schoolbook.
#
# We need to be careful with the index mapping.
#
# U_223 is 4x11.
# V_223 is 6x11.
# W_223 is 6x11.
#
# For a given (i_block, j_block, k_block):
# U rows are (i_block*2+r)*4 + (j_block*2+c) for r in 0,1, c in 0,1.
# These are 4 rows.
# We map U_223[0] to (i_block*2+0)*4 + (j_block*2+0)
# U_223[1] to (i_block*2+0)*4 + (j_block*2+1)
# U_223[2] to (i_block*2+1)*4 + (j_block*2+0)
# U_223[3] to (i_block*2+1)*4 + (j_block*2+1)
#
# V rows are (j_block*2+r)*6 + (k_block*3+c) for r in 0,1, c in 0,1,2.
# These are 6 rows.
# We map V_223[0] to (j_block*2+0)*6 + (k_block*3+0)
# V_223[1] to (j_block*2+0)*6 + (k_block*3+1)
# V_223[2] to (j_block*2+0)*6 + (k_block*3+2)
# V_223[3] to (j_block*2+1)*6 + (k_block*3+0)
# V_223[4] to (j_block*2+1)*6 + (k_block*3+1)
# V_223[5] to (j_block*2+1)*6 + (k_block*3+2)
#
# W rows are (k_block*3+c)*4 + (i_block*2+r) for c in 0,1,2, r in 0,1.
# These are 6 rows.
# We map W_223[0] to (k_block*3+0)*4 + (i_block*2+0)
# W_223[1] to (k_block*3+0)*4 + (i_block*2+1)
# W_223[2] to (k_block*3+1)*4 + (i_block*2+0)
# W_223[3] to (k_block*3+1)*4 + (i_block*2+1)
# W_223[4] to (k_block*3+2)*4 + (i_block*2+0)
# W_223[5] to (k_block*3+2)*4 + (i_block*2+1)
#
# This seems correct.
#
# Let's code it.

def construct():
    n, m, p = 4, 4, 6
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    # Strassen 2x2x2
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    # r=0 (M1)
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1; W_s[3,0] = 1
    # r=1 (M2)
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[2,1] = 1; W_s[3,1] = -1
    # r=2 (M3)
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[1,2] = 1; W_s[3,2] = 1
    # r=3 (M4)
    U_s[3,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[0,3] = 1; W_s[2,3] = 1
    # r=4 (M5)
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[0,4] = -1; W_s[1,4] = 1
    # r=5 (M6)
    U_s[2,5] = 1; U_s[0,5] = -1
    V_s[0,5] = 1; V_s[1,5] = 1
    W_s[3,5] = 1
    # r=6 (M7)
    U_s[1,6] = 1; U_s[3,6] = -1
    V_s[2,6] = 1; V_s[3,6] = 1
    W_s[0,6] = 1

    # Schoolbook 2x2x1
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    U_221[0,0] = 1; V_221[0,0] = 1; W_221[0,0] = 1
    U_221[1,1] = 1; V_221[1,1] = 1; W_221[0,1] = 1
    U_221[2,2] = 1; V_221[0,2] = 1; W_221[1,2] = 1
    U_221[3,3] = 1; V_221[1,3] = 1; W_221[1,3] = 1

    # Assemble 2x2x3
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))
    W_223 = np.zeros((6, 11))

    U_223[:, 0:7] = U_s
    V_223[0:4, 0:7] = V_s
    # V_223[4:6, 0:7] = 0 (already zero)
    W_223[0:4, 0:7] = W_s
    # W_223[4:6, 0:7] = 0 (already zero)

    U_223[:, 7:11] = U_221
    V_223[4:6, 7:11] = V_221
    W_223[4:6, 7:11] = W_221

    # Fill U, V, W
    term_offset = 0
    for i_block in range(2):
        for k_block in range(2):
            for j_block in range(2):
                # U
                u_rows = [(i_block*2+r)*4 + (j_block*2+c) for r in range(2) for c in range(2)]
                for r_u in range(4):
                    U[u_rows[r_u], term_offset:term_offset+11] = U_223[r_u, :]

                # V
                v_rows = [(j_block*2+r)*6 + (k_block*3+c) for r in range(2) for c in range(3)]
                for r_v in range(6):
                    V[v_rows[r_v], term_offset:term_offset+11] = V_223[r_v, :]

                # W
                w_rows = [(k_block*3+c)*4 + (i_block*2+r) for c in range(3) for r in range(2)]
                for r_w in range(6):
                    W[w_rows[r_w], term_offset:term_offset+11] = W_223[r_w, :]

                term_offset += 11

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
