"""Matrix multiplication 4x4x6 via block-strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
# Convention: C[i,j] = sum_k A[i,k]*B[k,j]
# We define U, V, W such that C[i,j] = sum_r U[i,r]*V[r,j]*W[r,j]... 
# Actually the tensor convention is:
# C[i,j] = sum_{r} (sum_k A[i,k] * U[k, r]) * (sum_l B[l, j] * V[l, r]) * W[r] ?
# No, the standard tensor decomposition for matrix multiplication T_{ijk} is:
# T_{ijk} = sum_r U_{ir} V_{kr} W_{jr}
# Where A is n x m, B is m x p, C is n x p.
# C_{ij} = sum_k A_{ik} B_{kj} = sum_r (sum_k A_{ik} U_{kr}) (sum_l B_{lj} V_{lr}) W_{jr}?
# Let's stick to the prompt's convention:
# U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
# The evaluator checks: C[i,j] = sum_k A[i,k] B[k,j]
# And the decomposition is:
# C[i,j] = sum_r W[k*n+i, r] * (sum_k A[i,k] U[i*m+k, r]) * (sum_l B[k,l] V[k*p+l, r])
# Wait, the prompt says:
# "output factor follows the official trace convention W[k*n+i,r]"
# This usually means the tensor T_{i,j,k} (for A_i, B_k, C_j) is decomposed as:
# T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} ?
# Let's look at the seed program to be sure.
# Seed:
# U[i*m+j, column] = 1
# V[j*p+k, column] = 1
# W[k*n+i, column] = 1
# Here i is row of A, j is col of A (row of B), k is col of B.
# So A is n x m (rows i, cols j). B is m x p (rows j, cols k). C is n x p (rows i, cols k).
# The term is A[i,j] * B[j,k].
# The seed sets U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
# This implies the contraction is:
# sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k] ?
# No, the standard bilinear form is:
# C[i,k] = sum_j A[i,j] B[j,k]
# = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
# Let's verify this matches the seed.
# If U[i*m+j, r] = delta_{j, j_r}, V[j*p+k, r] = delta_{k, k_r}, W[k*n+i, r] = delta_{i, i_r}
# Then term r contributes A[i, j_r] * B[j_r, k_r] * delta_{i, i_r} delta_{k, k_r}.
# Summing over r where i=i_r and k=k_r, we get sum_{j} A[i,j] B[j,k] if for every (i,k) there is a unique r with i_r=i, k_r=k and j_r=j?
# No, the seed iterates i, j, k. For each (i,j,k) it creates a rank-1 term.
# U[i*m+j, r] = 1. This picks out A[i,j].
# V[j*p+k, r] = 1. This picks out B[j,k].
# W[k*n+i, r] = 1. This picks out the position (i,k) in C.
# So C[i,k] = sum_r W[k*n+i, r] * (sum_j' A[i,j'] U[i*m+j', r]) * (sum_j' B[j',k] V[j'*p+k, r])
# For a specific r corresponding to (i,j,k), U[i*m+j, r]=1, others 0. So sum is A[i,j].
# V[j*p+k, r]=1, others 0. So sum is B[j,k].
# W[k*n+i, r]=1.
# Product is A[i,j] B[j,k].
# Summing over all r, for a fixed (i,k), the sum is over all j of A[i,j]B[j,k]. Correct.

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    
    # We want to decompose the 4x4x6 multiplication.
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x6) into 2x2 blocks of size 2x3: B11, B12, B21, B22
    # C (4x6) will have 2x2 blocks of size 2x3: C11, C12, C21, C22
    #
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    #
    # This is a 2x2 block multiplication where the "scalar" multiplication is actually
    # multiplication of 2x2 matrices by 2x3 matrices.
    # Let X be a 2x2 matrix, Y be a 2x3 matrix. X*Y is a 2x3 matrix.
    # We need to compute M = X * Y.
    # The rank of 2x2x3 multiplication is not immediately Strassen's 7.
    # However, we can view 2x2x3 as part of a larger structure or use specific decompositions.
    #
    # Alternative approach:
    # Use the known rank for 4x4x4 which is 49 (or 48? 49 is the best known for 4x4x4 over R? 
    # Actually, for 4x4x4, the rank is 49. Strassen gives 7^2=49 for 4x4x4 if we use 2x2x2 blocks?
    # Wait, 4x4x4 can be seen as 2x2 blocks of 2x2x2.
    # Rank(4x4x4) <= 4 * Rank(2x2x2) = 4 * 7 = 28? No, that's not how it works.
    # The block structure of 4x4x4 involves 8 multiplications of 2x2x2 tensors?
    # No, 4x4x4 tensor T_{ijk} with i,j,k in {1..4}.
    # If we group indices into pairs, we get 8 blocks of size 2x2x2.
    # But they are not independent.
    #
    # Let's look at the 4x4x6 case again.
    # We can embed 4x4x6 into 4x4x8? Or use the 4x4x4 rank and add 2 columns?
    #
    # Let's try a different strategy: Direct construction using Strassen on the 4x4x4 part and handling the extra 2 columns of B.
    # B is 4x6. Let B = [B_left (4x4), B_right (4x2)].
    # C = A * B = [A * B_left, A * B_right].
    # The first part is 4x4x4 multiplication. The second is 4x4x2.
    # These share the same A.
    #
    # Rank(4x4x4) is 49 (best known).
    # Rank(4x4x2) is 16 (schoolbook). Can we do better?
    # 4x4x2: A (4x4) * B (4x2).
    # This is equivalent to multiplying a 4x4 matrix by a 4x2 matrix.
    # We can view this as two 4x4x1 multiplications?
    # Or use Strassen?
    # 4x4x2 can be decomposed into blocks of 2x2x1?
    # 2x2x1: A (2x2) * B (2x1).
    # Rank(2x2x1) is 4 (schoolbook). Is it lower?
    # T_{ijk} for i,j in {1,2}, k in {1,2} (B has 1 col? No, B is 2x1, so k is row of B, j is col of A).
    # Let's stick to the definition: A (n x m), B (m x p).
    # For 4x4x2: n=4, m=4, p=2.
    # We can split A into 2x2 blocks, B into 2x1 blocks (vertically).
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B1], [B2]] (each 2x2? No, B is 4x2. So B1 is 2x2, B2 is 2x2? No, B is 4 rows, 2 cols.
    # B = [B_top (2x2); B_bottom (2x2)]
    # C = A * B = [A11 B_top + A12 B_bottom, ...]
    # C is 4x2. C = [C_top (2x2); C_bottom (2x2)]
    # C_top = A11 B_top + A12 B_bottom
    # C_bottom = A21 B_top + A22 B_bottom
    # This requires 4 multiplications of 2x2x2 matrices?
    # A11 B_top is 2x2 * 2x2.
    # So we have 4 instances of 2x2x2 multiplication.
    # Rank <= 4 * 7 = 28.
    #
    # So total rank <= Rank(4x4x4) + Rank(4x4x2) ?
    # But they share A. We can't just add ranks if they share inputs, unless we treat them as separate tensors.
    # The tensor for 4x4x6 is T_{i,j,k} where i is row of A, j is col of A (row of B), k is col of B.
    # k ranges from 1 to 6.
    # We can decompose the tensor T = T_1 + T_2 where T_1 corresponds to k=1..4 and T_2 to k=5..6.
    # T_1 is the 4x4x4 tensor. T_2 is the 4x4x2 tensor.
    # Since the supports are disjoint in the k-index, the rank of the sum is the sum of the ranks.
    # So Rank(4x4x6) <= Rank(4x4x4) + Rank(4x4x2).
    #
    # Best known Rank(4x4x4) = 49.
    # Best known Rank(4x4x2) = ?
    # 4x4x2 is equivalent to 4x4x1 + 4x4x1? No.
    # Let's calculate Rank(4x4x2) using block Strassen.
    # As derived above, it reduces to 4 multiplications of 2x2x2.
    # Rank(2x2x2) = 7.
    # So Rank(4x4x2) <= 4 * 7 = 28.
    # Total rank <= 49 + 28 = 77.
    #
    # Can we do better for 4x4x4?
    # The standard Strassen decomposition for 4x4x4 uses 49 multiplications.
    # There is a known decomposition of rank 48? No, 49 is the bound.
    #
    # Can we do better for 4x4x2?
    # Maybe 2x2x2 is not the best block size?
    # What if we use 4x4x2 directly?
    #
    # Let's implement the 49 + 28 = 77 rank decomposition.
    #
    # First, we need the Strassen decomposition for 2x2x2.
    # Then we assemble it for 4x4x4 and 4x4x2.
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # i = m1 + m4 - m5 + m7
    # j = m3 + m5
    # k = m2 + m4
    # l = m1 - m2 + m3 + m6
    
    # We need to express this in the form:
    # C[i,j] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    # Wait, the indices in U, V, W are flattened.
    # For 2x2x2: n=2, m=2, p=2.
    # U is 4x7, V is 4x7, W is 4x7.
    
    # Let's define the 7 rank-1 terms for 2x2x2.
    # Each term r has U_r, V_r, W_r vectors of length 4.
    # U_r[i*m+j] is the coefficient for A[i,j].
    # V_r[j*p+k] is the coefficient for B[j,k].
    # W_r[k*n+i] is the coefficient for C[i,k].
    
    # Term 1: m1 = (a+d)(e+h)
    # U1: a=1, d=1 -> U1[0]=1 (A[0,0]), U1[3]=1 (A[1,1])
    # V1: e=1, h=1 -> V1[0]=1 (B[0,0]), V1[3]=1 (B[1,1])
    # W1: i = m1, l = m1 -> W1[0]=1 (C[0,0]), W1[3]=1 (C[1,1])
    # Note: W index is k*n+i. C[0,0] -> k=0, i=0 -> 0. C[1,1] -> k=1, i=1 -> 3.
    
    # Term 2: m2 = (c+d)e
    # U2: c=1, d=1 -> U2[2]=1 (A[1,0]), U2[3]=1 (A[1,1])
    # V2: e=1 -> V2[0]=1 (B[0,0])
    # W2: k = m2 -> W2[2]=1 (C[1,0]) -> k=0, i=1 -> 2.
    
    # Term 3: m3 = a(f-h)
    # U3: a=1 -> U3[0]=1
    # V3: f=1, h=-1 -> V3[1]=1 (B[0,1]), V3[3]=-1 (B[1,1])
    # W3: j = m3, l = m3 -> W3[1]=1 (C[0,1]), W3[3]=1 (C[1,1])
    
    # Term 4: m4 = d(g-e)
    # U4: d=1 -> U4[3]=1
    # V4: g=1, e=-1 -> V4[2]=1 (B[1,0]), V4[0]=-1 (B[0,0])
    # W4: i = m4, k = m4 -> W4[0]=1 (C[0,0]), W4[2]=1 (C[1,0])
    
    # Term 5: m5 = (a+b)h
    # U5: a=1, b=1 -> U5[0]=1, U5[1]=1
    # V5: h=1 -> V5[3]=1
    # W5: i = -m5, j = m5 -> W5[0]=-1, W5[1]=1
    
    # Term 6: m6 = (c-a)(e+f)
    # U6: c=1, a=-1 -> U6[2]=1, U6[0]=-1
    # V6: e=1, f=1 -> V6[0]=1, V6[1]=1
    # W6: l = m6 -> W6[3]=1
    
    # Term 7: m7 = (b-d)(g+h)
    # U7: b=1, d=-1 -> U7[1]=1, U7[3]=-1
    # V7: g=1, h=1 -> V7[2]=1, V7[3]=1
    # W7: i = m7, j = -m7 -> W7[0]=1, W7[1]=-1
    
    # Let's verify Term 1:
    # (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # C[0,0] gets A[0,0]B[0,0] + A[1,1]B[1,1] (from m1, m4, m5, m7?)
    # Let's re-verify the Strassen formulas.
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    #
    # Let's check c11:
    # m1 = ae + ah + de + dh
    # m4 = dg - de
    # -m5 = -ah - bh
    # m7 = bg + bh - dg - dh
    # Sum: ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg. Correct.
    
    # Now, we construct the U, V, W matrices for the 2x2x2 case.
    # n=2, m=2, p=2.
    # U is 4x7, V is 4x7, W is 4x7.
    
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # Term 3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = -1
        
        return U, V, W
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Now assemble 4x4x4.
    # n=4, m=4, p=4.
    # U is 16x49, V is 16x49, W is 16x49.
    # We have 4 block multiplications of 2x2x2? No, 4x4x4 is composed of 8 2x2x2 blocks?
    # No, the tensor T_{ijk} for i,j,k in {1..4}.
    # We can view this as 2x2 blocks of 2x2x2 tensors.
    # Let i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2.
    # This is not a simple direct sum.
    #
    # Standard way to get 4x4x4 from 2x2x2:
    # Use Strassen's algorithm recursively?
    # Strassen for 4x4x4:
    # Split A, B, C into 2x2 blocks of size 2x2.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # We need to compute 8 matrix products of 2x2 matrices?
    # No, Strassen reduces the number of multiplications.
    # For 4x4x4, we treat the 2x2 blocks as "scalars".
    # Then we apply Strassen's 7-multiplication algorithm to these "scalars".
    # But the "scalars" are 2x2 matrices, and their multiplication is non-commutative.
    # Strassen's algorithm works for non-commutative rings if we are careful with the order.
    #
    # Let M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mk is a 2x2 matrix product.
    # So we need 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    #
    # Now we need to map this to the U, V, W matrices.
    # Each Mk is a 2x2x2 tensor decomposition.
    # Let's say Mk = sum_{r=1..7} U^{(k)}_r V^{(k)}_r W^{(k)}_r.
    # The total decomposition is the sum of all Mk decompositions.
    #
    # We need to construct U, V, W for the 4x4x4 case.
    # U is 16x49.
    # The rows of U correspond to (i,j) in A. i,j in {0,1,2,3}.
    # We can index them as i*m+j = i*4+j.
    #
    # For M1 = (A11+A22)(B11+B22):
    # A11 is A[0:2, 0:2], A22 is A[2:4, 2:4].
    # B11 is B[0:2, 0:2], B22 is B[2:4, 2:4].
    #
    # The term (A11+A22) means we sum A[0,0]+A[2,2], A[0,1]+A[2,3], etc.
    # So for the U part of M1, we need to add the U vectors of the 2x2x2 decomposition
    # for A11 and A22.
    #
    # Let's define a function to place a 2x2x2 decomposition into the 4x4x4 U, V, W matrices.
    # The 2x2x2 decomposition has U2 (4x7), V2 (4x7), W2 (4x7).
    # We need to map the 2x2 indices to the 4x4 indices.
    #
    # For A11: i in {0,1}, j in {0,1}. Map to i*4+j.
    # For A22: i in {2,3}, j in {2,3}. Map to i*4+j.
    #
    # We will create a helper to add a scaled 2x2x2 decomposition to the global U, V, W.
    
    def add_2x2x2_to_4x4x4(U, V, W, U2, V2, W2, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C, scale=1):
        # row_offset_A, col_offset_A: top-left corner of the 2x2 block in A
        # row_offset_B, col_offset_B: top-left corner of the 2x2 block in B
        # row_offset_C, col_offset_C: top-left corner of the 2x2 block in C
        # Note: A is 4x4, so row_offset_A is in {0,2}, col_offset_A is in {0,2}.
        
        for r in range(7):
            # U part
            for i2 in range(2):
                for j2 in range(2):
                    val = U2[i2*2+j2, r]
                    if val != 0:
                        i = row_offset_A + i2
                        j = col_offset_A + j2
                        U[i*4+j, r] += scale * val
            
            # V part
            for j2 in range(2):
                for k2 in range(2):
                    val = V2[j2*2+k2, r]
                    if val != 0:
                        j = row_offset_B + j2
                        k = col_offset_B + k2
                        V[j*4+k, r] += scale * val
            
            # W part
            for k2 in range(2):
                for i2 in range(2):
                    val = W2[k2*2+i2, r]
                    if val != 0:
                        k = col_offset_C + k2
                        i = row_offset_C + i2
                        W[k*4+i, r] += scale * val
    
    # Now, let's build the 4x4x4 decomposition.
    # We have 7 terms M1..M7.
    # Each Mk is a 2x2x2 multiplication.
    # We need to determine the offsets for A, B, C for each Mk.
    
    # M1 = (A11+A22)(B11+B22)
    # This is a sum of two products: A11*B11 + A22*B22.
    # So we add the 2x2x2 decomposition for A11*B11 and A22*B22.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # M1: A11*B11 + A22*B22
    # A11: row 0, col 0. B11: row 0, col 0. C11: row 0, col 0.
    add_2x2x2_to_4x4x4(U4, V4, W4, U2, V2, W2, 0, 0, 0, 0, 0, 0)
    # A22: row 2, col 2. B22: row 2, col 2. C11: row 0, col 0.
    add_2x2x2_to_4x4x4(U4, V4, W4, U2, V2, W2, 2, 2, 2, 2, 0, 0)
    
    # M2: (A21+A22)B11
    # A21: row 2, col 0. A22: row 2, col 2.
    # B11: row 0, col 0.
    # C21: row 2, col 0.
    add_2x2x2_to_4x4x4(U4, V4, W4, U2, V2, W2, 2, 0, 0, 0, 2, 0)
    add_2x2x2_to_4x4x4(U4, V4, W4, U2, V2, W2, 2, 2, 0, 0, 2, 0)
    
    # M3: A11(B12-B22)
    # A11: row 0, col 0.
    # B12: row 0, col 2. B22: row 2, col 2.
    # C12: row 0, col 2.
    add_2x2x2_to_4x4x4(U4, V4, W4, U2, V2, W2, 0, 0, 0, 2, 0, 2)
    add_2x2x2_to_4x4x4(U4, V4, W4, U2, V2, W2, 0, 0, 2, 2, 0, 2, scale=-1)
    
    # M4: A22(B21-B11)
    # A22: row 2, col 2.
    # B21: row 2, col 0. B11: row 0, col 0.
    # C11: row 0, col 0. (Wait, M4 contributes to C11 and C21)
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 + M4
    # So M4 contributes to C11 (row 0, col 0) and C21 (row 2, col 0).
    # But M4 is a single 2x2x2 multiplication. Its output is a 2x2 matrix.
    # Where does it go?
    # In Strassen, M4 = A22(B21-B11). This is a 2x2 matrix.
    # It contributes to C11 and C21?
    # No, C11 is the top-left 2x2 block. C21 is the bottom-left.
    # M4 is added to C11 and C21.
    # But M4 is a single matrix product. How can it contribute to two different blocks?
    # It can't, unless the blocks are the same size and we are summing them.
    # Wait, C11 and C21 are different blocks.
    # M4 is a 2x2 matrix. It must be added to C11 (top-left) and C21 (bottom-left)?
    # No, that would mean C11 and C21 are both 2x2, and M4 is added to both.
    # But M4 is a single 2x2 matrix. It can't be added to two different 2x2 blocks unless they are the same block.
    # I think I made a mistake in the Strassen formulas.
    #
    # Let's re-check Strassen for 4x4x4.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # M4 appears in C11 and C21.
    # This means M4 is a 2x2 matrix that is added to C11 and also to C21.
    # But C11 and C21 are different 2x2 blocks of C.
    # This implies that M4 is not a single 2x2 matrix product, but rather its result is used in two places.
    # But M4 = A22(B21-B11) is a single 2x2 matrix product.
    # How can its result be added to two different blocks?
    # It can't, unless the blocks are the same.
    #
    # I think the issue is that Strassen's algorithm for 4x4x4 is not just 7 multiplications of 2x2 matrices.
    # It is 7 multiplications of 2x2 matrices, but the results are combined in a specific way.
    # M4 is a 2x2 matrix. It is added to C11 and C21.
    # This means C11 and C21 both get the same 2x2 matrix M4 added to them.
    # But C11 and C21 are different blocks.
    # This is possible if C is not just a 2x2 block matrix, but the blocks are overlapping? No.
    #
    # Actually, I think the standard Strassen decomposition for 4x4x4 uses 49 multiplications of scalars, not 7 multiplications of 2x2 matrices.
    # The 7 multiplications are of 2x2 matrices, but each 2x2 matrix multiplication requires 7 scalar multiplications.
    # So total scalar multiplications = 7 * 7 = 49.
    #
    # The key is that M4 is a 2x2 matrix, and it is added to C11 and C21.
    # This means that the 2x2 matrix M4 is added to the top-left block and the bottom-left block of C.
    # This is fine, as long as we add the same 2x2 matrix to both blocks.
    #
    # So, for M4, we need to add the 2x2x2 decomposition for A22*(B21-B11) to both C11 and C21.
    # But A22*(B21-B11) is a single 2x2 matrix product.
    # So we add the 2x2x2 decomposition once, but with the output going to both C11 and C21.
    #
    # How do we do this in the U, V, W matrices?
    # The W matrix has entries for C.
    # For M4, the W part should have entries for both C11 and C21.
    #
    # Let's modify the add_2x2x2_to_4x4x4 function to allow multiple output blocks.
    
    def add_2x2x2_to_4x4x4_multi(U, V, W, U2, V2, W2, row_offset_A, col_offset_A, row_offset_B, col_offset_B, output_blocks, scale=1):
        # output_blocks is a list of (row_offset_C, col_offset_C)
        for r in range(7):
            # U part
            for i2 in range(2):
                for j2 in range(2):
                    val = U2[i2*2+j2, r]
                    if val != 0:
                        i = row_offset_A + i2
                        j = col_offset_A + j2
                        U[i*4+j, r] += scale * val
            
            # V part
            for j2 in range(2):
                for k2 in range(2):
                    val = V2[j2*2+k2, r]
                    if val != 0:
                        j = row_offset_B + j2
                        k = col_offset_B + k2
                        V[j*4+k, r] += scale * val
            
            # W part
            for k2 in range(2):
                for i2 in range(2):
                    val = W2[k2*2+i2, r]
                    if val != 0:
                        for row_offset_C, col_offset_C in output_blocks:
                            k = col_offset_C + k2
                            i = row_offset_C + i2
                            W[k*4+i, r] += scale * val
    
    # Now, let's rebuild the 4x4x4 decomposition with the corrected logic.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # M1: (A11+A22)(B11+B22) -> A11*B11 + A22*B22
    # Output: C11
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 0, 0, 0, 0, [(0, 0)])
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 2, 2, 2, 2, [(0, 0)])
    
    # M2: (A21+A22)B11 -> A21*B11 + A22*B11
    # Output: C21
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 2, 0, 0, 0, [(2, 0)])
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 2, 2, 0, 0, [(2, 0)])
    
    # M3: A11(B12-B22) -> A11*B12 - A11*B22
    # Output: C12
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 0, 0, 0, 2, [(0, 2)])
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 0, 0, 2, 2, [(0, 2)], scale=-1)
    
    # M4: A22(B21-B11) -> A22*B21 - A22*B11
    # Output: C11 and C21
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 2, 2, 2, 0, [(0, 0), (2, 0)])
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 2, 2, 0, 0, [(0, 0), (2, 0)], scale=-1)
    
    # M5: (A11+A12)B22 -> A11*B22 + A12*B22
    # Output: C11 (with -1) and C12
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 0, 0, 2, 2, [(0, 0), (0, 2)])
    add_2x2x2_to_4x4x4_multi(U4, V4, W4, U2, V2, W2, 0, 2, 2, 2, [(0, 0), (0, 2)])
    # Wait, M5 is subtracted from C11 and added to C12.
    # So we need to add M5 to C12 and subtract M5 from C11.
    # My add function adds to all output blocks with the same scale.
    # I need to handle different scales for different output blocks.
    
    # Let's modify the function to accept a list of (row_offset_C, col_offset_C, scale)
    
    def add_2x2x2_to_4x4x4_multi_v2(U, V, W, U2, V2, W2, row_offset_A, col_offset_A, row_offset_B, col_offset_B, output_blocks, scale=1):
        # output_blocks is a list of (row_offset_C, col_offset_C, scale)
        for r in range(7):
            # U part
            for i2 in range(2):
                for j2 in range(2):
                    val = U2[i2*2+j2, r]
                    if val != 0:
                        i = row_offset_A + i2
                        j = col_offset_A + j2
                        U[i*4+j, r] += scale * val
            
            # V part
            for j2 in range(2):
                for k2 in range(2):
                    val = V2[j2*2+k2, r]
                    if val != 0:
                        j = row_offset_B + j2
                        k = col_offset_B + k2
                        V[j*4+k, r] += scale * val
            
            # W part
            for k2 in range(2):
                for i2 in range(2):
                    val = W2[k2*2+i2, r]
                    if val != 0:
                        for row_offset_C, col_offset_C, s in output_blocks:
                            k = col_offset_C + k2
                            i = row_offset_C + i2
                            W[k*4+i, r] += scale * s * val
    
    # Now, let's rebuild again.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # M1: A11*B11 + A22*B22 -> C11
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 0, 0, 0, [(0, 0, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 2, 2, 2, [(0, 0, 1)])
    
    # M2: A21*B11 + A22*B11 -> C21
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 0, 0, 0, [(2, 0, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 2, 0, 0, [(2, 0, 1)])
    
    # M3: A11*B12 - A11*B22 -> C12
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 0, 0, 2, [(0, 2, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 0, 2, 2, [(0, 2, -1)])
    
    # M4: A22*B21 - A22*B11 -> C11 (with +1) and C21 (with +1)
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 2, 2, 0, [(0, 0, 1), (2, 0, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 2, 0, 0, [(0, 0, -1), (2, 0, -1)])
    
    # M5: A11*B22 + A12*B22 -> C11 (with -1) and C12 (with +1)
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 0, 2, 2, [(0, 0, -1), (0, 2, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 2, 2, 2, [(0, 0, -1), (0, 2, 1)])
    
    # M6: (A21-A11)(B11+B12) -> A21*B11 + A21*B12 - A11*B11 - A11*B12
    # Output: C22
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 0, 0, 0, [(2, 2, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 0, 0, 2, [(2, 2, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 0, 0, 0, [(2, 2, -1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 0, 0, 2, [(2, 2, -1)])
    
    # M7: (A12-A22)(B21+B22) -> A12*B21 + A12*B22 - A22*B21 - A22*B22
    # Output: C11 (with +1) and C22 (with -1)
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 2, 2, 0, [(0, 0, 1), (2, 2, -1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 0, 2, 2, 2, [(0, 0, 1), (2, 2, -1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 2, 2, 0, [(0, 0, -1), (2, 2, 1)])
    add_2x2x2_to_4x4x4_multi_v2(U4, V4, W4, U2, V2, W2, 2, 2, 2, 2, [(0, 0, -1), (2, 2, 1)])
    
    # Now we have the 4x4x4 decomposition with rank 49.
    # Next, we need the 4x4x2 decomposition with rank 28.
    #
    # 4x4x2: A (4x4) * B (4x2).
    # B is 4x2. We can split B into B_top (2x2) and B_bottom (2x2).
    # A is split into A11, A12, A21, A22 (each 2x2).
    # C is 4x2. C_top (2x2), C_bottom (2x2).
    # C_top = A11*B_top + A12*B_bottom
    # C_bottom = A21*B_top + A22*B_bottom
    #
    # This requires 4 multiplications of 2x2x2 matrices.
    # Rank = 4 * 7 = 28.
    
    U2x2 = np.zeros((16, 28))
    V2x2 = np.zeros((8, 28))  # B is 4x2, so m*p = 4*2 = 8
    W2x2 = np.zeros((8, 28))  # C is 4x2, so p*n = 2*4 = 8
    
    # M1: A11*B_top -> C_top
    add_2x2x2_to_4x4x2(U2x2, V2x2, W2x2, U2, V2, W2, 0, 0, 0, 0, 0, 0)
    
    # M2: A12*B_bottom -> C_top
    add_2x2x2_to_4x4x2(U2x2, V2x2, W2x2, U2, V2, W2, 0, 2, 2, 0, 0, 0)
    
    # M3: A21*B_top -> C_bottom
    add_2x2x2_to_4x4x2(U2x2, V2x2, W2x2, U2, V2, W2, 2, 0, 0, 0, 2, 0)
    
    # M4: A22*B_bottom -> C_bottom
    add_2x2x2_to_4x4x2(U2x2, V2x2, W2x2, U2, V2, W2, 2, 2, 2, 0, 2, 0)
    
    # Now we combine the 4x4x4 and 4x4x2 decompositions.
    # The total rank is 49 + 28 = 77.
    # U is 16x77, V is 24x77 (16+8), W is 24x77 (16+8).
    # Wait, V for 4x4x4 is 16x49 (m*p = 4*4 = 16).
    # V for 4x4x2 is 8x28 (m*p = 4*2 = 8).
    # But the V matrix for the combined 4x4x6 should be 24x77 (m*p = 4*6 = 24).
    # The first 16 rows of V correspond to B (4x4), and the last 8 rows correspond to B (4x2).
    # But B is 4x6, so it's a single matrix.
    # The 4x4 part is B[:, 0:4], and the 4x2 part is B[:, 4:6].
    # So we need to map the V indices correctly.
    
    U_total = np.zeros((16, 77))
    V_total = np.zeros((24, 77))
    W_total = np.zeros((24, 77))
    
    # Copy 4x4x4 part
    U_total[:, 0:49] = U4
    V_total[0:16, 0:49] = V4
    W_total[0:16, 0:49] = W4
    
    # Copy 4x4x2 part
    U_total[:, 49:77] = U2x2
    V_total[16:24, 49:77] = V2x2
    W_total[16:24, 49:77] = W2x2
    
    # Now we need to verify the decomposition.
    # Let's test it with a random matrix.
    
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 6)
    C_test = A_test @ B_test
    
    # Compute C from the decomposition
    C_decomp = np.zeros((4, 6))
    for r in range(77):
        u_vec = U_total[:, r]
        v_vec = V_total[:, r]
        w_vec = W_total[:, r]
        
        # u_vec is 16x1, v_vec is 24x1, w_vec is 24x1
        # u_vec[i*4+j] is the coefficient for A[i,j]
        # v_vec[j*6+k] is the coefficient for B[j,k]
        # w_vec[k*4+i] is the coefficient for C[i,k]
        
        a_val = 0
        for i in range(4):
            for j in range(4):
                a_val += A_test[i, j] * u_vec[i*4+j]
        
        b_val = 0
        for j in range(4):
            for k in range(6):
                b_val += B_test[j, k] * v_vec[j*6+k]
        
        for i in range(4):
            for k in range(6):
                C_decomp[i, k] += w_vec[k*4+i] * a_val * b_val
    
    if not np.allclose(C_test, C_decomp, atol=1e-10):
        # If the decomposition is incorrect, fall back to schoolbook
        rank = 96
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(24)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i * 4 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 4 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Helper function for 4x4x2
def add_2x2x2_to_4x4x2(U, V, W, U2, V2, W2, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C):
    for r in range(7):
        for i2 in range(2):
            for j2 in range(2):
                val = U2[i2*2+j2, r]
                if val != 0:
                    i = row_offset_A + i2
                    j = col_offset_A + j2
                    U[i*4+j, r] += val
        
        for j2 in range(2):
            for k2 in range(2):
                val = V2[j2*2+k2, r]
                if val != 0:
                    j = row_offset_B + j2
                    k = col_offset_B + k2
                    V[j*2+k, r] += val
        
        for k2 in range(2):
            for i2 in range(2):
                val = W2[k2*2+i2, r]
                if val != 0:
                    k = col_offset_C + k2
                    i = row_offset_C + i2
                    W[k*4+i, r] += val
