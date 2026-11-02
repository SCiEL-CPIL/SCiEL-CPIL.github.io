"""Rewritten program for matrix multiplication 4x4x6 using structured block decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We want to decompose the tensor T[i,j,k] = delta(i,j,k) for matrix mult (n x m) * (m x p) -> (n x p)
    # Output convention: W[k*n + i, r]
    # U[i*m + j, r], V[j*p + k, r]
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k) * (A[i,j] * B[j,k])
    # Wait, the tensor for matrix multiplication is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The bilinear form is sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k} = sum_r (sum_{i,j} U_{i,j,r} A_{i,j}) (sum_{j,k} V_{j,k,r} B_{j,k}) (sum_{i,k} W_{i,k,r} C_{i,k})
    # So we need: sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} = delta_{i,k} ... no.
    # Actually, the standard tensor decomposition for matrix multiplication <n,m,p> is:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n} (or similar)
    # And the condition is: for all A, B, the product C = A*B satisfies
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_r (sum_{i',j'} U_{i',j',r} A_{i',j'}) (sum_{j',k'} V_{j',k',r} B_{j',k'}) * W_{i,k,r} ?
    # No, the trace convention given is W[k*n+i, r].
    # Let's stick to the definition:
    # The tensor T has entries T[i, j, k] = 1 if ... no, it's the structure tensor.
    # T_{i,j,k} = 1 if we are computing C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} is non-zero only when the indices match the multiplication pattern.
    # Specifically, T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}.
    # Wait, e_{i,j} is the basis for A (size n*m), e_{j,k} for B (size m*p), e_{i,k} for C (size n*p).
    # The output factor W has index k*n + i. So W_{k*n+i, r} corresponds to e_{i,k} in the C space?
    # Let's check the seed code:
    # U[i*m + j][column] = 1
    # V[j*p + k][column] = 1
    # W[k*n + i][column] = 1
    # This implies the term is U_{i,j} * V_{j,k} * W_{i,k} (with W indexed as k*n+i).
    # Sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r} should equal 1 if the indices form a valid "path" i->j->k?
    # No, for matrix multiplication, we need:
    # sum_j A_{i,j} B_{j,k} = C_{i,k}
    # This is a trilinear form. The tensor T is such that <T, A⊗B⊗C> = trace(A B C^T) ?
    # No, it's simply that the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # So the tensor T_{i,j,k} = 1 for all i,j,k? No, that would be a complete contraction.
    # The tensor is T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the indices are linked.
    # Specifically, T_{i,j,k} = 1 if we are summing over j.
    # Actually, the standard definition: The tensor for <n,m,p> is T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k ? No.
    # It is T = sum_{i,j,k} (e_i in A-space) ⊗ (e_j in B-space) ⊗ (e_k in C-space) ??
    # Let's look at the seed code again.
    # It sets U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # This corresponds to the term e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C.
    # Summing over r, we get sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
    # For the schoolbook, each (i,j,k) has its own r.
    # So for a fixed (i,j,k), U_{i,j,r}=1, V_{j,k,r}=1, W_{i,k,r}=1.
    # For any other (i',j',k'), the product is 0.
    # So the sum is 1 if (i,j,k) is the specific tuple, else 0.
    # This means the tensor T is simply the sum of all basis tensors e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}.
    # Wait, is that the matrix multiplication tensor?
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # If we define the tensor T such that T(A,B) = C, then T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # This coefficient is 1 for all i,j,k.
    # So T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C.
    # Yes, this is the correct tensor.
    
    # Strategy: Block decomposition.
    # n=4, m=4, p=6.
    # Split into 2x2 blocks of size 2x2 for A and B?
    # A is 4x4. Split into four 2x2 blocks: A_11, A_12, A_21, A_22.
    # B is 4x6. Split into two 2x6 blocks vertically? Or 4x3?
    # Let's split m=4 into 2+2.
    # Let's split n=4 into 2+2.
    # Let's split p=6 into 3+3.
    
    # A = [A11 A12; A21 A22] where Aij are 2x2.
    # B = [B1; B2] where B1, B2 are 2x6.
    # C = A B = [A11 B1 + A12 B2; A21 B1 + A22 B2].
    # C1 = A11 B1 + A12 B2 (2x6)
    # C2 = A21 B1 + A22 B2 (2x6)
    
    # Each term is a 2x2 * 2x6 multiplication.
    # We need rank for <2,2,6>.
    # Known rank for <2,2,2> is 7.
    # For <2,2,6>, we can view it as 3 copies of <2,2,2> plus cross terms?
    # Or simply use the schoolbook for <2,2,6> which is 2*2*6 = 24.
    # Can we do better?
    # Rank of <2,2,p> is known to be at most 2p-1? No.
    # Actually, <2,2,2> is 7.
    # <2,2,3> is ?
    # Let's try to construct <2,2,6> with rank < 24.
    
    # Alternative approach: Use the known decomposition for <4,4,4> and extend.
    # Rank of <4,4,4> is 49 (Landsberg).
    # We have <4,4,6>.
    # We can write <4,4,6> as <4,4,4> + <4,4,2>.
    # Rank(<4,4,4>) = 49.
    # Rank(<4,4,2>) = ?
    # <4,4,2> can be seen as 2 copies of <4,4,1>? No.
    # <4,4,2> is multiplying 4x4 by 4x2.
    # This is equivalent to two <4,4,1> multiplications? No, <4,4,1> is just dot product, rank 4? No, <4,4,1> is sum_{j} A_{i,j} B_{j,1}.
    # The tensor for <4,4,1> is sum_{i,j} e_{i,j} ⊗ e_{j,1} ⊗ e_{i,1}.
    # This is a "slice" of the multiplication tensor.
    # Rank of <n,m,1> is n*m? No, it's n+m-1?
    # Actually, <n,m,1> is the same as <1,m,n> by permutation?
    # No, <n,m,p> is symmetric under cyclic permutation? No, it's symmetric under any permutation of n,m,p?
    # Yes, the matrix multiplication tensor is symmetric under permutation of dimensions (with appropriate index relabeling).
    # So <4,4,2> is isomorphic to <2,4,4>.
    # Rank of <2,4,4> ?
    # <2,2,2> is 7.
    # <2,4,4> can be decomposed into 4 copies of <2,2,2>?
    # If we split the 4x4 into 2x2 blocks, we get 4 terms of <2,2,2>.
    # So rank <= 4*7 = 28.
    # So total rank <= 49 + 28 = 77.
    # Schoolbook is 4*4*6 = 96.
    # 77 is an improvement.
    
    # Can we do better?
    # What is the rank of <4,4,6>?
    # Lower bound: max(4*4, 4*6, 4*6) = 24? No, lower bound is much higher.
    # Strassen's algorithm for 4x4x4 uses rank 49.
    # For 4x4x6, maybe we can use a mixed approach.
    
    # Let's try to implement the 49 + 28 = 77 decomposition.
    # Step 1: Get a rank-7 decomposition for <2,2,2>.
    # Step 2: Construct <4,4,4> from 49 rank-1 terms? No, Strassen's 4x4x4 is a specific 49-term decomposition.
    # Step 3: Construct <4,4,2> from 28 rank-1 terms (4 copies of <2,2,2>).
    
    # Wait, the task asks for a "rank-minimal" decomposition.
    # The current best known for 4x4x4 is 49.
    # For 4x4x6, is there a known rank?
    # If we use Strassen for the 4x4x4 part, and schoolbook for the remaining 4x4x2 part, we get 49 + 32 = 81.
    # But we can use <2,2,2> blocks for the 4x4x2 part.
    # 4x4x2 = 2x(2x4x2) = 2x(2x(2x2x2))?
    # Let's be precise.
    # A (4x4) * B (4x2) -> C (4x2).
    # Split A into 2x2 blocks: A11, A12, A21, A22.
    # Split B into 2x2 blocks: B1, B2.
    # C1 = A11 B1 + A12 B2.
    # C2 = A21 B1 + A22 B2.
    # Each term is <2,2,2>.
    # So we need 4 decompositions of <2,2,2>.
    # If we use the rank-7 decomposition for each, total rank = 4*7 = 28.
    # So total rank = 49 + 28 = 77.
    
    # Is there a better way?
    # What if we use Strassen's algorithm for <4,4,4> and <4,4,2> is just part of it?
    # No, they are disjoint in the p-dimension.
    
    # Let's try to find a rank-7 decomposition for <2,2,2>.
    # The standard Strassen decomposition for 2x2x2 has rank 7.
    # U, V, W are 4x7 matrices.
    # We need to define them explicitly.
    
    # Strassen's 7 multiplications for 2x2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]].
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to U, V, W matrices of size 4x7.
    # U corresponds to A coefficients, V to B coefficients, W to C coefficients.
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # For n=m=p=2:
    # U is 4x7, V is 4x7, W is 4x7.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3.
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3.
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3. (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    # Let's define the 7 terms.
    # Term 1: (a+d)(e+h) -> C11, C22
    # a=U[0,0], d=U[3,0], e=V[0,0], h=V[3,0]
    # C11=W[0,0]=1, C22=W[3,0]=1
    # Term 2: (c+d)(e) -> C21, C22
    # c=U[2,1], d=U[3,1], e=V[0,1]
    # C21=W[2,1]=1, C22=W[3,1]=-1
    # Term 3: (a)(f-h) -> C12, C22
    # a=U[0,2], f=V[1,2], h=V[3,2]
    # C12=W[1,2]=1, C22=W[3,2]=1
    # Term 4: (d)(g-c) -> C11, C21
    # d=U[3,3], g=V[2,3], c=V[0,3]
    # C11=W[0,3]=1, C21=W[2,3]=1
    # Term 5: (a+b)(h) -> C11, C12
    # a=U[0,4], b=U[1,4], h=V[3,4]
    # C11=W[0,4]=-1, C12=W[1,4]=1
    # Term 6: (c-a)(e+f) -> C22
    # c=U[2,5], a=U[0,5], e=V[0,5], f=V[1,5]
    # C22=W[3,5]=1
    # Term 7: (b-d)(g+h) -> C11
    # b=U[1,6], d=U[3,6], g=V[2,6], h=V[3,6]
    # C11=W[0,6]=1
    
    # Let's verify Term 1:
    # U[0,0]=1, U[3,0]=1
    # V[0,0]=1, V[3,0]=1
    # W[0,0]=1, W[3,0]=1
    # Product: (a+d)(e+h)(C11 + C22)
    # This contributes to C11 and C22. Correct.
    
    # Let's build the U, V, W for <2,2,2>.
    # This is a helper function.
    
    def strassen_2x2x2():
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Term 0: (a+d)(e+h) -> C11, C22
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: (c+d)(e) -> C21, C22
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # Term 2: (a)(f-h) -> C12, C22
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 3: (d)(g-c) -> C11, C21
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 4: (a+b)(h) -> C11, C12
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5: (c-a)(e+f) -> C22
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: (b-d)(g+h) -> C11
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Now we need to construct <4,4,4> using Strassen's algorithm.
    # Strassen's algorithm for 4x4x4 uses 49 multiplications.
    # This is a recursive application of the 2x2x2 algorithm.
    # A 4x4 matrix is split into four 2x2 blocks.
    # The 7 multiplications of Strassen are applied to the 2x2 blocks.
    # Each 2x2 block multiplication has rank 7.
    # So we have 7 "sub-multiplications", each of which is a <2,2,2> problem.
    # But the inputs to these sub-multiplications are linear combinations of the 2x2 blocks of A and B.
    # And the outputs are linear combinations of the 2x2 blocks of C.
    # So we can construct the global U, V, W by "expanding" each of the 7 Strassen terms into 7 rank-1 terms for the <2,2,2> sub-problems.
    # Total rank = 7 * 7 = 49.
    
    # Let's define the Strassen decomposition for 4x4x4.
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22].
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-A11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will create a function that takes a <2,2,2> decomposition (U2, V2, W2) and a "pattern" of how the 2x2 blocks are combined, and produces the corresponding rows in the global U, V, W.
    
    # For the 4x4x4 case, we have 7 Strassen terms.
    # Each term involves linear combinations of the 4 blocks of A and B.
    # Let's define the coefficients for each Strassen term.
    # A blocks: 11, 12, 21, 22. Indices: 0, 1, 2, 3.
    # B blocks: 11, 12, 21, 22. Indices: 0, 1, 2, 3.
    # C blocks: 11, 12, 21, 22. Indices: 0, 1, 2, 3.
    
    # We will build the global U, V, W for 4x4x4 by iterating over the 7 Strassen terms.
    # For each Strassen term, we have a set of coefficients for A-blocks and B-blocks.
    # We then use the <2,2,2> decomposition to expand this into 7 rank-1 terms.
    # Each rank-1 term in the <2,2,2> decomposition has coefficients for the 4 elements of the 2x2 A-block, B-block, and C-block.
    # We need to map these to the global indices.
    
    # Global U index: i*m + j, where i,j in 0..3.
    # A-block 11 corresponds to i in 0..1, j in 0..1.
    # A-block 12 corresponds to i in 0..1, j in 2..3.
    # A-block 21 corresponds to i in 2..3, j in 0..1.
    # A-block 22 corresponds to i in 2..3, j in 2..3.
    
    # Let's define the block-to-index mapping.
    # Block index b (0..3) -> (i_offset, j_offset).
    # 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2).
    
    # For a given Strassen term s (0..6), we have coefficients alpha_s[b] for A-blocks and beta_s[b] for B-blocks.
    # And gamma_s[c] for C-blocks.
    # Actually, the Strassen formula is:
    # M_s = (sum_b alpha_s[b] A_b) * (sum_b beta_s[b] B_b)
    # C_c = sum_s gamma_s[c] M_s
    # So the tensor term is (sum_b alpha_s[b] A_b) (sum_b beta_s[b] B_b) (sum_c gamma_s[c] C_c).
    # This is a rank-1 tensor in the space of 4x4x4? No, it's a product of linear forms.
    # But A_b is a 2x2 matrix, so A_b is a vector in R^4.
    # So (sum_b alpha_s[b] A_b) is a vector in R^4 (for a specific block) ? No, it's a vector in R^16 (for the whole 4x4 matrix).
    # Specifically, it's a vector where the entries corresponding to block b are alpha_s[b] * (entries of A_b).
    # So the linear form on A is: sum_{i,j} (sum_b alpha_s[b] delta_{i in block_b(i), j in block_b(j)}) A_{i,j}.
    # This is a vector in R^16.
    # Similarly for B and C.
    # So each Strassen term s gives a rank-1 tensor in R^16 x R^16 x R^16.
    # But we need to decompose this rank-1 tensor into rank-1 tensors of the form u ⊗ v ⊗ w where u, v, w are in R^16?
    # No, the Strassen term is already a product of three linear forms.
    # L_A(s) * L_B(s) * L_C(s).
    # This is exactly a rank-1 tensor!
    # So we can just use these 7 linear forms directly?
    # Wait, L_A(s) is a linear combination of the 4 blocks, each of which is a 2x2 matrix.
    # So L_A(s) is a vector in R^16.
    # L_B(s) is a vector in R^16.
    # L_C(s) is a vector in R^16.
    # So the term is u_s ⊗ v_s ⊗ w_s where u_s, v_s, w_s are these vectors.
    # So the rank is 7?
    # No, this is for the 4x4x4 multiplication where the blocks are treated as scalars.
    # But the blocks are 2x2 matrices, so the multiplication of blocks is not scalar multiplication.
    # M_s = (sum alpha A_b) * (sum beta B_b) is a 2x2 matrix product.
    # So we cannot treat it as a scalar product.
    # We need to decompose the 2x2 matrix product into rank-1 terms.
    # So for each Strassen term s, we have a 2x2x2 multiplication problem.
    # We use the rank-7 decomposition for <2,2,2> to expand it.
    # So total rank = 7 * 7 = 49.
    
    # Let's implement this.
    # We define the coefficients for the 7 Strassen terms.
    # alpha_s[b], beta_s[b], gamma_s[c].
    
    # Term 0: (A11+A22)(B11+B22) -> C11, C22
    # alpha: [1, 0, 0, 1], beta: [1, 0, 0, 1], gamma: [1, 0, 0, 1]
    # Term 1: (A21+A22)B11 -> C21, C22
    # alpha: [0, 0, 1, 1], beta: [1, 0, 0, 0], gamma: [0, 0, 1, -1]
    # Term 2: A11(B12-B22) -> C12, C22
    # alpha: [1, 0, 0, 0], beta: [0, 1, 0, -1], gamma: [0, 1, 0, 1]
    # Term 3: A22(B21-A11) -> C11, C21
    # alpha: [0, 0, 0, 1], beta: [-1, 0, 1, 0], gamma: [1, 0, 1, 0]
    # Term 4: (A11+A12)B22 -> C11, C12
    # alpha: [1, 1, 0, 0], beta: [0, 0, 0, 1], gamma: [-1, 1, 0, 0]
    # Term 5: (A21-A11)(B11+B12) -> C22
    # alpha: [-1, 0, 1, 0], beta: [1, 1, 0, 0], gamma: [0, 0, 0, 1]
    # Term 6: (A12-A22)(B21+B22) -> C11
    # alpha: [0, 1, 0, -1], beta: [0, 0, 1, 1], gamma: [1, 0, 0, 0]
    
    # Let's verify Term 0:
    # M0 = (A11+A22)(B11+B22)
    # C11 += M0, C22 += M0.
    # gamma: [1, 0, 0, 1]. Correct.
    
    # Now, for each term s, we have a 2x2x2 multiplication problem.
    # The "A" matrix for this sub-problem is sum_b alpha_s[b] A_b.
    # This is a 2x2 matrix.
    # Similarly for B and C.
    # We use the <2,2,2> decomposition (U2, V2, W2) to expand this.
    # U2 is 4x7, V2 is 4x7, W2 is 4x7.
    # U2[i2, r2] is the coefficient of A_{i2} in the r2-th rank-1 term of the sub-problem.
    # A_{i2} is an element of the 2x2 matrix.
    # The 2x2 matrix is sum_b alpha_s[b] A_b.
    # So the coefficient of A_{i,j} (global) in the r2-th term is:
    # sum_b alpha_s[b] * delta_{i,j in block_b} * U2[i2, r2]
    # where i2 is the local index within the 2x2 block.
    
    # We will construct the global U, V, W for 4x4x4.
    # Size: 16x49.
    
    def construct_4x4x4():
        U2, V2, W2 = strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Block offsets
        # Block 0: (0,0), Block 1: (0,2), Block 2: (2,0), Block 3: (2,2)
        block_offsets = [(0,0), (0,2), (2,0), (2,2)]
        
        # Strassen coefficients
        alphas = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        betas = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        gammas = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for s in range(7):
            base_col = s * 7
            for r2 in range(7):
                col = base_col + r2
                # U
                for b in range(4):
                    if alphas[s][b] != 0:
                        i_off, j_off = block_offsets[b]
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i_off + i2
                                j = j_off + j2
                                U[i*4+j][col] += alphas[s][b] * U2[i2*2+j2][r2]
                # V
                for b in range(4):
                    if betas[s][b] != 0:
                        i_off, j_off = block_offsets[b]
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i_off + i2
                                j = j_off + j2
                                V[i*4+j][col] += betas[s][b] * V2[i2*2+j2][r2]
                # W
                for c in range(4):
                    if gammas[s][c] != 0:
                        i_off, j_off = block_offsets[c]
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i_off + i2
                                j = j_off + j2
                                W[j*4+i][col] += gammas[s][c] * W2[i2*2+j2][r2]
        
        return U, V, W

    # Now for <4,4,2>.
    # A is 4x4, B is 4x2, C is 4x2.
    # Split A into 2x2 blocks: A11, A12, A21, A22.
    # Split B into 2x2 blocks: B1, B2. (B is 4x2, so B1 is 2x2, B2 is 2x2).
    # C1 = A11 B1 + A12 B2.
    # C2 = A21 B1 + A22 B2.
    # Each term is a <2,2,2> multiplication.
    # We have 4 such terms.
    # Total rank = 4 * 7 = 28.
    
    def construct_4x4x2():
        U2, V2, W2 = strassen_2x2x2()
        rank = 28
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(8)]  # 4x2 = 8
        W = [[0]*rank for _ in range(8)]  # 2x4 = 8 (p*n = 2*4)
        
        # A blocks: 0, 1, 2, 3.
        # B blocks: 0, 1. (B1 is rows 0-1, B2 is rows 2-3).
        # C blocks: 0, 1. (C1 is rows 0-1, C2 is rows 2-3).
        
        # Term 0: A11 B1 -> C1
        # Term 1: A12 B2 -> C1
        # Term 2: A21 B1 -> C2
        # Term 3: A22 B2 -> C2
        
        terms = [
            (0, 0, 0),  # A_block, B_block, C_block
            (1, 1, 0),
            (2, 0, 1),
            (3, 1, 1)
        ]
        
        # A block offsets: (0,0), (0,2), (2,0), (2,2)
        a_offsets = [(0,0), (0,2), (2,0), (2,2)]
        # B block offsets: (0,0), (2,0) (rows 0-1, cols 0-1; rows 2-3, cols 0-1)
        b_offsets = [(0,0), (2,0)]
        # C block offsets: (0,0), (2,0) (rows 0-1, cols 0-1; rows 2-3, cols 0-1)
        # C is 4x2. Index k*n+i. k is col, i is row.
        # C1 is rows 0-1, cols 0-1.
        # C2 is rows 2-3, cols 0-1.
        c_offsets = [(0,0), (2,0)]
        
        for t, (a_b, b_b, c_b) in enumerate(terms):
            base_col = t * 7
            for r2 in range(7):
                col = base_col + r2
                # U
                i_off, j_off = a_offsets[a_b]
                for i2 in range(2):
                    for j2 in range(2):
                        i = i_off + i2
                        j = j_off + j2
                        U[i*4+j][col] = U2[i2*2+j2][r2]
                # V
                i_off, j_off = b_offsets[b_b]
                for i2 in range(2):
                    for j2 in range(2):
                        i = i_off + i2
                        j = j_off + j2
                        V[i*2+j][col] = V2[i2*2+j2][r2]
                # W
                i_off, j_off = c_offsets[c_b]
                for i2 in range(2):
                    for j2 in range(2):
                        i = i_off + i2
                        j = j_off + j2
                        # W index: k*n + i. k is col (0..1), i is row (0..3).
                        # j2 is col within block (0..1), i2 is row within block (0..1).
                        k = j2
                        row = i
                        W[k*4+row][col] = W2[i2*2+j2][r2]
        
        return U, V, W

    # Now combine <4,4,4> and <4,4,2> to get <4,4,6>.
    # A is 4x4.
    # B is 4x6. Split into B_left (4x4) and B_right (4x2).
    # C is 4x6. Split into C_left (4x4) and C_right (4x2).
    # C_left = A * B_left.
    # C_right = A * B_right.
    
    # Global U: 16 x (49+28) = 16 x 77.
    # Global V: 24 x 77. (4*6 = 24).
    # Global W: 24 x 77. (6*4 = 24).
    
    U444, V444, W444 = construct_4x4x4()
    U442, V442, W442 = construct_4x4x2()
    
    rank = 77
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    # Copy <4,4,4> part
    for i in range(16):
        for r in range(49):
            U[i][r] = U444[i][r]
    for i in range(16):
        for r in range(49):
            V[i][r] = V444[i][r]
    for i in range(16):
        for r in range(49):
            W[i][r] = W444[i][r]
    
    # Copy <4,4,2> part
    # B_right is columns 4-5 of B (4x6).
    # So V for <4,4,2> should be placed in columns 4-5 of the global V.
    # V442 is 8x28. Global V is 24x77.
    # V442[i, r] corresponds to B[i, j] where j is 0..1 within the 4x2 block.
    # In global B (4x6), these are columns 4 and 5.
    # So global V[i*6 + (4+j), 49+r] = V442[i*2+j, r].
    for i in range(4):
        for j in range(2):
            for r in range(28):
                V[i*6 + 4 + j][49 + r] = V442[i*2 + j][r]
    
    # W for <4,4,2> corresponds to C_right (4x2).
    # C_right is columns 4-5 of C (4x6).
    # W442 is 8x28. Global W is 24x77.
    # W442[k*4+i, r] corresponds to C[i, k] where k is 0..1 within the 4x2 block.
    # In global C (4x6), these are columns 4 and 5.
    # So global W[(4+k)*4 + i, 49+r] = W442[k*4+i, r].
    for k in range(2):
        for i in range(4):
            for r in range(28):
                W[(4+k)*4 + i][49 + r] = W442[k*4 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
