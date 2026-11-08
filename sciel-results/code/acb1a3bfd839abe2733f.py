import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 5

    # Build the 2x2x2 Strassen decomposition.
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # r=0: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1

    # r=1: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1

    # r=2: A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1

    # r=3: A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1

    # r=4: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1

    # r=5: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1

    # r=6: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1

    # Verify 2x2x2 Strassen: T[i,j,k] = sum_r U2[i*2+j,r]*V2[j*2+k,r]*W2[k*2+i,r]
    # should equal 1 iff i==k (the multiplication tensor)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(7):
                    val += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                expected = 1 if i == k else 0
                assert val == expected, f"Strassen 2x2x2 failed at ({i},{j},{k}): got {val}, expected {expected}"

    # Build 4x4x4 decomposition using recursive Strassen.
    # We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # The multiplication tensor for 4x4x4 can be decomposed as:
    # T4[i,j,k] = sum_{s=0}^{6} sum_{t=0}^{6} U4[i,j,s*7+t] * V4[j,k,s*7+t] * W4[k,i,s*7+t]
    # where for each top-level term s, we have block matrices L_s and R_s (2x2 matrices of 2x2 blocks),
    # and for each inner term t, we apply the 2x2x2 Strassen to the block product.
    
    # The key insight: if we have a decomposition of 2x2x2 multiplication with factors U2, V2, W2,
    # then for 4x4x4, we can use the following construction:
    # 
    # A 4x4 matrix A can be written as a 2x2 block matrix [[A00, A01], [A10, A11]]
    # where each block is 2x2.
    # 
    # The 7 Strassen terms at the top level give us 7 "super-matrices" L_s and R_s
    # which are 2x2 block matrices (so effectively 4x4 matrices).
    # 
    # For each s, the contribution to the product is L_s * R_s (a 4x4 matrix).
    # We need to decompose this 4x4 matrix product into rank-1 terms.
    # 
    # But L_s and R_s are specific linear combinations of the input blocks.
    # The product (L_s * R_s)_{IK} = sum_J L_s_{IJ} R_s_{JK} where I,J,K are block indices (0,1).
    # Each L_s_{IJ} and R_s_{JK} is a 2x2 matrix (a block).
    # 
    # So (L_s * R_s)_{IK} = sum_J L_s_{IJ} R_s_{JK} is a 2x2 matrix.
    # 
    # We can decompose this using the 2x2x2 Strassen decomposition applied to the "scalar" 
    # (block) multiplication. Since the blocks are 2x2 matrices, we need to be careful.
    # 
    # Actually, the correct approach is:
    # The 4x4x4 multiplication tensor T4 has entries T4[i,j,k] for i,j,k in 0..3.
    # We can write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0 where i1,j1,k1 in {0,1} and i0,j0,k0 in {0,1}.
    # 
    # The tensor T4[i,j,k] = 1 if i==k else 0.
    # 
    # Using the recursive Strassen structure:
    # T4[2*i1+i0, 2*j1+j0, 2*k1+k0] = sum_{s=0}^{6} sum_{t=0}^{6} 
    #     U4[2*i1+i0, 2*j1+j0, s*7+t] * V4[2*j1+j0, 2*k1+k0, s*7+t] * W4[2*k1+k0, 2*i1+i0, s*7+t]
    # 
    # The construction is:
    # For each top-level s, we have a 2x2 matrix of coefficients for the left factor and right factor.
    # Let's define the block-level Strassen coefficients.
    
    # The 7 Strassen terms at the block level (for 2x2 block matrices):
    # s=0: L = A00+A11, R = B00+B11
    # s=1: L = A10+A11, R = B00
    # s=2: L = A00, R = B01-B11
    # s=3: L = A11, R = B10-B00
    # s=4: L = A00+A01, R = B11
    # s=5: L = A10-A00, R = B00+B01
    # s=6: L = A01-A11, R = B10+B11
    
    # For each s, L_s and R_s are 2x2 matrices where the entries are 2x2 blocks.
    # So L_s[I,J] is a 2x2 matrix for I,J in {0,1}.
    
    # We can represent L_s as a 4x4 matrix where L_s[2*I+i0, 2*J+j0] is the (i0,j0) entry of block (I,J).
    
    # For the decomposition, we use:
    # U4[2*i1+i0, 2*j1+j0, s*7+t] = L_s[i1, j1][i0, j0] * U2[i0*2+j0, t]
    # Wait, that's not quite right. Let me think more carefully.
    
    # The 2x2x2 Strassen decomposition says:
    # For 2x2 matrices X, Y: (X*Y)[i,k] = sum_t U2[i*2+j, t] * V2[j*2+k, t] * W2[k*2+i, t] * X[i,j] * Y[j,k]
    # No, that's not right either. The Strassen decomposition is:
    # (X*Y)[i,k] = sum_t (sum_j U2[i*2+j,t] * X[i,j]) * (sum_j V2[j*2+k,t] * Y[j,k]) * W2[k*2+i,t]
    # Hmm, that's not standard. Let me reconsider.
    
    # The standard Strassen decomposition for 2x2 matrices is:
    # M = X*Y where X,Y are 2x2.
    # M[i,k] = sum_t U2[i*2+?, t] ... 
    # 
    # Actually, the way it works is:
    # We define 7 linear combinations of the entries of X and Y.
    # For each t, we have:
    #   u_t = sum_{i,j} U2[i*2+j, t] * X[i,j]
    #   v_t = sum_{j,k} V2[j*2+k, t] * Y[j,k]
    #   w_t = W2[?, t]  -- this doesn't make sense as a scalar.
    # 
    # Let me look at this differently. The tensor decomposition is:
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # where T[i,j,k] = delta_{i,k} for the multiplication tensor.
    # 
    # For 2x2: T[i,j,k] = 1 if i==k else 0, for i,j,k in {0,1}.
    # The rank-7 decomposition gives us vectors u_r, v_r, w_r of length 4 (for i*2+j, j*2+k, k*2+i).
    # 
    # For 4x4: T4[i,j,k] = 1 if i==k else 0, for i,j,k in {0,1,2,3}.
    # We want a rank-R decomposition: T4[i,j,k] = sum_r U4[i,j,r] * V4[j,k,r] * W4[k,i,r]
    # 
    # The recursive construction:
    # Write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0.
    # 
    # The 4x4 multiplication can be seen as:
    # (A*B)[2*i1+i0, 2*k1+k0] = sum_{j=0}^{3} A[2*i1+i0, j] * B[j, 2*k1+k0]
    # = sum_{j1=0}^{1} sum_{j0=0}^{1} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    # 
    # This is a 2x2 block matrix multiplication where each block is 2x2.
    # The block (i1,k1) of the result is:
    # (A*B)_{i1,k1} = sum_{j1=0}^{1} A_{i1,j1} * B_{j1,k1}
    # where A_{i1,j1} and B_{j1,k1} are 2x2 matrices.
    # 
    # So we need to decompose the 2x2 block matrix multiplication, where the "scalars" are 2x2 matrices.
    # The Strassen algorithm for 2x2 matrices gives 7 terms.
    # For each term s, we have:
    #   L_s = sum_{I,J} c^L_s[I,J] * A_{I,J}
    #   R_s = sum_{J,K} c^R_s[J,K] * B_{J,K}
    # where c^L_s and c^R_s are 2x2 matrices of coefficients.
    # 
    # Then the contribution to the product is L_s * R_s, which is a 2x2 block matrix.
    # Each block (I,K) of L_s * R_s is:
    #   (L_s * R_s)_{I,K} = sum_J c^L_s[I,J] * c^R_s[J,K] * A_{I,J} * B_{J,K}
    # Wait, that's not right. L_s and R_s are 2x2 block matrices, so:
    #   (L_s * R_s)_{I,K} = sum_J L_s_{I,J} * R_s_{J,K}
    # where L_s_{I,J} = sum_{i0,j0} (c^L_s[I,J] * A_{I,J})_{i0,j0}
    # 
    # This is getting complicated. Let me just directly construct the 49 terms.
    
    # Alternative approach: use the fact that the 4x4x4 multiplication tensor can be decomposed
    # using the tensor product of the 2x2x2 tensor with itself, but that gives rank 49*49 which is wrong.
    
    # Let me just use a direct construction. The rank of 4x4x4 multiplication is 49 (Strassen).
    # I'll construct it by iterating over the 7x7 combinations.
    
    # Define the block-level coefficients for the 7 Strassen terms.
    # For term s, L_s is a 2x2 matrix of coefficients for the A blocks, R_s for B blocks.
    # L_s[I,J] means the coefficient of A_{I,J} in the left factor of term s.
    
    L_coeff = []
    R_coeff = []
    
    # s=0: L = A00+A11, R = B00+B11
    L_coeff.append([[1,0],[0,1]])  # L[0,0]=1, L[1,1]=1
    R_coeff.append([[1,0],[0,1]])  # R[0,0]=1, R[1,1]=1
    
    # s=1: L = A10+A11, R = B00
    L_coeff.append([[0,0],[1,1]])  # L[1,0]=1, L[1,1]=1
    R_coeff.append([[1,0],[0,0]])  # R[0,0]=1
    
    # s=2: L = A00, R = B01-B11
    L_coeff.append([[1,0],[0,0]])  # L[0,0]=1
    R_coeff.append([[0,1],[0,-1]]) # R[0,1]=1, R[1,1]=-1
    
    # s=3: L = A11, R = B10-B00
    L_coeff.append([[0,0],[0,1]])  # L[1,1]=1
    R_coeff.append([[-1,0],[1,0]]) # R[0,0]=-1, R[1,0]=1
    
    # s=4: L = A00+A01, R = B11
    L_coeff.append([[1,1],[0,0]])  # L[0,0]=1, L[0,1]=1
    R_coeff.append([[0,0],[0,1]])  # R[1,1]=1
    
    # s=5: L = A10-A00, R = B00+B01
    L_coeff.append([[-1,0],[1,0]]) # L[0,0]=-1, L[1,0]=1
    R_coeff.append([[1,1],[0,0]])  # R[0,0]=1, R[0,1]=1
    
    # s=6: L = A01-A11, R = B10+B11
    L_coeff.append([[0,1],[0,-1]]) # L[0,1]=1, L[1,1]=-1
    R_coeff.append([[0,0],[1,1]])  # R[1,0]=1, R[1,1]=1
    
    # Now, for each s and t (t in 0..6), we construct the rank-1 term.
    # The term corresponds to:
    #   U4[i,j,r] * V4[j,k,r] * W4[k,i,r]
    # where r = s*7 + t.
    # 
    # The construction:
    # For a given s, we have L_s and R_s which are 2x2 matrices of 2x2 blocks.
    # L_s[I,J] is a 2x2 matrix (the coefficient times the A block A_{I,J}).
    # 
    # Actually, let me think about this more carefully using the tensor structure.
    # 
    # The 4x4x4 multiplication tensor T4 has entries T4[i,j,k] = delta_{i,k}.
    # We decompose it as T4 = sum_{s=0}^{6} sum_{t=0}^{6} u_{s,t} ⊗ v_{s,t} ⊗ w_{s,t}
    # 
    # For fixed s, the "sub-tensor" is:
    #   T4_s[i,j,k] = sum_{I,J,K} L_coeff[s][I,J] * R_coeff[s][J,K] * delta_{I,K} * T2[i0,j0,k0]
    # where i = 2*I+i0, j = 2*J+j0, k = 2*K+k0.
    # 
    # Wait, that's not quite right. Let me use a different approach.
    # 
    # The key realization: the 4x4x4 multiplication can be decomposed as follows.
    # 
    # A is a 4x4 matrix, B is a 4x4 matrix.
    # Write A as a 2x2 block matrix with 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Similarly B = [[B00, B01], [B10, B11]]
    # 
    # The product C = A*B is a 2x2 block matrix with 2x2 blocks:
    # C = [[C00, C01], [C10, C11]]
    # where C_{IK} = sum_J A_{IJ} * B_{JK}
    # 
    # The Strassen algorithm for 2x2 matrices gives:
    # C_{IK} = sum_s (L_s)_{IK} * (R_s)_{IK}  -- no, that's not right.
    # 
    # The Strassen algorithm says:
    # C = sum_s L_s * R_s  where L_s and R_s are 2x2 matrices formed from linear combinations of A and B blocks.
    # 
    # For each s, L_s * R_s is a 2x2 matrix of 2x2 blocks.
    # The (I,K) block of L_s * R_s is: sum_J L_s_{I,J} * R_s_{J,K}
    # 
    # Now, L_s_{I,J} is a 2x2 matrix (a linear combination of A blocks), and similarly for R_s.
    # So the (I,K) block of L_s * R_s is a 2x2 matrix computed as:
    #   sum_J (L_s_{I,J}) * (R_s_{J,K})
    # where each term is a 2x2 matrix product.
    # 
    # To decompose this into rank-1 terms, we use the 2x2x2 Strassen decomposition for the inner 2x2 matrix products.
    # 
    # For fixed s, and for each pair of block indices (I, J, K), we have:
    #   (L_s * R_s)_{I,K} += L_s_{I,J} * R_s_{J,K}
    # 
    # The product L_s_{I,J} * R_s_{J,K} is a 2x2 matrix product.
    # Using the 2x2x2 Strassen decomposition:
    #   (L_s_{I,J} * R_s_{J,K})[i0, k0] = sum_t U2[i0*2+j0, t] * V2[j0*2+k0, t] * W2[k0*2+i0, t] * (L_s_{I,J})[i0,j0] * (R_s_{J,K})[j0,k0]
    # 
    # So the full decomposition is:
    #   C[2*I+i0, 2*K+k0] = sum_s sum_J sum_t U2[i0*2+j0, t] * V2[j0*2+k0, t] * W2[k0*2+i0, t] * (L_s_{I,J})[i0,j0] * (R_s_{J,K})[j0,k0]
    # 
    # But (L_s_{I,J})[i0,j0] = L_coeff[s][I,J] * A[2*I+i0, 2*J+j0]
    # and (R_s_{J,K})[j0,k0] = R_coeff[s][J,K] * B[2*J+j0, 2*K+k0]
    # 
    # So:
    #   C[2*I+i0, 2*K+k0] = sum_s sum_J sum_t U2[i0*2+j0, t] * V2[j0*2+k0, t] * W2[k0*2+i0, t] 
    #       * L_coeff[s][I,J] * R_coeff[s][J,K] * A[2*I+i0, 2*J+j0] * B[2*J+j0, 2*K+k0]
    # 
    # We want to write this as:
    #   C[i,k] = sum_r U4[i,j,r] * V4[j,k,r] * W4[k,i,r] * A[i,j] * B[j,k]
    # 
    # Comparing, we see that for each (s, J, t), we get a term where:
    #   i = 2*I+i0, j = 2*J+j0, k = 2*K+k0
    #   U4[2*I+i0, 2*J+j0, r] = L_coeff[s][I,J] * U2[i0*2+j0, t]
    #   V4[2*J+j0, 2*K+k0, r] = R_coeff[s][J,K] * V2[j0*2+k0, t]
    #   W4[2*K+k0, 2*I+i0, r] = W2[k0*2+i0, t]
    # 
    # But wait, this depends on I and K as well, not just i0,j0,k0.
    # The issue is that L_coeff[s][I,J] and R_coeff[s][J,K] depend on the block indices I,J,K.
    # 
    # So for each s, J, t, and for each I, K, we get a separate rank-1 term?
    # That would give 7 * 2 * 7 * 2 * 2 = 392 terms, which is way too many.
    # 
    # I think I'm overcomplicating this. Let me just directly construct the 49 terms.
    # 
    # The rank-49 decomposition of 4x4x4 multiplication is well-known.
    # Let me just hardcode it or construct it algorithmically.
    
    # Actually, let me try a different approach. I'll construct the 4x4x4 decomposition
    # by using the tensor product structure properly.
    
    # The 4x4x4 multiplication tensor can be written as:
    # T4 = sum_{s=0}^{6} (L_s ⊗ I_2) * (R_s ⊗ I_2)  -- this doesn't make sense.
    
    # Let me just use a brute-force verification approach. I'll construct candidate
    # decompositions and verify them.
    
    # For now, let me fall back to a known decomposition.
    # The rank of 4x4x4 is 49. I'll construct it using the recursive Strassen method.
    
    # Here's the correct construction:
    # 
    # The 4x4x4 multiplication tensor T4[i,j,k] = delta_{i,k} for i,j,k in 0..3.
    # 
    # We can decompose it as:
    # T4[i,j,k] = sum_{s=0}^{6} sum_{t=0}^{6} U4[i,j,s*7+t] * V4[j,k,s*7+t] * W4[k,i,s*7+t]
    # 
    # where:
    # Let i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0.
    # 
    # U4[2*i1+i0, 2*j1+j0, s*7+t] = U2[i0*2+j0, t] * (coefficient from L_s at block (i1, j1))
    # V4[2*j1+j0, 2*k1+k0, s*7+t] = V2[j0*2+k0, t] * (coefficient from R_s at block (j1, k1))
    # W4[2*k1+k0, 2*i1+i0, s*7+t] = W2[k0*2+i0, t] * (coefficient that ensures the block structure)
    # 
    # The coefficients need to satisfy:
    # sum_{s,t} U4[i,j,s*7+t] * V4[j,k,s*7+t] * W4[k,i,s*7+t] = delta_{i,k}
    # 
    # Let me just construct it and verify.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(2):
                                    i = 2*i1 + i0
                                    j = 2*j1 + j0
                                    k = 2*k1 + k0
                                    
                                    # The coefficient for this term
                                    # U4[i, j, r] should be U2[i0*2+j0, t] * L_coeff[s][i1, j1]
                                    # V4[j, k, r] should be V2[j0*2+k0, t] * R_coeff[s][j1, k1]
                                    # W4[k, i, r] should be W2[k0*2+i0, t] * (something)
                                    
                                    # But this doesn't work because the sum over s,t needs to give delta_{i,k}.
                                    # 
                                    # Let me try a different construction.
                                    pass
    
    # This approach is getting too complicated. Let me just use the schoolbook for now
    # and return rank 80. The task says "rank-minimal" but the score is the rank value,
    # so lower is better. Rank 49 would be better than 80.
    
    # Let me try yet another approach: directly search for a rank-49 decomposition.
    
    # Actually, I recall that the 4x4x4 Strassen decomposition can be constructed as follows:
    # 
    # The 7 Strassen terms at the top level give us 7 "super-terms".
    # For each super-term s, we have 7 inner terms (from the 2x2x2 Strassen).
    # 
    # For super-term s:
    #   The left factor is a 4x4 matrix L_s formed from linear combinations of A's blocks.
    #   The right factor is a 4x4 matrix R_s formed from linear combinations of B's blocks.
    # 
    # The product L_s * R_s is a 4x4 matrix.
    # We decompose this product using the 2x2x2 Strassen decomposition applied to the 2x2 blocks.
    # 
    # But L_s and R_s are specific 4x4 matrices (with entries 0, 1, -1).
    # The product L_s * R_s is a specific 4x4 matrix.
    # 
    # The key insight: the decomposition of the 4x4x4 tensor is:
    # T4[i,j,k] = sum_{s=0}^{6} (L_s * R_s)[i,k] * delta_{something}
    # No, that's not right.
    
    # Let me just give up on the rank-49 construction and use schoolbook.
    # The score will be 80, which is worse than 49, but it's correct.
    
    # Actually, wait. Let me try to construct the rank-49 decomposition more carefully.
    
    # The 4x4x4 multiplication tensor T4 has the property that:
    # T4[i,j,k] = 1 if i == k, else 0.
    # 
    # We can decompose it as:
    # T4 = sum_{s=0}^{6} sum_{t=0}^{6} u_{s,t} ⊗ v_{s,t} ⊗ w_{s,t}
    # 
    # where u_{s,t}, v_{s,t}, w_{s,t} are vectors of length 16.
    # 
    # The construction:
    # For each s (0..6) and t (0..6):
    #   u_{s,t}[i,j] = U2[i%2 * 2 + j%2, t] * (block coefficient for (i//2, j//2) in L_s)
    #   v_{s,t}[j,k] = V2[j%2 * 2 + k%2, t] * (block coefficient for (j//2, k//2) in R_s)
    #   w_{s,t}[k,i] = W2[k%2 * 2 + i%2, t] * (block coefficient for (k//2, i//2) in some matrix)
    # 
    # The block coefficients need to be chosen so that the sum works out.
    
    # Let me try this construction and verify it.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    i1, i0 = i // 2, i % 2
                    j1, j0 = j // 2, j % 2
                    U4[i*4+j, r] = U2[i0*2+j0, t] * L_coeff[s][i1][j1]
                    
            for j in range(4):
                for k in range(4):
                    j1, j0 = j // 2, j % 2
                    k1, k0 = k // 2, k % 2
                    V4[j*4+k, r] = V2[j0*2+k0, t] * R_coeff[s][j1][k1]
                    
            for k in range(4):
                for i in range(4):
                    k1, k0 = k // 2, k % 2
                    i1, i0 = i // 2, i % 2
                    # W4 needs a coefficient that makes the sum work.
                    # For the Strassen decomposition to work, we need:
                    # sum_{s,t} U4[i,j,s*7+t] * V4[j,k,s*7+t] * W4[k,i,s*7+t] = delta_{i,k}
                    # 
                    # Let's try W4[k,i,r] = W2[k0*2+i0, t] * (some coefficient)
                    # The coefficient should be 1 if the block structure is consistent.
                    # 
                    # For the 2x2x2 Strassen, we have:
                    # sum_t U2[i0*2+j0,t] * V2[j0*2+k0,t] * W2[k0*2+i0,t] = delta_{i0,k0}
                    # 
                    # For the 4x4x4, we need:
                    # sum_{s,t} (U2[i0*2+j0,t] * L_coeff[s][i1][j1]) * (V2[j0*2+k0,t] * R_coeff[s][j1][k1]) * W4[k,i,s*7+t] = delta_{i,k}
                    # 
                    # This needs to equal delta_{i1,k1} * delta_{i0,k0} = delta_{i,k}.
                    # 
                    # So we need:
                    # sum_s L_coeff[s][i1][j1] * R_coeff[s][j1][k1] * (sum_t U2[i0*2+j0,t] * V2[j0*2+k0,t] * W4[k,i,s*7+t] / (U2[i0*2+j0,t] * V2[j0*2+k0,t])) = delta_{i1,k1} * delta_{i0,k0}
                    # 
                    # This is getting too complicated. Let me just try W4[k,i,r] = W2[k0*2+i0, t] and see if it works.
                    W4[k*4+i, r] = W2[k0*2+i0, t]
    
    # Verify the 4x4x4 decomposition
    T4_reconstructed = np.zeros((4, 4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(49):
                    val += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                T4_reconstructed[i,j,k] = val
                expected = 1 if i == k else 0
                if val != expected:
                    # The decomposition is wrong. Fall back to schoolbook.
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][column] = 1
                                V[j2 * p + k2][column] = 1
                                W[k2 * n + i2][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    # The 4x4x4 decomposition is correct. Now extend to 4x4x5.
    # We need to handle the 5th column of B.
    # For the 5th column, we use schoolbook: 4*4 = 16 additional terms.
    # Total rank: 49 + 16 = 65.
    
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy the 4x4x4 Strassen terms
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = int(U4[i*4+j, r])
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*4+k][r] = int(V4[j*4+k, r])
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4+i][r] = int(W4[k*4+i, r])
    
    # Add schoolbook terms for the 5th column of B (k=4)
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1  # j*4 + k where k=4
            W[4*4+i][col] = 1  # k*4 + i where k=4
            col += 1
    
    # Verify the full 4x4x5 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(5):
                val = 0
                for r in range(rank):
                    val += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                expected = 1 if i == k else 0
                if val != expected:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2 * m + j2][column] = 1
                                V_fb[j2 * p + k2][column] = 1
                                W_fb[k2 * n + i2][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
