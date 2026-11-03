import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # We decompose the multiplication of a 4x4 matrix A by a 4x7 matrix B.
    # A is 4x4, B is 4x7.
    # We can view A as a 2x2 block matrix of 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B is 4x7. We can split B into two parts: B_left (4x4) and B_right (4x3).
    # Let B = [B_L, B_R] where B_L is 4x4 and B_R is 4x3.
    # Then C = A * B = [A * B_L, A * B_R].
    # C is 4x7.
    
    # Strategy:
    # 1. Decompose A * B_L (4x4 * 4x4) using Strassen for 2x2 blocks.
    #    This part has rank 28 (4 block multiplications, each using 7 Strassen terms).
    # 2. Decompose A * B_R (4x4 * 4x3) using schoolbook, as Strassen doesn't 
    #    directly apply to non-square without padding, and padding would increase rank.
    #    Schoolbook for 4x4 * 4x3 is 4*4*3 = 48 terms.
    # Total rank = 28 + 48 = 76.
    
    # Let's verify the Strassen decomposition for 4x4 * 4x4.
    # A = [[A00, A01], [A10, A11]]
    # B_L = [[B00, B01], [B10, B11]]
    # C_L = A * B_L
    # C_L00 = A00*B00 + A01*B10
    # C_L01 = A00*B01 + A01*B11
    # C_L10 = A10*B00 + A11*B10
    # C_L11 = A10*B01 + A11*B11
    
    # We use Strassen for each 2x2 block multiplication.
    # However, the standard Strassen algorithm computes 2x2 * 2x2 in 7 multiplications.
    # Here we have 4 separate 2x2 * 2x2 multiplications (for the 4 blocks of C_L).
    # If we apply Strassen independently to each block product, we get 4 * 7 = 28 terms.
    # This is valid because the blocks are independent in the tensor decomposition sense?
    # No, wait. The Strassen algorithm is for a single 2x2 * 2x2 multiplication.
    # The 4x4 * 4x4 multiplication is NOT the same as 4 independent 2x2 * 2x2 multiplications.
    # The block C_L00 depends on A00, A01, B00, B10.
    # The block C_L01 depends on A00, A01, B01, B11.
    # etc.
    # If we just decompose each block product separately, we are doing 4 separate 2x2*2x2 mults.
    # But the 4x4 multiplication tensor is the sum of these 4 block tensors?
    # Yes, the tensor for 4x4*4x4 is the direct sum of the tensors for the 4 block products,
    # BUT the inputs A and B are shared.
    # In tensor decomposition, if we have a tensor T = T1 + T2 + T3 + T4, where T1 is the tensor for A00*B00 + A01*B10, etc.,
    # and if the supports of T1, T2, T3, T4 are disjoint in the output space, we can sum their ranks.
    # The output blocks C_L00, C_L01, C_L10, C_L11 are disjoint parts of the output matrix C_L.
    # So the rank of the 4x4*4x4 tensor is at most the sum of the ranks of the 4 block tensors.
    # Each block tensor is a 2x2 * 2x2 multiplication tensor (with some linear combinations of inputs).
    # Wait, the inputs to the block multiplications are linear combinations of the original A and B blocks.
    # For example, to compute C_L00, we need A00, A01, B00, B10.
    # The tensor for C_L00 = A00*B00 + A01*B10 is a sum of two 2x2*2x2 tensors.
    # If we use Strassen to compute a 2x2*2x2 product, it uses 7 multiplications.
    # But here we have a SUM of two products.
    # Actually, the standard Strassen algorithm computes the full 2x2*2x2 product C = A*B in 7 multiplications.
    # It does NOT compute A00*B00 + A01*B10 directly.
    # The Strassen algorithm for 2x2*2x2 gives the 4 output blocks of a 2x2 matrix product.
    # So if we view the 4x4*4x4 multiplication as a 2x2 block matrix multiplication where each "element" is a 2x2 matrix,
    # then the Strassen algorithm for 2x2*2x2 (with matrix entries) applies directly!
    # The rank of the 2x2*2x2 matrix multiplication tensor (where entries are scalars) is 7.
    # The rank of the 2x2*2x2 matrix multiplication tensor (where entries are 2x2 matrices) is 7 * (rank of 2x2*2x2 scalar tensor)?
    # No. The Kronecker product of two tensors T1 (n1,m1,p1) and T2 (n2,m2,p2) has rank rank(T1)*rank(T2).
    # The 4x4*4x4 tensor is the Kronecker product of the 2x2*2x2 tensor with itself.
    # So rank(4x4*4x4) <= rank(2x2*2x2) * rank(2x2*2x2) = 7 * 7 = 49.
    # This is the Kronecker product approach I tried earlier and failed to implement.
    
    # Let's try to implement the Kronecker product correctly this time.
    # T_2x2 is a tensor of size 2x2x2 with rank 7.
    # T_4x4 = T_2x2 x T_2x2 (Kronecker product).
    # The rank-1 terms of T_4x4 are the Kronecker products of the rank-1 terms of T_2x2.
    # If T_2x2 = sum_{k=1}^7 u_k v_k w_k^T, then T_4x4 = sum_{k,l=1}^7 (u_k x u_l) (v_k x v_l) (w_k x w_l)^T.
    # This gives 49 terms.
    
    # Let's define the 7 rank-1 terms for 2x2*2x2.
    # Indices: u[i,j], v[j,k], w[k,i] for i,j,k in 0..1.
    # Flattened: u[i*2+j], v[j*2+k], w[k*2+i].
    
    # Strassen terms for 2x2*2x2:
    # M1 = A00 * (B01 - B11) -> C01 += M1
    # M2 = (A00 + A01) * B11 -> C00 += M2
    # M3 = (A10 - A00) * B00 -> C10 += M3
    # M4 = A11 * (B10 - B00) -> C11 += M4
    # M5 = (A00 + A11) * (B00 + B11) -> C00 += M5
    # M6 = (A01 + A11) * (B10 + B11) -> C11 += M6
    # M7 = (A00 - A10 + A11) * B00 -> C00 += M7
    
    # Let's write the vectors u, v, w for each term.
    # u is length 4 (A00, A01, A10, A11)
    # v is length 4 (B00, B01, B10, B11)
    # w is length 4 (C00, C01, C10, C11)
    
    terms_2x2 = []
    
    # Term 1: A00 * (B01 - B11) -> C01
    u1 = [1, 0, 0, 0]
    v1 = [0, 1, 0, -1]
    w1 = [0, 1, 0, 0]
    terms_2x2.append((u1, v1, w1))
    
    # Term 2: (A00 + A01) * B11 -> C00
    u2 = [1, 1, 0, 0]
    v2 = [0, 0, 0, 1]
    w2 = [1, 0, 0, 0]
    terms_2x2.append((u2, v2, w2))
    
    # Term 3: (A10 - A00) * B00 -> C10
    u3 = [-1, 0, 1, 0]
    v3 = [1, 0, 0, 0]
    w3 = [0, 0, 1, 0]
    terms_2x2.append((u3, v3, w3))
    
    # Term 4: A11 * (B10 - B00) -> C11
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [0, 0, 0, 1]
    terms_2x2.append((u4, v4, w4))
    
    # Term 5: (A00 + A11) * (B00 + B11) -> C00
    u5 = [1, 0, 0, 1]
    v5 = [1, 0, 0, 1]
    w5 = [1, 0, 0, 0]
    terms_2x2.append((u5, v5, w5))
    
    # Term 6: (A01 + A11) * (B10 + B11) -> C11
    u6 = [0, 1, 0, 1]
    v6 = [0, 0, 1, 1]
    w6 = [0, 0, 0, 1]
    terms_2x2.append((u6, v6, w6))
    
    # Term 7: (A00 - A10 + A11) * B00 -> C00
    u7 = [1, 0, -1, 1]
    v7 = [1, 0, 0, 0]
    w7 = [1, 0, 0, 0]
    terms_2x2.append((u7, v7, w7))
    
    # Now construct the 4x4*4x4 terms using Kronecker products.
    # u_kron has length 16. Index: (i1*2+i2)*4 + (j1*2+j2)
    # u_kron[idx] = u1[i1*2+j1] * u2[i2*2+j2]
    # Wait, the indices for the 2x2 tensor are i, j, k in 0..1.
    # u[i*2+j].
    # For the 4x4 tensor, indices are I, J, K in 0..3.
    # I = 2*i1 + i2, J = 2*j1 + j2, K = 2*k1 + k2.
    # u_kron[I*4+J] = u1[i1*2+j1] * u2[i2*2+j2]
    # v_kron[J*4+K] = v1[j1*2+k1] * v2[j2*2+k2]
    # w_kron[K*4+I] = w1[k1*2+i1] * w2[k2*2+i2]
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for t1 in terms_2x2:
        for t2 in terms_2x2:
            u1, v1, w1 = t1
            u2, v2, w2 = t2
            
            u_kron = [0.0] * 16
            v_kron = [0.0] * 16
            w_kron = [0.0] * 16
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            I = i1 * 2 + i2
                            J = j1 * 2 + j2
                            u_idx = I * 4 + J
                            u_kron[u_idx] = u1[i1 * 2 + j1] * u2[i2 * 2 + j2]
                            
                            # v indices: J, K
                            # J = j1*2 + j2
                            # K = k1*2 + k2
                            # We need to sum over k1, k2? No, the term is a product.
                            # v_kron[J*4+K] = v1[j1*2+k1] * v2[j2*2+k2]
                            # This depends on k1, k2. So we need a loop for k1, k2 as well?
                            # Yes.
                            pass
                            
            # Let's redo the Kronecker product construction properly with full loops.
            u_kron = [0.0] * 16
            v_kron = [0.0] * 16
            w_kron = [0.0] * 16
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            I = i1 * 2 + i2
                            J = j1 * 2 + j2
                            u_idx = I * 4 + J
                            u_kron[u_idx] = u1[i1 * 2 + j1] * u2[i2 * 2 + j2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    K = k1 * 2 + k2
                                    v_idx = J * 4 + K
                                    v_kron[v_idx] += v1[j1 * 2 + k1] * v2[j2 * 2 + k2]
                                    
                                    w_idx = K * 4 + I
                                    w_kron[w_idx] += w1[k1 * 2 + i1] * w2[k2 * 2 + i2]
            
            U_4x4.append(u_kron)
            V_4x4.append(v_kron)
            W_4x4.append(w_kron)
            
    # Now we have 49 terms for the 4x4*4x4 part.
    # We need to add the 4x4*4x3 part (schoolbook, 48 terms).
    # Total rank = 49 + 48 = 97.
    # This is worse than the 76 I calculated earlier?
    # Why?
    # Because the Kronecker product gives rank 49 for the square part.
    # The block-strassen approach (4 independent 2x2*2x2 Strassen decompositions) gave rank 28.
    # Why is 28 < 49?
    # Because the 4x4*4x4 tensor is NOT the Kronecker product of two 2x2*2x2 tensors?
    # It IS the Kronecker product.
    # rank(T x S) = rank(T) * rank(S).
    # rank(2x2*2x2) = 7.
    # So rank(4x4*4x4) <= 49.
    # But is it 28?
    # The block-strassen approach assumes that we can decompose the 4 block products independently.
    # C00 = A00*B00 + A01*B10. This is a sum of two 2x2*2x2 products.
    # If we use Strassen to compute A00*B00, it uses 7 terms.
    # If we use Strassen to compute A01*B10, it uses 7 terms.
    # Total 14 terms for C00.
    # Similarly for C01, C10, C11.
    # Total 4 * 14 = 56 terms?
    # No, Strassen computes the FULL 2x2*2x2 product in 7 terms.
    # It does not compute a single block like A00*B00 in 7 terms.
    # The 7 terms of Strassen produce all 4 output blocks of a 2x2*2x2 multiplication.
    # So if we want to compute C00 = A00*B00 + A01*B10, we cannot just use Strassen on A00*B00.
    # We must use Strassen on the full 4x4*4x4 multiplication.
    # And the rank of that is 49 (via Kronecker).
    # So 49 is the correct rank for the square part using this method.
    # My previous estimate of 28 was wrong.
    
    # So the total rank is 49 (for 4x4*4x4) + 48 (for 4x4*4x3) = 97.
    # This is an improvement over 112.
    
    # Let's assemble the final U, V, W matrices.
    # U is (n*m, rank) = (16, 97)
    # V is (m*p, rank) = (28, 97)
    # W is (p*n, rank) = (28, 97)
    
    rank_total = len(U_4x4) + n * m * (p - m) # 49 + 16*3 = 49 + 48 = 97
    # Wait, p=7, m=4. p-m=3.
    # The 4x4*4x3 part uses schoolbook.
    # A is 4x4, B_R is 4x3.
    # C_R is 4x3.
    # Schoolbook: for i in 0..3, j in 0..3, k in 0..2:
    # U[i*4+j] += 1
    # V[j*4+k] += 1  (Note: B_R is the right part of B. B is 4x7.
    # B_L is columns 0..3, B_R is columns 4..6.
    # So V index for B_R is j*7 + (4+k) where k in 0..2.
    # W index for C_R is (4+k)*4 + i.
    
    U_final = [[0.0] * rank_total for _ in range(n * m)]
    V_final = [[0.0] * rank_total for _ in range(m * p)]
    W_final = [[0.0] * rank_total for _ in range(p * n)]
    
    # Add 4x4*4x4 terms
    for idx in range(len(U_4x4)):
        u = U_4x4[idx]
        v = V_4x4[idx]
        w = W_4x4[idx]
        for i in range(16):
            U_final[i][idx] = u[i]
        for j in range(16):
            V_final[j][idx] = v[j]
        for k in range(16):
            W_final[k][idx] = w[k]
            
    # Add 4x4*4x3 schoolbook terms
    col_offset = len(U_4x4)
    col = col_offset
    for i in range(n):
        for j in range(m):
            for k in range(p - m):
                # B_R column index is m + k
                b_col = m + k
                U_final[i * m + j][col] = 1.0
                V_final[j * p + b_col][col] = 1.0
                W_final[b_col * n + i][col] = 1.0
                col += 1
                
    return {"U": U_final, "V": V_final, "W": W_final}
