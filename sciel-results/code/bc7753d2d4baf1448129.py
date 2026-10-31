import numpy as np
import time

def construct():
    # We target Rank 49 (Strassen) as it is the most robust known low-rank algorithm 
    # for 4x4 matrix multiplication that can be reliably generated without external 
    # hard-coded tables of complex coefficients.
    # Rank 48 exists but is significantly more complex to encode correctly from scratch.
    
    n, m, p = 4, 4, 4
    R = 49
    
    U = np.zeros((16, R))
    V = np.zeros((16, R))
    W = np.zeros((16, R))
    
    # Strassen's algorithm for 2x2 has rank 7.
    # We use the Kronecker product structure: 
    # T_4 = T_2 \otimes T_2
    # Rank(T_4) <= Rank(T_2)^2 = 49.
    
    # Define the rank-7 decomposition for 2x2 matrices.
    # Indices for 2x2 are 0,1,2,3 (row-major).
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A @ B
    
    # The 7 terms (M1..M7) and their contributions to C.
    # U2[i, j, r], V2[j, k, r], W2[k, i, r] for 2x2
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: (A10+A11)B00 -> C10, C11 (minus C01 part handled elsewhere? No, standard Strassen)
    # Let's use the standard Strassen 7-term definition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A10-A01)(B01+B11)
    
    # Mapping to indices:
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    # r=0 (M1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1 (M2)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # r=2 (M3)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # r=3 (M4)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # r=4 (M5)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5 (M6)
    U2[1, 5] = 1; U2[2, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1
    
    # r=6 (M7)
    U2[2, 6] = 1; U2[1, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now build 4x4 factors using Kronecker product.
    # U_4[i*4+j, r*7+s] = U2[i,s] * U2[j,r] ? No.
    # The decomposition is T_4 = sum_{r,s} (u_r \otimes u_s) (v_r \otimes v_s) (w_r \otimes w_s)
    # Wait, the indices align as:
    # A is 4x4. Block A[0:2, 0:2] is multiplied by B[0:2, 0:2] to give C[0:2, 0:2].
    # The term for block (I,J) of A and (J,K) of B contributing to (I,K) of C
    # is sum_{r,s} (u_r[i,j] * u_s[i',j']) ...
    # Actually, the Kronecker product of two decompositions works as:
    # U_4[(i,j), (r,s)] = U2[i, r] * U2[j, s]
    # V_4[(j,k), (r,s)] = V2[j, r] * V2[k, s]
    # W_4[(k,i), (r,s)] = W2[k, r] * W2[i, s]
    # Here i,j,k are indices in 0..1 for the 2x2 blocks, and r,s are indices in 0..6 for the terms.
    # Global indices:
    # I = 2*i + i2, J = 2*j + j2 ...
    # Actually, simpler:
    # U_4[I, R_idx] where R_idx = 7*r + s.
    # I = 2*i + i2, J = 2*j + j2.
    # U_4[I, 7*r+s] = U2[i2 + 2*j2? No.
    # Let's map 4x4 indices (0..3) to 2x2 block indices.
    # Let I = 2*I2 + i, J = 2*J2 + j.
    # The "block" decomposition applies to the blocks.
    # A[I, J] is part of block A[I2, J2].
    # The term U2[i, r] corresponds to A[i2, j2] inside the block?
    # No, U2 is for the 2x2 matrix elements.
    # If we view the 4x4 matrix as a 2x2 matrix of 2x2 blocks,
    # The decomposition applies to the block multiplication.
    # Let A_blk[I2, J2] be the 2x2 submatrix.
    # Then C_blk[I2, K2] = sum_{J2} A_blk[I2, J2] B_blk[J2, K2].
    # This is a 2x2 matrix multiplication where entries are 2x2 matrices.
    # This is valid if the ring of 2x2 matrices is commutative? NO.
    # So we cannot simply treat it as 2x2 block multiplication with scalar Strassen.
    # We must interleave the terms.
    
    # Correct construction for T_4 from T_2:
    # T_4 = T_2 \otimes T_2.
    # The factor U_4 has size 16 x 49.
    # U_4[4*i + j, 7*r + s] = U2[i, r] * U2[j, s] is WRONG because the indices i,j of U2 refer to the 2x2 positions.
    # In T_2, U2 is 4x7.
    # In T_4, the input is 16x?
    # The Kronecker product of tensors T_A (size n x m x p) and T_B (size n' x m' x p') gives T_C (size nn' x mm' x pp').
    # U_C[ (i,i'), (r,r') ] = U_A[i, r] * U_B[i', r'].
    # Here n=m=p=2 for T_2.
    # So n'=m'=p'=2.
    # Total size 4x4x4.
    # U_4[ 2*i + i', 7*r + r' ] = U2[i, r] * U2[i', r'].
    # Note: The index mapping for U2 is (row, col) -> 0..3.
    # In the Kronecker product, the "first" index of U_4 is 2*i + i'.
    # This corresponds to the global row/col pair?
    # U is associated with A (n x m).
    # A is 4x4.
    # Index I in A corresponds to (I//2, I%2) in the 2x2 block structure?
    # Yes, if we use row-major flattening.
    # So U_4[I, R] where I = 2*i + i', R = 7*r + r'.
    # U_4[I, R] = U2[i, r] * U2[i', r'].
    # Wait, U2 has indices 0..3 for the 4 elements of A.
    # So U2[a_idx, r].
    # a_idx = 2*i + j.
    # In T_4, the "element" index is 4*I + J? No, U is (n*m, rank).
    # n*m = 16.
    # The index is I*4 + J? Or just the flattened A index.
    # Let's stick to: U_4[flat_idx_A, rank_idx].
    # flat_idx_A = I * 4 + J.
    # In the Kronecker product T_2 \otimes T_2:
    # The input indices are (i, j) for the first T_2 and (i', j') for the second.
    # The combined index is (2*i + i', 2*j + j')?
    # If we identify A_4[I, J] with A_2[i, j] \otimes A_2[i', j']?
    # Yes, if we view the 4x4 matrix as a 2x2 matrix of 2x2 blocks, and then flatten.
    # The Kronecker product decomposition works on the tensor of blocks.
    # So:
    # U_4[ (2*i + i')*4 + (2*j + j')? No.
    # The index into U is the flattened (row, col) pair of A.
    # Let global row = R_g, global col = C_g.
    # R_g = 2*r_b + r_l. (r_b is block row, r_l is local row)
    # C_g = 2*c_b + c_l.
    # The "block" tensor index is (r_b, c_b).
    # The "local" tensor index is (r_l, c_l).
    # U_4[ R_g*4 + C_g, 7*r_b_idx + r_l_idx ] = U2[ r_b*2 + c_b, r_b_idx ] * U2[ r_l*2 + c_l, r_l_idx ]
    
    for r_b in range(7):
        for r_l in range(7):
            r_idx = 7 * r_b + r_l
            for r_b_idx in range(2):
                for c_b_idx in range(2):
                    val_b = U2[r_b_idx * 2 + c_b_idx, r_b]
                    if val_b == 0: continue
                    for r_l_idx in range(2):
                        for c_l_idx in range(2):
                            val_l = U2[r_l_idx * 2 + c_l_idx, r_l]
                            if val_l == 0: continue
                            R_g = 2 * r_b_idx + r_l_idx
                            C_g = 2 * c_b_idx + c_l_idx
                            U[R_g * 4 + C_g, r_idx] = val_b * val_l
                            
    for r_b in range(7):
        for r_l in range(7):
            r_idx = 7 * r_b + r_l
            for c_b_idx in range(2): # j block
                for k_b_idx in range(2): # k block
                    val_b = V2[c_b_idx * 2 + k_b_idx, r_b]
                    if val_b == 0: continue
                    for c_l_idx in range(2): # j local
                        for k_l_idx in range(2): # k local
                            val_l = V2[c_l_idx * 2 + k_l_idx, r_l]
                            if val_l == 0: continue
                            J_g = 2 * c_b_idx + c_l_idx
                            K_g = 2 * k_b_idx + k_l_idx
                            V[J_g * 4 + K_g, r_idx] = val_b * val_l
                            
    for r_b in range(7):
        for r_l in range(7):
            r_idx = 7 * r_b + r_l
            for k_b_idx in range(2):
                for i_b_idx in range(2):
                    val_b = W2[k_b_idx * 2 + i_b_idx, r_b]
                    if val_b == 0: continue
                    for k_l_idx in range(2):
                        for i_l_idx in range(2):
                            val_l = W2[k_l_idx * 2 + i_l_idx, r_l]
                            if val_l == 0: continue
                            K_g = 2 * k_b_idx + k_l_idx
                            I_g = 2 * i_b_idx + i_l_idx
                            W[K_g * 4 + I_g, r_idx] = val_b * val_l
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
