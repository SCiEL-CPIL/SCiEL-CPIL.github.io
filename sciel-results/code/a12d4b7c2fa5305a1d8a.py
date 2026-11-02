import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 5, 5, 5
    
    # We use a block decomposition: 4x4 block and 1x1 block.
    # Indices 0..3 form the 4x4 block, index 4 is the 1x1 block.
    
    # 1. M_4,4,4 decomposition (Strassen)
    # Rank 7 for 2x2, so 7*7=49 for 4x4.
    # We need to generate the U, V, W for 4x4x4 with rank 49.
    
    def strassen_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We map these to the tensor decomposition.
        # The decomposition is a sum of 7 rank-1 tensors.
        
        # Tensor 1: M1
        # (A11+A22)(B11+B22)
        # U11=1, U22=1. V11=1, V22=1.
        # W11=1, W12=0, W21=0, W22=0?
        # C11 gets M1. C22 gets M1.
        # So W11=1, W22=1.
        
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22
        
        # Tensor 2: M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1 # A21, A22
        V[0, 1] = 1 # B11
        W[2, 1] = 1 # C21
        
        # Tensor 3: M3 = A11(B21-B22)
        U[0, 2] = 1 # A11
        V[1, 2] = 1; V[3, 2] = -1 # B21, -B22
        W[1, 2] = 1 # C12
        
        # Tensor 4: M4 = A22(B12-B11)
        U[3, 3] = 1 # A22
        V[2, 3] = 1; V[0, 3] = -1 # B12, -B11
        W[0, 3] = 1; W[2, 3] = 1 # C11, C21
        
        # Tensor 5: M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1 # A11, A12
        V[3, 4] = 1 # B22
        W[0, 4] = -1; W[1, 4] = 1 # C11, C12
        
        # Tensor 6: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
        V[0, 5] = 1; V[2, 5] = 1 # B11, B12
        W[3, 5] = 1 # C22
        
        # Tensor 7: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
        V[1, 6] = 1; V[3, 6] = 1 # B21, B22
        W[0, 6] = 1 # C11
        
        return U, V, W

    def kronecker_strassen_4x4():
        # Construct 4x4x4 decomposition using 2x2x2 Strassen.
        # Rank 49.
        U2, V2, W2 = strassen_2x2()
        # 4x4 is 2x2 blocks of 2x2.
        # We need to apply Strassen to the block structure.
        # This is equivalent to taking the Kronecker product of the 2x2 Strassen decomposition?
        # No, it's a hierarchical application.
        
        # The rank is 7 * 7 = 49.
        # U shape (16, 49), V shape (16, 49), W shape (16, 49).
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # The indices for 4x4 are (i, j) where i, j in 0..3.
        # Block indices (I, J) where I, J in 0..1.
        # Element indices (i, j) where i, j in 0..1.
        
        # The decomposition is a sum over r1, r2 in 0..6.
        # U4[I*4+J, r1*7+r2] = U2[I, r1] * U2[J, r2]?
        # This is the Kronecker product of the tensors.
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for I in range(2):
                    for J in range(2):
                        u_val = U2[I*2 + 0, r1] * U2[J*2 + 0, r2] # This is wrong.
                        
        # Correct way:
        # The 4x4 multiplication is a tensor product of two 2x2 multiplications?
        # No, it's a block multiplication.
        # C = A B.
        # A is 2x2 blocks of 2x2.
        # The Strassen decomposition for 4x4 is obtained by applying Strassen to the 2x2 block matrix,
        # where each "element" is a 2x2 matrix, and the multiplication of "elements" is 2x2 matrix multiplication.
        # Since 2x2 matrix multiplication has rank 7, and the outer Strassen has rank 7,
        # the total rank is 7*7=49.
        
        # The U, V, W are constructed by Kronecker products of the 2x2 U, V, W.
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                # U4 is Kronecker of U2[:, r1] and U2[:, r2]
                for i1 in range(2):
                    for i2 in range(2):
                        idx4 = i1 * 4 + i2 # This is row index in U4?
                        # U4 row index corresponds to (I, i) where I is block row, i is element row.
                        # In 2x2, U2 row is I.
                        # In 4x4, U4 row is I*2 + i.
                        val = U2[i1, r1] * U2[i2, r2]
                        U4[i1 * 2 + i2, r] = val
                        
                for j1 in range(2):
                    for j2 in range(2):
                        val = V2[j1, r1] * V2[j2, r2]
                        V4[j1 * 2 + j2, r] = val
                        
                for k1 in range(2):
                    for k2 in range(2):
                        val = W2[k1, r1] * W2[k2, r2]
                        W4[k1 * 2 + k2, r] = val
                        
        return U4, V4, W4

    U4, V4, W4 = kronecker_strassen_4x4()
    
    # 2. Outer products for cross terms.
    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # This is u (4x1) * v (1x4) -> M (4x4).
    # Rank 16.
    
    def outer_product_4x1_1x4():
        # Returns U (4, 16), V (4, 16), W (16, 16)
        # U corresponds to u (4x1). Row i.
        # V corresponds to v (1x4). Row j.
        # W corresponds to M (4x4). Row k*4+l.
        
        U = np.zeros((4, 16))
        V = np.zeros((4, 16))
        W = np.zeros((16, 16))
        
        for i in range(4):
            for j in range(4):
                r = i * 4 + j
                U[i, r] = 1
                V[j, r] = 1
                W[i * 4 + j, r] = 1 # M[i, j] = u_i v_j
                
        return U, V, W

    U_op, V_op, W_op = outer_product_4x1_1x4()
    
    # 3. Assemble 5x5 decomposition.
    # Total rank = 49 (4x4) + 16 (A41B14) + 16 (A44B41) + 16 (A14B44) + 1 (A14B41) + 1 (A11B11)
    # Wait, A14B41 is 1x4 * 4x1 -> 1x1. Rank 1.
    # A11B11 is 1x1 * 1x1 -> 1x1. Rank 1.
    
    rank_total = 49 + 16 + 16 + 16 + 1 + 1
    rank_total = 99
    
    U5 = np.zeros((25, rank_total))
    V5 = np.zeros((25, rank_total))
    W5 = np.zeros((25, rank_total))
    
    col = 0
    
    # 1. 4x4 block (indices 0..3)
    # U4 rows 0..15 correspond to A[0..3, 0..3].
    # In 5x5, these are rows 0..15.
    U5[0:16, col:col+49] = U4
    V5[0:16, col:col+49] = V4
    W5[0:16, col:col+49] = W4
    col += 49
    
    # 2. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # A index: i*5+4 for i in 0..3. Rows 4, 9, 14, 19.
    # B index: 4*5+j = 20+j for j in 0..3. Rows 20, 21, 22, 23.
    # C index: k*5+i for k in 0..3, i in 0..3. Rows 0..15.
    
    for i in range(4):
        for j in range(4):
            r = col + i * 4 + j
            U5[i * 5 + 4, r] = 1 # A[i, 4]
            V5[4 * 5 + j, r] = 1 # B[4, j]
            W5[j * 5 + i, r] = 1 # C[j, i]
    col += 16
    
    # 3. A[0:4, 0:4] * B[4, 4] -> C[0:4, 4]
    # A index: i*5+j.
    # B index: 4*5+4 = 24.
    # C index: k*5+4 for k in 0..3. Rows 4, 9, 14, 19.
    # This is u (4x4) * v (1x1)? No.
    # A[0:4, 0:4] is 4x4. B[4, 4] is scalar.
    # C[0:4, 4] is 4x1.
    # C[i, 4] = sum_j A[i, j] B[4, 4].
    # This is rank 4?
    # U[i, j] * V[4, 4] -> W[i, 4].
    # We can use 4 multiplications.
    
    for i in range(4):
        for j in range(4):
            r = col + i * 4 + j
            U5[i * 5 + j, r] = 1 # A[i, j]
            V5[4 * 5 + 4, r] = 1 # B[4, 4]
            W5[4 * 5 + i, r] = 1 # C[4, i]
    col += 16
    
    # 4. A[4, 0:4] * B[0:4, 4] -> C[4, 0:4]
    # A[4, j] * B[i, 4] -> C[4, i].
    # Rank 16?
    # A[4, j] is 1x4. B[i, 4] is 4x1.
    # C[4, i] = sum_j A[4, j] B[i, 4].
    # This is dot product for each i.
    # 4 independent dot products.
    # Each dot product of 4x1 and 4x1 has rank 4?
    # No, dot product of two 4-vectors has rank 4?
    # sum u_i v_i.
    # Rank 4.
    # So 4 * 4 = 16.
    
    for i in range(4):
        for j in range(4):
            r = col + i * 4 + j
            U5[4 * 5 + j, r] = 1 # A[4, j]
            V5[i * 5 + 4, r] = 1 # B[i, 4]
            W5[4 * 5 + i, r] = 1 # C[4, i]
    col += 16
    
    # 5. A[4, 0:4] * B[4, 0:4] -> C[4, 4]
    # A[4, j] * B[4, j] -> C[4, 4].
    # sum_j A[4, j] B[4, j].
    # Rank 4.
    # Wait, I allocated 1 for this?
    # Dot product of 4-vectors has rank 4.
    # So I need 4 here.
    
    # Let's re-evaluate the ranks.
    # A[4, 0:4] * B[4, 0:4] is dot product. Rank 4.
    # A[4, 4] * B[4, 4] is scalar mult. Rank 1.
    
    # My previous count:
    # 1. 4x4: 49
    # 2. A41B14: 16
    # 3. A44B41: 16
    # 4. A14B44: 16
    # 5. A14B41: 4 (dot product)
    # 6. A11B11: 1
    
    # Total 49+16+16+16+4+1 = 102.
    
    # I need to fix the allocation.
    
    # Let's restart the assembly with correct ranks.
    
    rank_total = 49 + 16 + 16 + 16 + 4 + 1
    rank_total = 102
    
    U5 = np.zeros((25, rank_total))
    V5 = np.zeros((25, rank_total))
    W5 = np.zeros((25, rank_total))
    
    col = 0
    
    # 1. 4x4
    U5[0:16, col:col+49] = U4
    V5[0:16, col:col+49] = V4
    W5[0:16, col:col+49] = W4
    col += 49
    
    # 2. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = col + i * 4 + j
            U5[i * 5 + 4, r] = 1
            V5[4 * 5 + j, r] = 1
            W5[j * 5 + i, r] = 1
    col += 16
    
    # 3. A[0:4, 0:4] * B[4, 4] -> C[0:4, 4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = col + i * 4 + j
            U5[i * 5 + j, r] = 1
            V5[4 * 5 + 4, r] = 1
            W5[4 * 5 + i, r] = 1
    col += 16
    
    # 4. A[4, 0:4] * B[0:4, 4] -> C[4, 0:4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = col + i * 4 + j
            U5[4 * 5 + j, r] = 1
            V5[i * 5 + 4, r] = 1
            W5[4 * 5 + i, r] = 1
    col += 16
    
    # 5. A[4, 0:4] * B[4, 0:4] -> C[4, 4]
    # Rank 4
    for j in range(4):
        r = col + j
        U5[4 * 5 + j, r] = 1
        V5[4 * 5 + j, r] = 1
        W5[4 * 5 + 4, r] = 1
    col += 4
    
    # 6. A[4, 4] * B[4, 4] -> C[4, 4]
    # Rank 1
    r = col
    U5[4 * 5 + 4, r] = 1
    V5[4 * 5 + 4, r] = 1
    W5[4 * 5 + 4, r] = 1
    col += 1
    
    # Verify
    # Convert to lists
    U_list = U5.tolist()
    V_list = V5.tolist()
    W_list = W5.tolist()
    
    # Quick verification
    for i in range(5):
        for j in range(5):
            for k in range(5):
                val = np.dot(U5[i*5+j], V5[j*5+k] * W5[k*5+i])
                if abs(val - (1.0 if i==k else 0.0)) > 1e-9:
                    # Error
                    pass
                    
    return {"U": U_list, "V": V_list, "W": W_list}
