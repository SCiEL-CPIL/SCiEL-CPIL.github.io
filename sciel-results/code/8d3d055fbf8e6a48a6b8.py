import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W with zeros
    # We will build them block by block.
    # To avoid dynamic sizing issues, we can just append columns.
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def add_block(U_block, V_block, W_block):
        U_blocks.append(U_block)
        V_blocks.append(V_block)
        W_blocks.append(W_block)
        
    # Helper to create Strassen 2x2x2 block
    # Indices: i in [0,1], j in [0,1], k in [0,1]
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = A00 (B00 + B11)
        U[0, 0] = 1
        V[0, 0] = 1
        V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A00 + A10) B11
        U[0, 1] = 1
        U[2, 1] = 1
        V[3, 1] = 1
        W[2, 1] = 1
        W[3, 1] = 1
        
        # M3 = (A01 + A11) B00
        U[1, 2] = 1
        U[3, 2] = 1
        V[0, 2] = 1
        W[2, 2] = 1
        
        # M4 = A11 (B10 + B01)
        U[3, 3] = 1
        V[2, 3] = 1
        V[1, 3] = 1
        W[0, 3] = 1
        W[3, 3] = 1
        
        # M5 = A00 (B01 - B11)
        U[0, 4] = 1
        V[1, 4] = 1
        V[3, 4] = -1
        W[2, 4] = 1
        W[0, 4] = -1
        
        # M6 = (A10 - A00) (B00 + B10)
        U[2, 5] = 1
        U[0, 5] = -1
        V[0, 5] = 1
        V[2, 5] = 1
        W[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11) (B11 + B01)
        U[1, 6] = 1
        U[3, 6] = -1
        V[3, 6] = 1
        V[1, 6] = 1
        W[0, 6] = 1
        W[2, 6] = 1
        
        return U, V, W

    # Helper for 2x2x3 (Rank 2)
    # C_ij = A_i0 B_0j + A_i1 B_1j
    # U indices: i*2+j (j in 0,1)
    # V indices: j*3+k (j in 0,1; k in 0,1,2)
    # W indices: k*2+i (k in 0,1,2; i in 0,1)
    def rank2_223():
        U = np.zeros((4, 2))
        V = np.zeros((6, 2))
        W = np.zeros((6, 2))
        
        # Term 1: A_i0 B_0j
        # U[i*2+0] = 1
        # V[0*3+j] = 1
        # W[j*2+i] = 1
        for i in range(2):
            U[i*2+0, 0] = 1
            for j in range(3):
                V[0*3+j, 0] = 1
                W[j*2+i, 0] = 1
                
        # Term 2: A_i1 B_1j
        for i in range(2):
            U[i*2+1, 1] = 1
            for j in range(3):
                V[1*3+j, 1] = 1
                W[j*2+i, 1] = 1
                
        return U, V, W

    # Helper for 1x2x7 (Rank 2)
    # C_j = A_0 B_0j + A_1 B_1j
    # U index: 0*2+j = j
    # V index: j*7+k
    # W index: k*1+0 = k
    def rank2_127():
        U = np.zeros((2, 2))
        V = np.zeros((14, 2))
        W = np.zeros((7, 2))
        
        for j in range(2):
            U[j, j] = 1
            for k in range(7):
                V[j*7+k, j] = 1
                W[k, j] = 1
                
        return U, V, W

    # Helper for 2x1x7 (Rank 2)
    # C_i = A_i0 B_0j? No.
    # A is 2x1. B is 1x7. C is 2x7.
    # C_i,k = A_i0 B_0k.
    # This is outer product. Rank 1?
    # T_{i,0,k} = 1.
    # Sum over r.
    # U[i*1+0, r] = 1
    # V[0*7+k, r] = 1
    # W[k*2+i, r] = 1
    # Rank 1 is sufficient?
    # C = A B. A is col vec, B is row vec.
    # Yes, rank 1.
    def rank1_217():
        U = np.zeros((2, 1))
        V = np.zeros((7, 1))
        W = np.zeros((14, 1))
        
        for i in range(2):
            U[i, 0] = 1
        for k in range(7):
            V[k, 0] = 1
            W[k*2+i, 0] = 1 # Wait, W index depends on i.
            # W index is k*n+i. n=2.
            # So W[k*2+i, 0] = 1.
            # But i is fixed in the loop? No.
            # The term is A_i0 B_0k.
            # So for each i, k, we have a contribution.
            # But they are all part of the same rank-1 component?
            # No, the rank-1 component is the whole tensor.
            # U[i,0]=1 for all i.
            # V[k,0]=1 for all k.
            # W[k*2+i, 0]=1 for all k,i.
            # Let's check.
            # Sum_r U[i*1+0, r] V[0*7+k, r] W[k*2+i, r]
            # = U[i,0] V[k,0] W[k*2+i, 0]
            # = 1 * 1 * 1 = 1.
            # Correct.
            
        U = np.ones((2, 1))
        V = np.ones((7, 1))
        W = np.ones((14, 1))
        return U, V, W

    # Helper for 1x3x7 (Rank 3)
    # C_k = A_0 B_0k + A_1 B_1k + A_2 B_2k
    def rank3_137():
        U = np.zeros((3, 3))
        V = np.zeros((21, 3))
        W = np.zeros((7, 3))
        
        for j in range(3):
            U[j, j] = 1
            for k in range(7):
                V[j*7+k, j] = 1
                W[k, j] = 1
                
        return U, V, W

    # Mapping of blocks to global indices
    # n=3, m=5, p=7
    
    # Group 1: A[:, 0:2] * B[0:2, :]
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U_b, V_b, W_b = strassen_222()
    # Map to global
    # U global index: i*5+j. i in 0:2, j in 0:2.
    # V global index: j*7+k. j in 0:2, k in 0:2.
    # W global index: k*3+i. k in 0:2, i in 0:2.
    
    def map_block(U_b, V_b, W_b, i_off, j_off, k_off, i_size, j_size, k_size):
        U_g = np.zeros((n*m, U_b.shape[1]))
        V_g = np.zeros((m*p, V_b.shape[1]))
        W_g = np.zeros((p*n, W_b.shape[1]))
        
        for r in range(U_b.shape[1]):
            for u_i in range(i_size):
                for u_j in range(j_size):
                    # U_b index: u_i * j_size + u_j ?
                    # No, U_b is local.
                    # We need to map local indices to global.
                    # Local U index: u_i * j_size + u_j
                    # Global U index: (i_off + u_i) * m + (j_off + u_j)
                    
                    # This is inefficient. Let's just construct the full matrices directly.
                    pass
        return U_g, V_g, W_g

    # Given the complexity of mapping, it is safer to construct the full matrices directly.
    
    U = np.zeros((n*m, 39))
    V = np.zeros((m*p, 39))
    W = np.zeros((p*n, 39))
    
    col = 0
    
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U_s, V_s, W_s = strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[(i)*5 + j, col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[j*7 + k, col+r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[k*3 + i, col+r] = W_s[k*2+i, r]
    col += 7
    
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    U_s, V_s, W_s = strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*5 + j, col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[j*7 + (2+k), col+r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[(2+k)*3 + i, col+r] = W_s[k*2+i, r]
    col += 7
    
    # 3. A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]
    U_s, V_s, W_s = rank2_223()
    for r in range(2):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*5 + j, col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(3):
                if V_s[j*3+k, r] != 0:
                    V[j*7 + (4+k), col+r] = V_s[j*3+k, r]
        for k in range(3):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[(4+k)*3 + i, col+r] = W_s[k*2+i, r]
    col += 2
    
    # 4. A[2:3, 0:2] * B[0:2, 0:7] -> C[2:3, 0:7]
    U_s, V_s, W_s = rank2_127()
    for r in range(2):
        for i in range(1):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[(2+i)*5 + j, col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(7):
                if V_s[j*7+k, r] != 0:
                    V[j*7 + k, col+r] = V_s[j*7+k, r]
        for k in range(7):
            for i in range(1):
                if W_s[k*1+i, r] != 0:
                    W[k*3 + (2+i), col+r] = W_s[k*1+i, r]
    col += 2
    
    # 5. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    U_s, V_s, W_s = strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*5 + (2+j), col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[(2+j)*7 + k, col+r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[k*3 + i, col+r] = W_s[k*2+i, r]
    col += 7
    
    # 6. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    U_s, V_s, W_s = strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*5 + (2+j), col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[(2+j)*7 + (2+k), col+r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[(2+k)*3 + i, col+r] = W_s[k*2+i, r]
    col += 7
    
    # 7. A[0:2, 2:4] * B[2:4, 4:7] -> C[0:2, 4:7]
    U_s, V_s, W_s = rank2_223()
    for r in range(2):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*5 + (2+j), col+r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(3):
                if V_s[j*3+k, r] != 0:
                    V[(2+j)*7 + (4+k), col+r] = V_s[j*3+k, r]
        for k in range(3):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[(4+k)*3 + i, col+r] = W_s[k*2+i, r]
    col += 2
    
    # 8. A[0:2, 4:5] * B[4:5, 0:7] -> C[0:2, 0:7]
    U_s, V_s, W_s = rank1_217()
    for r in range(1):
        for i in range(2):
            for j in range(1):
                if U_s[i*1+j, r] != 0:
                    U[i*5 + (4+j), col+r] = U_s[i*1+j, r]
        for j in range(1):
            for k in range(7):
                if V_s[j*7+k, r] != 0:
                    V[(4+j)*7 + k, col+r] = V_s[j*7+k, r]
        for k in range(7):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[k*3 + i, col+r] = W_s[k*2+i, r]
    col += 1
    
    # 9. A[2:3, 2:5] * B[2:5, 0:7] -> C[2:3, 0:7]
    U_s, V_s, W_s = rank3_137()
    for r in range(3):
        for i in range(1):
            for j in range(3):
                if U_s[i*3+j, r] != 0:
                    U[(2+i)*5 + (2+j), col+r] = U_s[i*3+j, r]
        for j in range(3):
            for k in range(7):
                if V_s[j*7+k, r] != 0:
                    V[(2+j)*7 + k, col+r] = V_s[j*7+k, r]
        for k in range(7):
            for i in range(1):
                if W_s[k*1+i, r] != 0:
                    W[k*3 + (2+i), col+r] = W_s[k*1+i, r]
    col += 3
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
