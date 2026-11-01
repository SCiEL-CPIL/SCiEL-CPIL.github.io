import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    # Strassen factors for 2x2x2
    # U, V, W are 4x7 matrices (flattened)
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Strassen's 7 products
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
    # The factors U, V, W must satisfy the tensor equation.
    # For Strassen, the factors are known.
    
    # Let's construct the 4x7 matrices for one block.
    # U is 2x2, V is 2x2, W is 2x2.
    
    # M1: A00+A11, B00+B11
    # U[:,0] = [1, 0, 0, 1] (A00, A01, A10, A11) -> A00=1, A11=1
    # V[:,0] = [1, 1, 0, 0] (B00, B01, B10, B11) -> B00=1, B11=1? No.
    # V index j*p+k. p=2.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=1, k=0 -> 2. j=1, k=1 -> 3.
    # B00 is at 0. B01 is at 1. B10 is at 2. B11 is at 3.
    # V[:,0] = [1, 0, 0, 1] (B00, B11)
    # W[:,0] = [1, 0, 0, 1] (C00, C11)?
    # W index k*n+i. n=2.
    # k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
    # C00 is at 0. C11 is at 3.
    # W[:,0] = [1, 0, 0, 1]
    
    # This approach is tedious to hardcode correctly without errors.
    # Instead, we can use the known Strassen matrix factors.
    
    # Let's use a standard Strassen implementation for 2x2.
    
    def strassen_2x2_factors():
        # Returns U, V, W as 4x7 matrices
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00, C10
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00, C01
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1 # C00, C11
        
        return U, V, W

    U1, V1, W1 = strassen_2x2_factors()
    U2, V2, W2 = strassen_2x2_factors()
    
    # Assemble for 2x2x4
    # U is 4x14
    # V is 8x14
    # W is 8x14
    
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    # Block 0 (k=0,1)
    U[:, 0:7] = U1
    V[0:4, 0:7] = V1 # j=0,1; k=0,1
    W[0:4, 0:7] = W1 # k=0,1; i=0,1
    
    # Block 1 (k=2,3)
    U[:, 7:14] = U2
    V[4:8, 7:14] = V2 # j=0,1; k=2,3
    # V index for block 1: j*p+k. p=4.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 6. j=1, k=3 -> 7.
    # Wait, V shape is (m*p, rank) = (2*4, 14) = (8, 14).
    # Indices: 0,1,2,3,4,5,6,7.
    # j=0: 0,1,2,3. j=1: 4,5,6,7.
    # Block 0 uses k=0,1. So indices 0,1 for j=0 and 4,5 for j=1.
    # Block 1 uses k=2,3. So indices 2,3 for j=0 and 6,7 for j=1.
    
    # My V1 is 4x7. It corresponds to j=0,1 and k=0,1.
    # V1[0] -> j=0, k=0 (idx 0)
    # V1[1] -> j=0, k=1 (idx 1)
    # V1[2] -> j=1, k=0 (idx 4)
    # V1[3] -> j=1, k=1 (idx 5)
    
    # So we need to scatter V1 into V.
    V[0, 0:7] = V1[0]
    V[1, 0:7] = V1[1]
    V[4, 0:7] = V1[2]
    V[5, 0:7] = V1[3]
    
    # Block 1
    U[:, 7:14] = U2
    V[2, 7:14] = V2[0] # j=0, k=2
    V[3, 7:14] = V2[1] # j=0, k=3
    V[6, 7:14] = V2[2] # j=1, k=2
    V[7, 7:14] = V2[3] # j=1, k=3
    
    # W is 8x14.
    # W index k*n+i. n=2.
    # k=0: 0,1. k=1: 2,3. k=2: 4,5. k=3: 6,7.
    # Block 0 uses k=0,1. Indices 0,1,2,3.
    # Block 1 uses k=2,3. Indices 4,5,6,7.
    
    W[0:4, 0:7] = W1
    W[4:8, 7:14] = W2
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
