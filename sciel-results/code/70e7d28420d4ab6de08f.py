"""Optimized matrix multiplication 3x5x7 via structured decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 99.
    # Structure:
    # 1. Split n=3 into 2 and 1.
    #    Block 1: A[0:2, :] (2x5) * B (5x7) -> C[0:2, :] (2x7)
    #    Block 2: A[2:3, :] (1x5) * B (5x7) -> C[2:3, :] (1x7)
    # 2. Split Block 1 (2x5x7) by m=5 into 2, 2, 1.
    #    Sub-block 1.1: A[0:2, 0:2] (2x2) * B[0:2, :] (2x7) -> C[0:2, :]
    #    Sub-block 1.2: A[0:2, 2:4] (2x2) * B[2:4, :] (2x7) -> C[0:2, :]
    #    Sub-block 1.3: A[0:2, 4:5] (2x1) * B[4:5, :] (1x7) -> C[0:2, :]
    # 3. Decompose Sub-blocks:
    #    Sub-block 1.1 (2x2x7): 3x (2x2x2 Strassen) + 1x (2x2x1 Schoolbook) = 3*7 + 4 = 25
    #    Sub-block 1.2 (2x2x7): 3x (2x2x2 Strassen) + 1x (2x2x1 Schoolbook) = 3*7 + 4 = 25
    #    Sub-block 1.3 (2x1x7): Schoolbook = 2*1*7 = 14
    #    Block 2 (1x5x7): Schoolbook = 1*5*7 = 35
    # Total Rank = 25 + 25 + 14 + 35 = 99.

    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # Helper to create a zero vector of size N
    def zero_vec(size):
        return [0.0] * size

    # Dimensions for U, V, W
    n_U = n * m # 15
    n_V = m * p # 35
    n_W = p * n # 21

    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i

    # --- Block 2: A[2:3, :] * B -> C[2:3, :] (1x5x7) ---
    # Schoolbook: i=2, j=0..4, k=0..6
    for j in range(m):
        for k in range(p):
            u = zero_vec(n_U)
            v = zero_vec(n_V)
            w = zero_vec(n_W)
            i = 2
            u[i * m + j] = 1.0
            v[j * p + k] = 1.0
            w[k * n + i] = 1.0
            add_term(u, v, w)

    # --- Block 1.3: A[0:2, 4:5] * B[4:5, :] -> C[0:2, :] (2x1x7) ---
    # Schoolbook: i=0,1; j=4; k=0..6
    for i in range(2):
        for k in range(p):
            u = zero_vec(n_U)
            v = zero_vec(n_V)
            w = zero_vec(n_W)
            j = 4
            u[i * m + j] = 1.0
            v[j * p + k] = 1.0
            w[k * n + i] = 1.0
            add_term(u, v, w)

    # --- Block 1.1 and 1.2: 2x2x7 using Strassen on 2x2x2 blocks ---
    # We need a function to add Strassen terms for a 2x2x2 block
    # with specific indices i1, i2 for A rows, j1, j2 for A cols/B rows, k1, k2 for B cols/C cols
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2):
        # M1 = (a_i1j1 + a_i2j2) * (b_j1k1 + b_j2k2)
        u1 = zero_vec(n_U); v1 = zero_vec(n_V); w1 = zero_vec(n_W)
        u1[i1*m+j1] = 1.0; u1[i2*m+j2] = 1.0
        v1[j1*p+k1] = 1.0; v1[j2*p+k2] = 1.0
        w1[k1*n+i1] = 1.0; w1[k2*n+i1] = 1.0
        w1[k1*n+i2] = 1.0; w1[k2*n+i2] = 1.0
        # Wait, W is the output coefficient.
        # The term is sum_{i,j,k} U[i,j] V[j,k] W[k,i] A[i,j] B[j,k]
        # For a rank-1 term, U, V, W are fixed vectors.
        # The product is (sum U[i,j] A[i,j]) * (sum V[j,k] B[j,k]) * (sum W[k,i] C[k,i]?? No)
        # The tensor entry T[i,j,k] is reconstructed as sum_l U[i,j,l] V[j,k,l] W[k,i,l].
        # So for each term l, we add U[i,j,l]*V[j,k,l]*W[k,i,l] to T[i,j,k].
        
        # Strassen M1 contributes to:
        # c_i1k1 += a_i1j1 b_j1k1
        # c_i1k1 += a_i2j2 b_j1k1
        # c_i1k1 += a_i1j1 b_j2k2
        # c_i1k1 += a_i2j2 b_j2k2
        # And similarly for other c entries.
        
        # Let's define the U, V, W vectors for each of the 7 Strassen products.
        
        # M1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # This product appears in:
        # C[i1,k1]: +1
        # C[i1,k2]: -1 (from M1-M2+M3+M6? No, let's check Strassen formulas)
        
        # Standard Strassen:
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
        
        # We need to map (i1,i2,j1,j2,k1,k2) to (11,22,11,22,11,22) etc.
        # i1->1, i2->2, j1->1, j2->2, k1->1, k2->2
        
        # Term 1 (M1):
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i1*m+j1] = 1.0; u[i2*m+j2] = 1.0
        v[j1*p+k1] = 1.0; v[j2*p+k2] = 1.0
        # W coefficients for C[i,k]:
        w[k1*n+i1] = 1.0   # C11
        w[k2*n+i1] = 0.0   # C12? No, M1 is not in C12 directly.
        # Let's re-evaluate.
        # The term is scalar product of U-vector and A, V-vector and B.
        # Then multiplied by W-vector to add to C.
        # So W[k*n+i] is the coefficient for C[i,k] (note index order k*n+i).
        
        # M1 contributes to:
        # C[i1,k1] with coeff 1
        # C[i1,k2] with coeff 0? No.
        # Let's trace:
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # = A[i1,j1]B[j1,k1] + A[i1,j1]B[j2,k2] + A[i2,j2]B[j1,k1] + A[i2,j2]B[j2,k2]
        # These are the "cross" terms.
        # In C[i1,k1], we want A[i1,j1]B[j1,k1] + A[i1,j2]B[j2,k1] + A[i2,j1]B[j1,k1] + A[i2,j2]B[j2,k1].
        # M1 provides A[i1,j1]B[j1,k1] and A[i2,j2]B[j2,k2] (wrong k) and others.
        
        # Actually, it is easier to just generate the 7 U,V,W vectors based on the formulas.
        
        # M1:
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i1*m+j1] = 1.0; u[i2*m+j2] = 1.0
        v[j1*p+k1] = 1.0; v[j2*p+k2] = 1.0
        w[k1*n+i1] = 1.0
        w[k1*n+i2] = 0.0 # M1 is not in C21?
        # Let's check C21 = M2 + M4. M1 not there.
        # C22 = M1 - M2 + M3 + M6. M1 is there.
        w[k2*n+i2] = 1.0
        # C12 = M3 + M5. M1 not there.
        add_term(u, v, w)

        # M2: (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i2*m+j1] = 1.0; u[i2*m+j2] = 1.0
        v[j1*p+k1] = 1.0
        w[k1*n+i2] = 1.0 # C21
        w[k2*n+i2] = -1.0 # C22
        add_term(u, v, w)

        # M3: A[i1,j1] * (B[j2,k1]-B[j2,k2])
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i1*m+j1] = 1.0
        v[j2*p+k1] = 1.0; v[j2*p+k2] = -1.0
        w[k1*n+i1] = 1.0 # C11? No, C11=M1+M4-M5+M7.
        # C12 = M3+M5.
        w[k1*n+i1] = 0.0
        w[k2*n+i1] = 1.0 # C12
        # C22 = M1-M2+M3+M6.
        w[k2*n+i2] = 1.0
        add_term(u, v, w)

        # M4: A[i2,j2] * (B[j1,k2]-B[j1,k1])
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i2*m+j2] = 1.0
        v[j1*p+k2] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i1] = 1.0 # C11
        w[k2*n+i2] = 1.0 # C22? No, C22 has M1-M2+M3+M6. M4 not there.
        # C21 = M2+M4.
        w[k1*n+i2] = 1.0
        add_term(u, v, w)

        # M5: (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i1*m+j1] = 1.0; u[i1*m+j2] = 1.0
        v[j2*p+k2] = 1.0
        w[k1*n+i1] = -1.0 # C11
        w[k2*n+i1] = 1.0 # C12
        add_term(u, v, w)

        # M6: (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i2*m+j1] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k1] = 1.0; v[j1*p+k2] = 1.0
        w[k2*n+i2] = 1.0 # C22
        add_term(u, v, w)

        # M7: (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
        u[i1*m+j2] = 1.0; u[i2*m+j2] = -1.0
        v[j2*p+k1] = 1.0; v[j2*p+k2] = 1.0
        w[k1*n+i1] = 1.0 # C11
        add_term(u, v, w)

    # Apply Strassen to Block 1.1 (i=0,1; j=0,1; k=0..6)
    # Split k into 0,1; 2,3; 4,5; 6
    add_strassen_222(0, 1, 0, 1, 0, 1)
    add_strassen_222(0, 1, 0, 1, 2, 3)
    add_strassen_222(0, 1, 0, 1, 4, 5)
    # Last column k=6 (2x2x1 schoolbook)
    for i in range(2):
        for j in range(2):
            u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
            k = 6
            u[i*m+j] = 1.0
            v[j*p+k] = 1.0
            w[k*n+i] = 1.0
            add_term(u, v, w)

    # Apply Strassen to Block 1.2 (i=0,1; j=2,3; k=0..6)
    add_strassen_222(0, 1, 2, 3, 0, 1)
    add_strassen_222(0, 1, 2, 3, 2, 3)
    add_strassen_222(0, 1, 2, 3, 4, 5)
    # Last column k=6
    for i in range(2):
        for j in range(2, 4):
            u = zero_vec(n_U); v = zero_vec(n_V); w = zero_vec(n_W)
            k = 6
            u[i*m+j] = 1.0
            v[j*p+k] = 1.0
            w[k*n+i] = 1.0
            add_term(u, v, w)

    # Convert to numpy arrays for return (or lists as required)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
