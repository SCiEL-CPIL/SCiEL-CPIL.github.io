import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2 Strassen factors
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: (A11+A22)(B11+B22)
    # A11: (0,0), A22: (1,1) in 2x2 block? No, 2x2 matrix indices are 0,1.
    # A11 is element (0,0). A22 is element (1,1).
    # So for U2, we need coefficients for A elements.
    # A elements are (0,0), (0,1), (1,0), (1,1).
    # Indices in U2: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
    
    # P1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1 # C00 += P1
    W2[3, 0] = 1 # C11 += P1
    
    # P2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C10 += P2
    W2[3, 1] = -1 # C11 -= P2
    
    # P3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01 += P3
    W2[3, 2] = 1 # C11 += P3
    
    # P4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C00 += P4
    W2[2, 3] = 1 # C10 += P4
    
    # P5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00 -= P5
    W2[1, 4] = 1 # C01 += P5
    
    # P6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 += P6
    
    # P7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 += P7
    W2[3, 6] = 1 # C11 += P7

    # 2. Define block-level Strassen factors for 2x2 blocks
    # These are the same as U2, V2, W2 but interpreted at the block level.
    # U_b, V_b, W_b are 4x7 matrices.
    # Indices: U_b[I*2+J, r], V_b[J*2+K, r], W_b[K*2+I, r]
    # I, J, K in {0, 1}
    U_b = U2.copy()
    V_b = V2.copy()
    W_b = W2.copy()

    # 3. Construct 4x4x4 factors
    # Rank 49
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            R = 7 * r + s
            # Iterate over global indices i, j, k for 4x4
            for i in range(4):
                I = i // 2
                alpha = i % 2
                for j in range(4):
                    J = j // 2
                    beta = j % 2
                    # U coefficient
                    u_val = U_b[I * 2 + J, r] * U2[alpha * 2 + beta, s]
                    U_44[i * 4 + j, R] = u_val
                    
                    for k in range(4):
                        K = k // 2
                        gamma = k % 2
                        # V coefficient
                        v_val = V_b[J * 2 + K, r] * V2[beta * 2 + gamma, s]
                        V_44[j * 4 + k, R] = v_val
                        # W coefficient
                        w_val = W_b[K * 2 + I, r] * W2[gamma * 2 + alpha, s]
                        W_44[k * 4 + i, R] = w_val

    # 4. Extend to 4x4x9
    # B is 4x9. Split into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A B1 + A B2 + A B3
    # Rank = 49 + 49 + 4 = 102
    rank = 102
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Part 1: A * B1 (k=0..3)
    # U is same as U_44
    # V indices: j*9 + k. For k=0..3, V_44 has indices j*4 + k.
    # So V[j*9 + k, R] = V_44[j*4 + k, R]
    # W indices: k*4 + i. Same as W_44.
    for R in range(49):
        U[:, R] = U_44[:, R]
        for j in range(4):
            for k in range(4):
                V[j * 9 + k, R] = V_44[j * 4 + k, R]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, R] = W_44[k * 4 + i, R]
                
    # Part 2: A * B2 (k=4..7)
    # U is same as U_44
    # V indices: j*9 + (4+k'). k'=0..3.
    # V[j*9 + 4 + k', R+49] = V_44[j*4 + k', R]
    # W indices: (4+k')*4 + i.
    # W[(4+k')*4 + i, R+49] = W_44[k'*4 + i, R]
    for R in range(49):
        U[:, 49 + R] = U_44[:, R]
        for j in range(4):
            for k_prime in range(4):
                k = 4 + k_prime
                V[j * 9 + k, 49 + R] = V_44[j * 4 + k_prime, R]
        for k_prime in range(4):
            k = 4 + k_prime
            for i in range(4):
                W[k * 4 + i, 49 + R] = W_44[k_prime * 4 + i, R]
                
    # Part 3: A * B3 (k=8)
    # B3 is 4x1.
    # C[i, 8] = sum_j A[i, j] B[j, 8]
    # 4 terms.
    # U[i*4 + j, 98+j] = 1
    # V[j*9 + 8, 98+j] = 1
    # W[8*4 + i, 98+j] = 1
    for j in range(4):
        R = 98 + j
        for i in range(4):
            U[i * 4 + j, R] = 1
        V[j * 9 + 8, R] = 1
        for i in range(4):
            W[8 * 4 + i, R] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
