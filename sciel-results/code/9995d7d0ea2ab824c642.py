import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = DIMENSIONS
    rank = 49

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Define the 7x7 Strassen coefficients for the 2x2 block decomposition
    # These correspond to the 7 multiplications in Strassen's algorithm
    # beta, gamma, delta map the 2x2 block indices to the 7 scalar multiplications
    
    # We use the standard Strassen decomposition for 2x2 matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # The 4x4 matrix is divided into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2.
    # C = A * B = [[A00B00 + A01B10, A00B01 + A01B11],
    #              [A10B00 + A11B10, A10B01 + A11B11]]
    
    # We apply Strassen to the 2x2 block multiplication.
    # Each block multiplication (e.g., A00B00) is itself a 2x2 matrix multiplication.
    # So we apply Strassen recursively to each of the 7 block multiplications.
    # This results in 7 * 7 = 49 scalar multiplications.
    
    # U, V, W matrices for the 2x2 Strassen algorithm (rank 7)
    # Rows of U correspond to elements of A (4 elements: 00, 01, 10, 11)
    # Rows of V correspond to elements of B (4 elements: 00, 01, 10, 11)
    # Rows of W correspond to elements of C (4 elements: 00, 01, 10, 11)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # t=0 (M1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1 (M2)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2 (M3)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3 (M4)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4 (M5)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5 (M6)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6 (M7)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Coefficients for the 7 block multiplications
    # s is the block multiplication index (0..6)
    # I, J are the block row/col indices for A (0..1)
    # J, K are the block row/col indices for B (0..1)
    # K, I are the block row/col indices for C (0..1)
    
    # We define beta[s, I, J], gamma[s, J, K], delta[s, K, I]
    # such that C_block[K, I] = sum_s delta[s, K, I] * (sum_{i,j} beta[s, I, J] A[i,j]) * (sum_{j,k} gamma[s, J, K] B[j,k])
    # Wait, the indices in the code below use I, J for A block, J, K for B block, K, I for C block.
    # This matches the standard block multiplication C = A B.
    
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))
    
    # s=0 (M1): (A00+A11)(B00+B11) -> C00 += 1, C01 += 0, C10 += 0, C11 += 1
    # Actually, M1 contributes to all 4 blocks?
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Let's map s to the M index.
    # s=0 is M1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1  # A00 + A11
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1 # B00 + B11
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1 # C00 += 1, C11 += 1
    
    # s=1 is M2
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1  # A10 + A11
    gamma[1, 0, 0] = 1                     # B00
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1 # C10 += 1, C11 -= 1
    
    # s=2 is M3
    beta[2, 0, 0] = 1                      # A00
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1 # B10 - B11
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1  # C01 += 1, C11 += 1
    
    # s=3 is M4
    beta[3, 1, 1] = 1                      # A11
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1 # B01 - B00
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1  # C00 += 1, C10 += 1
    # Wait, C00 = M1 - M4 ... so delta[3, 0, 0] should be -1?
    # Let's re-verify the signs.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # So:
    # M1: C00+=1, C11+=1
    # M2: C10+=1, C11-=1
    # M3: C01+=1, C11+=1
    # M4: C00-=1, C10+=1
    # M5: C00-=1, C01+=1
    # M6: C11+=1
    # M7: C00+=1
    
    # Let's correct the delta coefficients.
    
    # s=0 (M1)
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1 (M2)
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2 (M3)
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3 (M4)
    delta[3, 0, 0] = -1; delta[3, 1, 0] = 1
    
    # s=4 (M5)
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1  # A00 + A01
    gamma[4, 1, 1] = 1                     # B11
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1 # C00 -= 1, C01 += 1
    
    # s=5 (M6)
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1 # A10 - A00
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1 # B00 + B01
    delta[5, 1, 1] = 1                     # C11 += 1
    
    # s=6 (M7)
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1 # A01 - A11
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1 # B10 + B11
    delta[6, 0, 0] = 1                     # C00 += 1
    
    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s, I, J] * U2[i_prime * 2 + j_prime, t]
                    if coeff != 0:
                        U[row_u, c] = coeff

            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s, J, K] * V2[j_prime * 2 + k_prime, t]
                    if coeff != 0:
                        V[row_v, c] = coeff

            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s, K, I] * W2[k_prime * 2 + i_prime, t]
                    if coeff != 0:
                        W[row_w, c] = coeff

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
