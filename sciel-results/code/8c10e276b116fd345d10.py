"""
6x6 matrix multiplication tensor decomposition.
Uses Strassen (rank 7) on 2x2 blocks of size 3x3.
Total rank = 7 * 3^3 = 189.
"""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication
    using the Strassen algorithm applied to 2x2 blocks of 3x3 matrices.
    
    The 6x6 matrices are partitioned into four 3x3 blocks:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    
    Strassen computes 7 products of 3x3 matrices, each computed via schoolbook
    (rank 27). Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (safety margin)
    
    # Strassen coefficients
    # beta[s][I][J]: coeff of A[I, J] in the s-th Strassen product's A-factor
    # gamma[s][K][L]: coeff of B[K, L] in the s-th Strassen product's B-factor
    # alpha[s][I][K]: coeff of s-th product in C[I, K]
    
    # Initialize
    beta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    alpha = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # s=0: M0 = (A00 + A11)(B00 + B11)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    # s=1: M1 = (A10 + A11)B00
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    # s=2: M2 = A00(B01 - B11)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    # s=3: M3 = A11(B10 - B00)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    # s=4: M4 = (A00 + A01)B11
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    # s=5: M5 = (A10 - A00)(B00 + B01)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    # s=6: M6 = (A01 - A11)(B10 + B11)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1
    
    # Build the factor matrices
    rank = 7 * 27  # 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    block_size = 3
    
    for s in range(7):
        for a in range(block_size):
            for b in range(block_size):
                for c in range(block_size):
                    r = s * 27 + a * 9 + b * 3 + c
                    
                    # U: A components - U[i*m + j, r]
                    for I in range(2):
                        for J in range(2):
                            coeff = beta[s][I][J]
                            if coeff != 0:
                                i_glob = I * block_size + a
                                j_glob = J * block_size + b
                                U[i_glob * m + j_glob, r] += coeff
                    
                    # V: B components - V[j*p + k, r]
                    for K in range(2):
                        for L in range(2):
                            coeff = gamma[s][K][L]
                            if coeff != 0:
                                j_glob = K * block_size + b
                                k_glob = L * block_size + c
                                V[j_glob * p + k_glob, r] += coeff
                    
                    # W: C components - W[k*n + i, r] (trace convention)
                    for I in range(2):
                        for K in range(2):
                            coeff = alpha[s][I][K]
                            if coeff != 0:
                                i_glob = I * block_size + a
                                k_glob = K * block_size + c
                                W[k_glob * n + i_glob, r] += coeff
    
    # Verify correctness using vectorized tensor entry check
    if time.time() < deadline - 10:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # Reshape for vectorized computation
        U_reshaped = U.reshape(n, m, rank)  # U_reshaped[i, j, r]
        V_reshaped = V.reshape(m, p, rank)  # V_reshaped[j, k, r]
        W_reshaped = W.reshape(p, n, rank)  # W_reshaped[k, i, r]
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum for efficient computation
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err > 1e-9:
            # Fallback to schoolbook if Strassen verification fails
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m * p + j * p + k
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
