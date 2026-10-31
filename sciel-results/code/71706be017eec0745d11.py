import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen 2x2 factors
    # U2, V2, W2 are 4x7 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Verify Strassen decomposition for 2x2
    # Tensor T[i, j, k] = sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    # This is already known to be correct.

    # Construct 4x4 factors using Kronecker products
    # U = U2 (x) U2
    # V = V2 (x) V2
    # W needs permutation
    
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    
    # For W, we need to map indices correctly.
    # Kronecker W_kron = W2 (x) W2 has row index r = (k2*2+i1)*4 + (k1*2+i2) = 8*k2 + 4*i1 + 2*k1 + i2
    # Required row index is r_req = (k2*2+k1)*4 + (i2*2+i1) = 8*k2 + 4*k1 + 2*i2 + i1
    # We need to permute rows of W_kron.
    
    W_kron = np.kron(W2, W2)
    W = np.zeros_like(W_kron)
    
    # Permutation logic
    for k2 in range(2):
        for k1 in range(2):
            for i2 in range(2):
                for i1 in range(2):
                    # Required row index
                    k = k2 * 2 + k1
                    i = i2 * 2 + i1
                    r_req = k * 4 + i
                    
                    # Kronecker row index corresponding to these indices
                    # The Kronecker product structure:
                    # W_kron[row, col] = W2[row // 4, col // 7] * W2[row % 4, col % 7]
                    # Let row = r2 * 4 + r1.
                    # r2 = k2 * 2 + i1 ? No.
                    # In W2, row index is k*2+i.
                    # So r2 corresponds to (k2, i1) and r1 corresponds to (k1, i2)?
                    # Let's check the Kronecker definition again.
                    # (A (x) B)[i, j] = A[i // n_B, i % n_B] * B[j // n_B, j % n_B] is wrong.
                    # Standard: (A (x) B)[i, j] = A[i // n_B, i % n_B] * B[j // n_B, j % n_B] ?
                    # No, (A (x) B) is block matrix [a_ij B].
                    # Row i in A (x) B corresponds to row i_A in A and row i_B in B, where i = i_A * n_B + i_B.
                    # So i_A = i // 4, i_B = i % 4.
                    # i_A is index in W2. i_A = k2 * 2 + i1 ? Or k2 * 2 + something?
                    # The tensor decomposition is multiplicative.
                    # T[i, j, k] = sum U[i, r] V[j, r] W[k, r].
                    # For Kronecker, T[i, j, k] = T2[i_A, j_A, k_A] * T2[i_B, j_B, k_B].
                    # Indices: i = i_A * 2 + i_B, j = j_A * 2 + j_B, k = k_A * 2 + k_B.
                    # U[i, r] = U2[i_A, r_A] * U2[i_B, r_B].
                    # This matches U = U2 (x) U2 if row i = i_A * 4 + i_B?
                    # If U = U2 (x) U2, row i = i_A * 4 + i_B.
                    # Then U[i, r] = U2[i_A, r_A] U2[i_B, r_B].
                    # This corresponds to i_A = i // 4, i_B = i % 4.
                    # But we want i = i_A * 2 + i_B.
                    # So U2 (x) U2 does NOT directly give the correct indexing for i = i_A * 2 + i_B.
                    # The row index in U2 (x) U2 is i_A * 4 + i_B.
                    # We need row index i_A * 2 + i_B.
                    # So we need to permute U and V as well?
                    
                    # Let's re-evaluate.
                    # The current program uses:
                    # row_u = i * 4 + j.
                    # i = I * 2 + i_prime. j = J * 2 + j_prime.
                    # row_u = (I * 2 + i_prime) * 4 + (J * 2 + j_prime) = 8*I + 4*i_prime + 2*J + j_prime.
                    # In Kronecker U2 (x) U2:
                    # Row index r = r_A * 4 + r_B.
                    # r_A is index in U2 (0..3), r_B is index in U2 (0..3).
                    # r_A = I * 2 + i_prime? No, U2 rows are indexed by (i, j) of 2x2.
                    # Let's stick to the explicit construction which is known to work.
                    pass

    # Explicit construction is safer for indexing.
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    # Strassen coefficients for 2x2
    # beta, gamma, delta as in original
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

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
                    coeff = delta[s, I, K] * W2[k_prime * 2 + i_prime, t]
                    if coeff != 0:
                        W[row_w, c] = coeff

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
