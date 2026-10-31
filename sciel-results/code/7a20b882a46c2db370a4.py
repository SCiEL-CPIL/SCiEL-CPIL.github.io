import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS

    # Build the 2x2x2 Strassen decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # t=0 (M1 inner)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # t=1 (M2 inner)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # t=2 (M3 inner)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # t=3 (M4 inner)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # t=4 (M5 inner)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # t=5 (M6 inner)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # t=6 (M7 inner)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 using 7 products of 2x2x2 (rank 49)
    rank_full = 49
    U = np.zeros((16, rank_full))
    V = np.zeros((16, rank_full))
    W = np.zeros((16, rank_full))

    # Strassen coefficients for 4x4 -> 2x2 blocks
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

    if time.time() > deadline - 20.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i*4+j, j*4+k, k*4+i] = 1.0

    def verify(U_mat, V_mat, W_mat):
        """Check if U,V,W exactly decompose T."""
        # Reconstruct: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i, r] * V_mat[j, r] * W_mat[k, r]
                    if abs(val - T[i, j, k]) > 1e-10:
                        return False
        return True

    # Fast verification using numpy einsum
    def verify_fast(U_mat, V_mat, W_mat):
        T_rec = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
        return np.allclose(T_rec, T, atol=1e-10)

    if not verify_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check for zero columns
    nonzero_cols = []
    for r in range(49):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            nonzero_cols.append(r)

    if len(nonzero_cols) < 49:
        U = U[:, nonzero_cols]
        V = V[:, nonzero_cols]
        W = W[:, nonzero_cols]
        if verify_fast(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt greedy rank reduction: for each column, check if its rank-1
    # contribution can be expressed as a linear combination of the others.
    # Flatten each rank-1 contribution to a 4096-dim vector and check
    # linear dependence via rank of the matrix of all contributions.
    if time.time() < deadline - 30.0:
        # Build matrix M where column r = vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r])
        M = np.zeros((4096, 49))
        for r in range(49):
            M[:, r] = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()

        # Compute rank of M
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        if rank_M < 49:
            # There is linear dependence. Find a minimal spanning set.
            # Use QR with column pivoting to find independent columns.
            Q, R, piv = np.linalg.qr(M, mode='complete')
            # Determine which columns are independent
            diag_R = np.abs(np.diag(R))
            threshold = 1e-10 * max(diag_R) if len(diag_R) > 0 else 1e-10
            independent = diag_R > threshold
            # Map pivoted indices back to original
            indep_cols = piv[:int(np.sum(independent))]
            if len(indep_cols) < 49:
                U_new = U[:, indep_cols]
                V_new = V[:, indep_cols]
                W_new = W[:, indep_cols]
                if verify_fast(U_new, V_new, W_new):
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
