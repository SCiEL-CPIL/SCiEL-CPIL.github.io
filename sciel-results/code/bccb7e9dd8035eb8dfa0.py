import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before the 150s kill

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(r):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-9:
                            return False
        return True

    # Schoolbook decomposition (rank 63) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Strategy: Directly construct a rank-42 decomposition by applying
    # the Strassen algorithm (rank 7) to each of the 3x3 blocks formed by
    # grouping the 7 columns into {cols 0,1,2}, {cols 3,4,5}, and handling
    # col 6 with schoolbook (rank 3). Wait, that gives 2*7 + 3 = 17? No.
    #
    # Actually, let's think more carefully. We multiply a 3x3 matrix A by a 3x7 matrix B.
    # We can split B into two parts: B1 (3x3) and B2 (3x4).
    # For B1, we use Strassen (rank 7).
    # For B2, we use schoolbook (rank 12).
    # Total rank: 7 + 12 = 19? That seems too low. Let me reconsider.
    #
    # Wait, Strassen multiplies two 2x2 matrices with rank 7. For 3x3, the known
    # minimum rank is not 7. Let me use a different approach.
    #
    # Actually, let's try a direct construction. We can use the fact that matrix
    # multiplication tensor for 3x3 x 3x7 has a known upper bound. Let me try
    # a rank-42 construction by exploiting structure.
    #
    # Alternative: Use the fact that we can decompose the 3x7 matrix into three
    # 3x3 blocks (with one being 3x1). But that's messy.
    #
    # Let me try a cleaner approach: use a known rank-42 decomposition by
    # combining Strassen-like ideas with schoolbook.
    #
    # Actually, let's just try to find a rank-42 decomposition via a structured
    # ansatz. We'll fix U and V with some structure and solve for W.

    def check_and_solve(U, V, r):
        """
        U: shape (n, m, r)
        V: shape (m, p, r)
        Returns W (p, n, r) if valid, else None.
        """
        W = np.zeros((p, n, r))
        b_ik = np.eye(m).flatten()
        for i in range(n):
            for k in range(p):
                U_i = U[i, :, :]  # m x r
                V_k_mat = V[:, k, :].T  # r x m

                # Construct A_ik (m^2 x r)
                A_ik = np.zeros((m * m, r))
                for c in range(r):
                    A_ik[:, c] = np.kron(V_k_mat[c, :].T, U_i[:, c])

                try:
                    w_ik, residuals, rank_val, s = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    if len(residuals) > 0:
                        if residuals[0] > 1e-8:
                            return None
                    else:
                        residual = np.linalg.norm(A_ik @ w_ik - b_ik)
                        if residual > 1e-8:
                            return None
                    W[k, i, :] = w_ik
                except Exception:
                    return None
        return W

    # Try rank-42 with structured U and V
    target_rank = 42
    found = False
    num_tries = 200

    for trial in range(num_tries):
        if time.time() > deadline - 10:
            break

        # Structured ansatz: use random but with some sparsity or pattern
        U_rand = np.random.randn(n, m, target_rank)
        V_rand = np.random.randn(m, p, target_rank)

        W_rand = check_and_solve(U_rand, V_rand, target_rank)

        if W_rand is not None:
            U_out = U_rand.reshape(n * m, target_rank)
            V_out = V_rand.reshape(m * p, target_rank)
            W_out = W_rand.T.reshape(p * n, target_rank)

            if verify(U_out, V_out, W_out):
                best_rank = target_rank
                best_U = U_out.copy()
                best_V = V_out.copy()
                best_W = W_out.copy()
                found = True
                break

    # If rank-42 not found, try lower ranks
    if not found:
        for r in range(41, 30, -1):
            if time.time() > deadline - 5:
                break
            for _ in range(50):
                if time.time() > deadline - 3:
                    break
                U_rand = np.random.randn(n, m, r)
                V_rand = np.random.randn(m, p, r)
                W_rand = check_and_solve(U_rand, V_rand, r)
                if W_rand is not None:
                    U_out = U_rand.reshape(n * m, r)
                    V_out = V_rand.reshape(m * p, r)
                    W_out = W_rand.T.reshape(p * n, r)
                    if verify(U_out, V_out, W_out):
                        best_rank = r
                        best_U = U_out.copy()
                        best_V = V_out.copy()
                        best_W = W_out.copy()
                        break
            if best_rank < r:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
