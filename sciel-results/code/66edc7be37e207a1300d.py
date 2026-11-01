"""Numerical tensor-rank search for 2x4x4 matrix multiplication via Burer-Monteiro + Adam."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135  # 15s safety margin before 150s kill

    n, m, p = 2, 4, 4

    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # (The matrix multiplication tensor for A(n×m) * B(m×p) -> C(n×p)
    #  has T[i,j,k] = 1 iff A[i,j]*B[j,k] contributes to C[i,k], which is always true.)
    T = np.ones((n, m, p))

    def verify(U, V, W, R):
        """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        # T_recon[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum
        recon = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3)
        return np.max(np.abs(recon - T))

    def strassen_init(R):
        """Initialize from Strassen rank-28, padded to R if R >= 28, or truncated with noise if R < 28."""
        # Build the rank-28 Strassen factors
        U28 = np.zeros((n * m, 28))
        V28 = np.zeros((m * p, 28))
        W28 = np.zeros((p * n, 28))

        # Strassen local block (2x2x2)
        U_loc = np.zeros((2, 2, 7))
        V_loc = np.zeros((2, 2, 7))
        W_loc = np.zeros((2, 2, 7))

        # r=0 (M1): (A00+A11)(B00+B11)
        U_loc[0, 0, 0] = 1; U_loc[1, 1, 0] = 1
        V_loc[0, 0, 0] = 1; V_loc[1, 1, 0] = 1
        W_loc[0, 0, 0] = 1; W_loc[1, 0, 0] = 1; W_loc[0, 1, 0] = 1; W_loc[1, 1, 0] = -1

        # r=1 (M2): (A10+A11)B00
        U_loc[1, 0, 1] = 1; U_loc[1, 1, 1] = 1
        V_loc[0, 0, 1] = 1
        W_loc[0, 1, 1] = 1; W_loc[1, 1, 1] = -1

        # r=2 (M3): A00(B10-B11)
        U_loc[0, 0, 2] = 1
        V_loc[1, 0, 2] = 1; V_loc[1, 1, 2] = -1
        W_loc[1, 0, 2] = 1; W_loc[1, 1, 2] = 1

        # r=3 (M4): A11(B01-B00)
        U_loc[1, 1, 3] = 1
        V_loc[0, 1, 3] = 1; V_loc[0, 0, 3] = -1
        W_loc[0, 0, 3] = 1; W_loc[0, 1, 3] = 1

        # r=4 (M5): (A00+A01)B11
        U_loc[0, 0, 4] = 1; U_loc[0, 1, 4] = 1
        V_loc[1, 1, 4] = 1
        W_loc[0, 0, 4] = -1; W_loc[1, 0, 4] = 1

        # r=5 (M6): (A10-A00)(B00+B10)
        U_loc[1, 0, 5] = 1; U_loc[0, 0, 5] = -1
        V_loc[0, 0, 5] = 1; V_loc[1, 0, 5] = 1
        W_loc[1, 1, 5] = 1

        # r=6 (M7): (A01-A11)(B01+B11)
        U_loc[0, 1, 6] = 1; U_loc[1, 1, 6] = -1
        V_loc[0, 1, 6] = 1; V_loc[1, 1, 6] = 1
        W_loc[0, 0, 6] = 1

        # Assemble 4 blocks
        blocks = [(0, 0, 0), (2, 0, 0), (0, 2, 2), (2, 2, 2)]
        for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
            r_off = b_idx * 7
            for i in range(2):
                for j in range(2):
                    U28[i * m + (a_off + j), r_off + j if False else r_off + 0] = 0  # placeholder
            # Actually let's do it properly
        # Redo assembly
        U28 = np.zeros((n * m, 28))
        V28 = np.zeros((m * p, 28))
        W28 = np.zeros((p * n, 28))
        for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
            r_off = b_idx * 7
            for i in range(2):
                for j in range(2):
                    U28[i * m + (a_off + j), r_off + 0] = U_loc[i, j, 0]
                    U28[i * m + (a_off + j), r_off + 1] = U_loc[i, j, 1]
                    U28[i * m + (a_off + j), r_off + 2] = U_loc[i, j, 2]
                    U28[i * m + (a_off + j), r_off + 3] = U_loc[i, j, 3]
                    U28[i * m + (a_off + j), r_off + 4] = U_loc[i, j, 4]
                    U28[i * m + (a_off + j), r_off + 5] = U_loc[i, j, 5]
                    U28[i * m + (a_off + j), r_off + 6] = U_loc[i, j, 6]
            for j in range(2):
                for k in range(2):
                    V28[(b_off + j) * p + (c_off + k), r_off + 0] = V_loc[j, k, 0]
                    V28[(b_off + j) * p + (c_off + k), r_off + 1] = V_loc[j, k, 1]
                    V28[(b_off + j) * p + (c_off + k), r_off + 2] = V_loc[j, k, 2]
                    V28[(b_off + j) * p + (c_off + k), r_off + 3] = V_loc[j, k, 3]
                    V28[(b_off + j) * p + (c_off + k), r_off + 4] = V_loc[j, k, 4]
                    V28[(b_off + j) * p + (c_off + k), r_off + 5] = V_loc[j, k, 5]
                    V28[(b_off + j) * p + (c_off + k), r_off + 6] = V_loc[j, k, 6]
            for k in range(2):
                for i in range(2):
                    W28[(c_off + k) * n + i, r_off + 0] = W_loc[k, i, 0]
                    W28[(c_off + k) * n + i, r_off + 1] = W_loc[k, i, 1]
                    W28[(c_off + k) * n + i, r_off + 2] = W_loc[k, i, 2]
                    W28[(c_off + k) * n + i, r_off + 3] = W_loc[k, i, 3]
                    W28[(c_off + k) * n + i, r_off + 4] = W_loc[k, i, 4]
                    W28[(c_off + k) * n + i, r_off + 5] = W_loc[k, i, 5]
                    W28[(c_off + k) * n + i, r_off + 6] = W_loc[k, i, 6]

        # Verify rank-28
        err28 = verify(U28, V28, W28, 28)
        # Should be ~0

        if R >= 28:
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            U[:, :28] = U28
            V[:, :28] = V28
            W[:, :28] = W28
            # Pad remaining columns with small random values
            if R > 28:
                U[:, 28:] = np.random.randn(n * m, R - 28) * 0.01
                V[:, 28:] = np.random.randn(m * p, R - 28) * 0.01
                W[:, 28:] = np.random.randn(p * n, R - 28) * 0.01
        else:
            # R < 28: truncate and add noise
            U = U28[:, :R].copy()
            V = V28[:, :R].copy()
            W = W28[:, :R].copy()
            # Add small random perturbation to escape the rank-28 manifold
            U += np.random.randn(n * m, R) * 0.05
            V += np.random.randn(m * p, R) * 0.05
            W += np.random.randn(p * n, R) * 0.05

        return U, V, W

    def random_init(R, seed=None):
        """Random initialization for rank R."""
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        return U, V, W

    def adam_optimize(U, V, W, R, lr=0.01, n_steps=3000, tol=1e-10):
        """
        Adam optimization to minimize ||sum_r U_i,j,r * V_j,k,r * W_k,i,r - 1||^2
        using Burer-Monteiro parametrization (direct factor matrices).
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()

        # Adam state
        U_m = np.zeros_like(U); U_v = np.zeros_like(U)
        V_m = np.zeros_like(V); V_v = np.zeros_like(V)
        W_m = np.zeros_like(W); W_v = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0

        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)

        for step in range(n_steps):
            t += 1
            # Compute recon and residual
            recon = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3)
            resid = recon - T  # shape (n, m, p)

            # Gradients:
            # d/dU[i,j,r] = sum over (k) of resid[i,j,k] * V[j,k,r] * W[k,i,r]
            # d/dV[j,k,r] = sum over (i) of resid[i,j,k] * U[i,j,r] * W[k,i,r]
            # d/dW[k,i,r] = sum over (j) of resid[i,j,k] * U[i,j,r] * V[j,k,r]

            # grad_U[i,j,r] = sum_k resid[i,j,k] * V[j,k,r] * W[k,i,r]
            grad_U = np.einsum('ijk,jkr,ikr->ijr', resid, V3, W3)
            grad_V = np.einsum('ijk,ijr,ikr->jkr', resid, U3, W3)
            grad_W = np.einsum('ijk,ijr,jkr->ikr', resid, U3, V3)

            # Adam update
            U_m = b1 * U_m + (1 - b1) * grad_U
            U_v = b2 * U_v + (1 - b2) * grad_U ** 2
            U_m_hat = U_m / (1 - b1 ** t)
            U_v_hat = U_v / (1 - b2 ** t)
            U -= lr * U_m_hat / (np.sqrt(U_v_hat) + eps)

            V_m = b1 * V_m + (1 - b1) * grad_V
            V_v = b2 * V_v + (1 - b2) * grad_V ** 2
            V_m_hat = V_m / (1 - b1 ** t)
            V_v_hat = V_v / (1 - b2 ** t)
            V -= lr * V_m_hat / (np.sqrt(V_v_hat) + eps)

            W_m = b1 * W_m + (1 - b1) * grad_W
            W_v = b2 * W_v + (1 - b2) * grad_W ** 2
            W_m_hat = W_m / (1 - b1 ** t)
            W_v_hat = W_v / (1 - b2 ** t)
            W -= lr * W_m_hat / (np.sqrt(W_v_hat) + eps)

            # Update views
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)

            # Check convergence
            if step % 200 == 0:
                err = np.max(np.abs(recon - T))
                if err < tol:
                    break

        # Final error
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        recon = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3)
        err = np.max(np.abs(recon - T))

        return U, V, W, err

    def try_rank(R, n_restarts=5, steps_per_restart=2000, lr=0.01):
        """Try to find a valid rank-R decomposition."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                # Start from Strassen init (perturbed)
                U, V, W = strassen_init(R)
            else:
                # Random init with different seeds
                U, V, W = random_init(R, seed=42 + restart * 1000)

            U, V, W, err = adam_optimize(U, V, W, R, lr=lr, n_steps=steps_per_restart)

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U, V, W

            if best_err < 1e-8:
                break

        return best_U, best_V, best_W, best_err

    # Fallback: schoolbook rank 32
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    # Fallback: Strassen rank 28
    def strassen_rank28():
        U, V, W = strassen_init(28)
        err = verify(U, V, W, 28)
        return U, V, W, 28

    best_result = None
    best_rank = 32

    # Try rank 27 (the target improvement)
    if time.time() < deadline - 30:
        U, V, W, err = try_rank(27, n_restarts=3, steps_per_restart=2500, lr=0.005)
        if err < 1e-6:
            # Round to nearest half-integer for cleanliness (optional)
            best_rank = 27
            best_result = (U, V, W)
        elif err < best_rank:  # Not valid, but keep track
            pass

    # Try rank 27 with different lr
    if time.time() < deadline - 30 and best_rank > 27:
        U, V, W, err = try_rank(27, n_restarts=2, steps_per_restart=2000, lr=0.002)
        if err < 1e-6:
            best_rank = 27
            best_result = (U, V, W)

    # Try rank 27 with more steps if time permits
    if time.time() < deadline - 20 and best_rank > 27:
        U, V, W, err = try_rank(27, n_restarts=1, steps_per_restart=5000, lr=0.003)
        if err < 1e-6:
            best_rank = 27
            best_result = (U, V, W)

    # If rank 27 failed, try rank 28 (Strassen)
    if best_rank > 28:
        U, V, W, R = strassen_rank28()
        if verify(U, V, W, R) < 1e-9:
            best_rank = 28
            best_result = (U, V, W)

    # Final fallback: schoolbook
    if best_result is None:
        U, V, W, R = schoolbook()
        best_rank = R
        best_result = (U, V, W)

    U, V, W = best_result

    # Convert to list of lists (as required by interface)
    U_list = [[float(U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(best_rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
