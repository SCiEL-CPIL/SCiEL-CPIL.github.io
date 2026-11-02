"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        """Check exact decomposition."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def verify_strict(U, V, W, tol=1e-6):
        """Check exact decomposition with tighter tolerance."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        max_err = np.max(np.abs(recon - T))
        return max_err < tol, max_err

    def to_lists(U, V, W, r):
        """Convert numpy arrays to list-of-lists format."""
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # === STRASSEN-BASED CONSTRUCTION (rank 28) ===
    def strassen_2x2_decomp():
        """
        Return Strassen decomposition of 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Strassen's 7 products
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0

        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0

        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0

        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0

        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0

        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0

        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0

        return U, V, W

    U_s, V_s, W_s = strassen_2x2_decomp()

    full_rank = 28
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))

    # Block 1: A_left * B_top_left -> C_left
    for t in range(7):
        col = 0 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + lj
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = lj * 4 + lk
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = lk * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 2: A_right * B_bot_left -> C_left
    for t in range(7):
        col = 1 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + (lj + 2)
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = (lj + 2) * 4 + lk
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = lk * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 3: A_left * B_top_right -> C_right
    for t in range(7):
        col = 2 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + lj
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = lj * 4 + (lk + 2)
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = (lk + 2) * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 4: A_right * B_bot_right -> C_right
    for t in range(7):
        col = 3 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + (lj + 2)
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = (lj + 2) * 4 + (lk + 2)
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = (lk + 2) * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    if verify(U_full, V_full, W_full, tol=1e-8):
        return to_lists(U_full, V_full, W_full, full_rank)

    # === ATTEMPT RANK-27 SEARCH ===
    def attempt_rank27():
        """
        Try to find a rank-27 decomposition via gradient descent with sparsification.
        Strategy: Start from rank-28 Strassen, merge two components, then optimize.
        """
        r = 27
        rng = np.random.default_rng(42)

        # Start from Strassen rank-28 and try to merge columns
        # Pick a random pair of columns to merge
        U27 = U_full[:, :27].copy()
        V27 = V_full[:, :27].copy()
        W27 = W_full[:, :27].copy()

        # Add small random perturbation to escape the rank-28 manifold
        U27 += rng.standard_normal((n * m, r)) * 0.01
        V27 += rng.standard_normal((m * p, r)) * 0.01
        W27 += rng.standard_normal((p * n, r)) * 0.01

        # Gradient descent with adaptive step size and sparsification
        best_err = np.inf
        best_U, best_V, best_W = U27.copy(), V27.copy(), W27.copy()

        for iteration in range(3000):
            if time.time() > deadline - 30:
                break

            # Compute reconstruction error
            recon = np.einsum('at,bt,ct->abc', U27, V27, W27)
            err = recon - T
            loss = 0.5 * np.sum(err ** 2)

            if loss < best_err:
                best_err = loss
                best_U, best_V, best_W = U27.copy(), V27.copy(), W27.copy()

            # Check for convergence
            if loss < 1e-20:
                break

            # Compute gradients
            # dL/dU[a,t] = sum_{b,c} err[a,b,c] * V[b,t] * W[c,t]
            # Use einsum for efficiency
            dU = np.einsum('abc,bt,ct->at', err, V27, W27)
            dV = np.einsum('abc,at,ct->bt', err, U27, W27)
            dW = np.einsum('abc,at,bt->ct', err, U27, V27)

            # Adaptive step size based on gradient norm
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-15:
                break
            step = 0.01 / max(grad_norm, 1e-10)

            # Update
            U27 -= step * dU
            V27 -= step * dV
            W27 -= step * dW

            # Periodic sparsification: zero out small entries
            if iteration % 200 == 199:
                threshold = 0.001
                U27[np.abs(U27) < threshold] = 0.0
                V27[np.abs(V27) < threshold] = 0.0
                W27[np.abs(W27) < threshold] = 0.0

        # Check if we found a valid rank-27 solution
        ok, max_err = verify_strict(best_U, best_V, best_W, tol=1e-6)
        if ok:
            return best_U, best_V, best_W, r

        # Try a different initialization: random rank-27
        for attempt in range(5):
            if time.time() > deadline - 30:
                break

            U27 = rng.standard_normal((n * m, r))
            V27 = rng.standard_normal((m * p, r))
            W27 = rng.standard_normal((p * n, r))

            for iteration in range(2000):
                if time.time() > deadline - 30:
                    break

                recon = np.einsum('at,bt,ct->abc', U27, V27, W27)
                err = recon - T
                loss = 0.5 * np.sum(err ** 2)

                if loss < best_err:
                    best_err = loss
                    best_U, best_V, best_W = U27.copy(), V27.copy(), W27.copy()

                if loss < 1e-20:
                    break

                dU = np.einsum('abc,bt,ct->at', err, V27, W27)
                dV = np.einsum('abc,at,ct->bt', err, U27, W27)
                dW = np.einsum('abc,at,bt->ct', err, U27, V27)

                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm < 1e-15:
                    break
                step = 0.01 / max(grad_norm, 1e-10)

                U27 -= step * dU
                V27 -= step * dV
                W27 -= step * dW

                if iteration % 200 == 199:
                    threshold = 0.001
                    U27[np.abs(U27) < threshold] = 0.0
                    V27[np.abs(V27) < threshold] = 0.0
                    W27[np.abs(W27) < threshold] = 0.0

            ok, max_err = verify_strict(best_U, best_V, best_W, tol=1e-6)
            if ok:
                return best_U, best_V, best_W, r

        return None

    result_27 = attempt_rank27()
    if result_27 is not None:
        U27, V27, W27, r27 = result_27
        if verify(U27, V27, W27, tol=1e-8):
            return to_lists(U27, V27, W27, r27)

    # Fallback: schoolbook rank 32
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U, V, W = build_schoolbook()
    return to_lists(U, V, W, n * m * p)
# EVOLVE-BLOCK-END
