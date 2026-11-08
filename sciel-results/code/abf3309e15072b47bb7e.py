import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    def check_decomposition(U, V, W, rank):
        """Verify the decomposition exactly reproduces matrix multiplication."""
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Check half-integer constraint
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-8):
            return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-8):
            return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-8):
            return False

        # Check correctness with random test matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B

        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc

        if not np.allclose(C, Cf, atol=1e-6):
            return False
        return True

    def try_rank(rank, time_budget=30.0):
        """Try to find a decomposition of given rank via L-BFGS-B optimization."""
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_size = size_u + size_v + size_w

        def objective(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u + size_v].reshape(m, p, rank)
            W_flat = params[size_u + size_v:].reshape(p, n, rank)

            # Compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Using broadcasting
            U_b = U_flat[:, :, np.newaxis, np.newaxis, :]  # (n, m, 1, 1, rank)
            V_b = V_flat[np.newaxis, :, :, np.newaxis, :]  # (1, m, p, 1, rank)
            W_b = W_flat[np.newaxis, np.newaxis, :, :, :]  # (1, 1, p, n, rank)

            prod = U_b * V_b * W_b  # (n, m, p, n, rank)
            S = np.sum(prod, axis=4)  # (n, m, p, n)

            # Extract diagonal: S[i, j, k, i]
            idx = np.arange(n)
            S_diag = S[idx, :, :, idx]  # (n, m, p)

            err = np.sum((S_diag - 1.0) ** 2)
            return err

        def gradient(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u + size_v].reshape(m, p, rank)
            W_flat = params[size_u + size_v:].reshape(p, n, rank)

            U_b = U_flat[:, :, np.newaxis, np.newaxis, :]
            V_b = V_flat[np.newaxis, :, :, np.newaxis, :]
            W_b = W_flat[np.newaxis, np.newaxis, :, :, :]

            prod = U_b * V_b * W_b
            S = np.sum(prod, axis=4)

            idx = np.arange(n)
            S_diag = S[idx, :, :, idx]

            err_term = 2.0 * (S_diag - 1.0)  # (n, m, p)

            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)

            E_b = err_term[:, :, :, np.newaxis, np.newaxis]  # (n, m, p, 1, 1)
            V_b2 = V_flat[np.newaxis, :, :, np.newaxis, :]  # (1, m, p, 1, rank)
            W_b2 = W_flat[np.newaxis, np.newaxis, :, :, :]  # (1, 1, p, n, rank)
            U_b2 = U_flat[:, :, np.newaxis, np.newaxis, :]  # (n, m, 1, 1, rank)

            # grad_U[i, j, r] = sum_k err_term[i, j, k] * V[j, k, r] * W[k, i, r]
            prod_gU = E_b * V_b2 * W_b2  # (n, m, p, n, rank)
            grad_U_flat = np.sum(prod_gU, axis=2)  # (n, m, n, rank)
            grad_U = grad_U_flat[idx, :, idx, :]  # (n, m, rank)

            # grad_V[j, k, r] = sum_i err_term[i, j, k] * U[i, j, r] * W[k, i, r]
            for i in range(n):
                E_i = err_term[i, :, :]  # (m, p)
                U_i = U_flat[i, :, :]  # (m, rank)
                W_i = W_flat[:, i, :]  # (p, rank)
                term_V = E_i[:, :, np.newaxis] * U_i[:, np.newaxis, :] * W_i[np.newaxis, :, :]
                grad_V += term_V  # (m, p, rank)

                # grad_W[k, i, r] += sum_j err_term[i, j, k] * U[i, j, r] * V[j, k, r]
                term_W = E_i[:, :, np.newaxis] * U_i[:, np.newaxis, :] * V_flat[:, :, :]
                grad_W[:, i, :] += np.sum(term_W, axis=0)  # (p, rank)

            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        best_params = None
        best_fun = np.inf

        from scipy.optimize import minimize

        n_restarts = 15
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            elapsed = time.time() - (deadline - time_budget)
            if elapsed > time_budget:
                break

            np.random.seed(rank * 7919 + restart * 104729)
            # Try different initialization scales
            scale = 0.5 * (1.0 + 0.3 * np.sin(restart * 0.7))
            params = np.random.randn(total_size) * scale

            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14,
                                        'maxfun': 2000})

                if res.fun < best_fun:
                    best_fun = res.fun
                    best_params = res.x.copy()

                if best_fun < 1e-12:
                    break

            except Exception:
                pass

        if best_params is None:
            return None

        # If optimization converged well, try rounding to half-integers
        if best_fun < 1e-8:
            U_opt = best_params[:size_u].reshape(n, m, rank)
            V_opt = best_params[size_u:size_u + size_v].reshape(m, p, rank)
            W_opt = best_params[size_u + size_v:].reshape(p, n, rank)

            # Try direct rounding
            U_int = np.round(2 * U_opt) / 2.0
            V_int = np.round(2 * V_opt) / 2.0
            W_int = np.round(2 * W_opt) / 2.0

            if check_decomposition(U_int, V_int, W_int, rank):
                return {"U": U_int.reshape(n * m, rank).tolist(),
                        "V": V_int.reshape(m * p, rank).tolist(),
                        "W": W_int.reshape(p * n, rank).tolist()}

            # Try small perturbations around the rounded values
            for attempt in range(50):
                if time.time() > deadline:
                    break
                U_try = U_int + 0.5 * np.random.randint(-1, 2, size=U_int.shape) * 0.5
                V_try = V_int + 0.5 * np.random.randint(-1, 2, size=V_int.shape) * 0.5
                W_try = W_int + 0.5 * np.random.randint(-1, 2, size=W_int.shape) * 0.5
                if check_decomposition(U_try, V_try, W_try, rank):
                    return {"U": U_try.reshape(n * m, rank).tolist(),
                            "V": V_try.reshape(m * p, rank).tolist(),
                            "W": W_try.reshape(p * n, rank).tolist()}

        return None

    # Strategy: Try rank 16 first (theoretical minimum for 2x4x4 matrix mult)
    # If that fails, try 17, 18, ..., up to 27
    # Fallback: rank 28 Strassen

    for rank in range(16, 28):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        budget = min(remaining - 2.0, 25.0)
        if budget < 3.0:
            break

        result = try_rank(rank, time_budget=budget)
        if result is not None:
            return result

    # Fallback: Rank 28 Strassen decomposition
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen's 7-term decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_terms = [
        # (U coeffs, V coeffs, W coeffs)
        # Each is dict: local_index -> coefficient
        # Local indices 0,1,2,3 correspond to positions (0,0),(0,1),(1,0),(1,1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})         # M7
    ]

    def apply_strassen_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        """Apply Strassen to a 2x2x2 block.
        a_col_off: offset of A columns (0 or 1, meaning cols 0-1 or 2-3)
        b_row_off: offset of B rows (0 or 1, meaning rows 0-1 or 2-3)
        b_col_off: offset of B cols (0 or 1, meaning cols 0-1 or 2-3)
        c_col_off: offset of C cols (0 or 1, meaning cols 0-1 or 2-3)
        """
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_i = r  # row in A (0 or 1)
                g_j = a_col_off * 2 + c  # col in A
                U[g_i * m + g_j, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_j = b_row_off * 2 + r  # row in B
                g_k = b_col_off * 2 + c  # col in B
                V[g_j * p + g_k, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_k = c_col_off * 2 + c  # col in C
                g_i = r  # row in C
                W[g_k * n + g_i, col] += coeff

    # We need to cover all combinations:
    # A has 2 cols (split into 2 blocks of 1? No, A is 2x4, split into 2 blocks of 2 cols)
    # B has 4 rows (split into 2 blocks of 2 rows) and 4 cols (split into 2 blocks of 2 cols)
    # C has 4 cols (split into 2 blocks of 2 cols)
    #
    # The multiplication C = A @ B:
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    #
    # Each term is a 2x2 * 2x2 multiplication, which Strassen handles in 7 terms.
    # Total: 4 blocks * 7 = 28 terms.

    apply_strassen_block(0, 0, 0, 0, 0)   # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    apply_strassen_block(1, 1, 0, 0, 7)   # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    apply_strassen_block(0, 0, 1, 1, 14)  # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    apply_strassen_block(1, 1, 1, 1, 21)  # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf):
        print("Verification Failed")
        print("C:")
        print(C)
        print("Cf:")
        print(Cf)
    else:
        print("Verification Passed")

    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
