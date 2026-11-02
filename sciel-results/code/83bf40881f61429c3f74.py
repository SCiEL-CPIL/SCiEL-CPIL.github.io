import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def verify_decomposition_np(U, V, W):
        """Verify using numpy arrays."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def current_solution():
        """Rank 50 Strassen-based solution."""
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]

        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1

            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7

        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1

        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1

        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4

        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2

        return {"U": U, "V": V, "W": W}

    def compute_errors(U, V, W, r):
        """Compute all tensor entry errors."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    errors[i, j, k] = val - 1.0
        return errors

    def solve_given_W_V(W, V, r):
        """Solve for U given fixed W and V."""
        U = np.zeros((n_rows, r), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        return U

    def solve_given_W_U(W, U, r):
        """Solve for V given fixed W and U."""
        V = np.zeros((m_rows, r), dtype=np.float64)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        return V

    def polish_solution(U, V, W, r, max_iters=200, tol=1e-12):
        """High-precision polish: fix sparsity pattern, solve exact system via Newton."""
        if time.time() > deadline - 10:
            return None

        # Fix the sparsity pattern (which entries are nonzero)
        U_pattern = (np.abs(U) > 1e-6).astype(np.float64)
        V_pattern = (np.abs(V) > 1e-6).astype(np.float64)
        W_pattern = (np.abs(W) > 1e-6).astype(np.float64)

        # Collect all variable positions
        U_vars = []
        for i in range(n_rows):
            for c in range(r):
                if U_pattern[i, c] > 0:
                    U_vars.append((i, c))

        V_vars = []
        for i in range(m_rows):
            for c in range(r):
                if V_pattern[i, c] > 0:
                    V_vars.append((i, c))

        W_vars = []
        for i in range(p_rows):
            for c in range(r):
                if W_pattern[i, c] > 0:
                    W_vars.append((i, c))

        nU = len(U_vars)
        nV = len(V_vars)
        nW = len(W_vars)
        nvar = nU + nV + nW

        if nvar == 0:
            return None

        # Build variable index maps
        U_idx = {}
        for idx, (i, c) in enumerate(U_vars):
            U_idx[(i, c)] = idx
        V_idx = {}
        for idx, (i, c) in enumerate(V_vars):
            V_idx[(i, c)] = nU + idx
        W_idx = {}
        for idx, (i, c) in enumerate(W_vars):
            W_idx[(i, c)] = nU + nV + idx

        # Current values
        x0 = np.zeros(nvar)
        for i in range(n_rows):
            for c in range(r):
                if (i, c) in U_idx:
                    x0[U_idx[(i, c)]] = U[i, c]
        for i in range(m_rows):
            for c in range(r):
                if (i, c) in V_idx:
                    x0[V_idx[(i, c)]] = V[i, c]
        for i in range(p_rows):
            for c in range(r):
                if (i, c) in W_idx:
                    x0[W_idx[(i, c)]] = W[i, c]

        def residuals(x):
            """Compute residual vector: for each (i,j,k), sum_c U*V*W - 1"""
            U_val = np.zeros((n_rows, r))
            V_val = np.zeros((m_rows, r))
            W_val = np.zeros((p_rows, r))
            for (i, c), idx in U_idx.items():
                U_val[i, c] = x[idx]
            for (i, c), idx in V_idx.items():
                V_val[i, c] = x[idx]
            for (i, c), idx in W_idx.items():
                W_val[i, c] = x[idx]

            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_val[i*m+j] * V_val[j*p+k] * W_val[k*n+i])
                        res[idx] = val - 1.0
                        idx += 1
            return res

        def jacobian(x):
            """Compute Jacobian of residuals w.r.t. variables."""
            U_val = np.zeros((n_rows, r))
            V_val = np.zeros((m_rows, r))
            W_val = np.zeros((p_rows, r))
            for (i, c), idx in U_idx.items():
                U_val[i, c] = x[idx]
            for (i, c), idx in V_idx.items():
                V_val[i, c] = x[idx]
            for (i, c), idx in W_idx.items():
                W_val[i, c] = x[idx]

            J = np.zeros((n * m * p, nvar))
            res_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c]
                        for (u_i, u_c), u_idx in U_idx.items():
                            if u_i == i*m+j:
                                J[res_idx, u_idx] = V_val[j*p+k, u_c] * W_val[k*n+i, u_c]
                        # d/dV[j*p+k, c] = U[i*m+j, c] * W[k*n+i, c]
                        for (v_i, v_c), v_idx in V_idx.items():
                            if v_i == j*p+k:
                                J[res_idx, nU + v_idx] = U_val[i*m+j, v_c] * W_val[k*n+i, v_c]
                        # d/dW[k*n+i, c] = U[i*m+j, c] * V[j*p+k, c]
                        for (w_i, w_c), w_idx in W_idx.items():
                            if w_i == k*n+i:
                                J[res_idx, nU + nV + w_idx] = U_val[i*m+j, w_c] * V_val[j*p+k, w_c]
                        res_idx += 1
            return J

        # Newton's method
        x = x0.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            res = residuals(x)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break

            J = jacobian(x)
            try:
                dx, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            except:
                break

            # Damping if step is too large
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm

            x_new = x + dx
            res_new = residuals(x_new)
            if np.max(np.abs(res_new)) < max_res:
                x = x_new
            else:
                # Halve step
                x = x + dx / 2
                res_new = residuals(x)
                if np.max(np.abs(res_new)) < max_res:
                    x = x
                else:
                    break

        # Reconstruct U, V, W
        U_pol = np.zeros((n_rows, r))
        V_pol = np.zeros((m_rows, r))
        W_pol = np.zeros((p_rows, r))
        for (i, c), idx in U_idx.items():
            U_pol[i, c] = x[idx]
        for (i, c), idx in V_idx.items():
            V_pol[i, c] = x[idx]
        for (i, c), idx in W_idx.items():
            W_pol[i, c] = x[idx]

        # Round to nearest 0.5
        U_r = np.round(U_pol * 2) / 2
        V_r = np.round(V_pol * 2) / 2
        W_r = np.round(W_pol * 2) / 2

        if verify_decomposition_np(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        # Try rounding to nearest integer
        U_r2 = np.round(U_pol)
        V_r2 = np.round(V_pol)
        W_r2 = np.round(W_pol)
        if verify_decomposition_np(U_r2, V_r2, W_r2):
            return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}

        return None

    def try_rank_with_fixed_W(r, max_iters=100, max_trials=15):
        """Try to find rank-r decomposition by fixing W structure and alternating solves."""
        if time.time() > deadline - 20:
            return None

        rng = np.random.RandomState(42 + r)

        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break

            # Initialize W with sparse 0-1 pattern
            W = np.zeros((p_rows, r), dtype=np.float64)
            for c in range(r):
                num_nz = rng.randint(2, 5)
                positions = rng.choice(p_rows, size=num_nz, replace=False)
                for pos in positions:
                    W[pos, c] = rng.choice([-1.0, 1.0])

            # Initialize U and V randomly
            U = np.random.randn(n_rows, r) * 0.5
            V = np.random.randn(m_rows, r) * 0.5

            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                U = solve_given_W_V(W, V, r)
                V = solve_given_W_U(W, U, r)

                errors = compute_errors(U, V, W, r)
                error = np.max(np.abs(errors))
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if error < 1e-10:
                    break

            # Check if we found an integer/half-integer solution
            if best_error < 1e-10:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2

                if verify_decomposition_np(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

            # Polish pass
            if best_error < 0.1:
                polished = polish_solution(best_U, best_V, best_W, r)
                if polished:
                    return polished

        return None

    def try_structured_rank(r, max_time=10):
        """Try structured approach: use Strassen-like patterns for 2x3x3 blocks."""
        if time.time() > deadline - 15:
            return None

        # The 2x3x9 tensor can be decomposed into:
        # - 3 blocks of 2x3x3 (for k=0,1,2; k=3,4,5; k=6,7,8)
        # - 1 block of 2x3x1 (for k=8, if we use 3x3 blocks for first 9)
        # Actually: 9 = 3+3+3, so three 2x3x3 blocks
        # Each 2x3x3 has known rank 7 (Strassen for 2x2x2 applied with care)
        # But 2x3x3 is not standard Strassen... Let's use the known rank-10 for 2x3x3
        # or try to find better.

        # Alternative: use the fact that 2x3x9 can be seen as 2x3x(3+3+3)
        # and each 2x3x3 block can share structure.

        # For now, try the ALS approach with structured initialization
        rng = np.random.RandomState(123 + r)

        for trial in range(5):
            if time.time() > deadline - 5:
                break

            # Initialize with Strassen-like structure for first 28 cols (4 pairs)
            U = np.zeros((n_rows, r))
            V = np.zeros((m_rows, r))
            W = np.zeros((p_rows, r))

            col = 0
            for b in range(4):
                if col + 7 > r:
                    break
                k0 = 2 * b
                k1 = 2 * b + 1
                v_e = 0 * p + k0
                v_f = 0 * p + k1
                v_g = 1 * p + k0
                v_h = 1 * p + k1
                u_a = 0 * m + 0
                u_b = 0 * m + 1
                u_c = 1 * m + 0
                u_d = 1 * m + 1

                U[u_a, col] += 1; U[u_d, col] += 1; V[v_e, col] += 1; V[v_h, col] += 1
                U[u_c, col+1] += 1; U[u_d, col+1] += 1; V[v_e, col+1] += 1
                U[u_a, col+2] += 1; V[v_f, col+2] += 1; V[v_h, col+2] -= 1
                U[u_d, col+3] += 1; V[v_g, col+3] += 1; V[v_e, col+3] -= 1
                U[u_a, col+4] += 1; U[u_b, col+4] += 1; V[v_h, col+4] += 1
                U[u_c, col+5] += 1; U[u_a, col+5] -= 1; V[v_e, col+5] += 1; V[v_f, col+5] += 1
                U[u_b, col+6] += 1; U[u_d, col+6] -= 1; V[v_g, col+6] += 1; V[v_h, col+6] += 1

                w_00 = k0 * n + 0
                w_01 = k1 * n + 0
                w_10 = k0 * n + 1
                w_11 = k1 * n + 1

                W[w_00, col] += 1; W[w_00, col+3] += 1; W[w_00, col+4] -= 1; W[w_00, col+6] += 1
                W[w_01, col+2] += 1; W[w_01, col+4] += 1
                W[w_10, col+1] += 1; W[w_10, col+3] += 1
                W[w_11, col] += 1; W[w_11, col+1] -= 1; W[w_11, col+2] += 1; W[w_11, col+5] += 1
                col += 7

            # Fill remaining columns with random sparse structure
            for c in range(col, r):
                u_idx = rng.choice(n_rows, size=rng.randint(2, 4), replace=False)
                for idx in u_idx:
                    U[idx, c] = rng.choice([-1.0, 1.0])
                v_idx = rng.choice(m_rows, size=rng.randint(2, 4), replace=False)
                for idx in v_idx:
                    V[idx, c] = rng.choice([-1.0, 1.0])
                w_idx = rng.choice(p_rows, size=rng.randint(2, 4), replace=False)
                for idx in w_idx:
                    W[idx, c] = rng.choice([-1.0, 1.0])

            # ALS refinement
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            for it in range(50):
                if time.time() > deadline - 5:
                    break

                U = solve_given_W_V(W, V, r)
                V = solve_given_W_U(W, U, r)

                errors = compute_errors(U, V, W, r)
                error = np.max(np.abs(errors))
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if error < 1e-10:
                    break

            if best_error < 1e-10:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if verify_decomposition_np(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

            if best_error < 0.05:
                polished = polish_solution(best_U, best_V, best_W, r)
                if polished:
                    return polished

        return None

    # Try structured approach for ranks below 50
    for r in range(49, 43, -1):
        if time.time() > deadline - 25:
            break
        res = try_structured_rank(r)
        if res:
            return res

    # Try random approach for remaining ranks
    for r in range(49, 43, -1):
        if time.time() > deadline - 15:
            break
        res = try_rank_with_fixed_W(r, max_iters=30, max_trials=5)
        if res:
            return res

    # Fallback to rank 50
    return current_solution()
