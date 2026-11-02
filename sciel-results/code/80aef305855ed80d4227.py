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

    def to_lists(U, V, W, r):
        """Convert numpy arrays to list-of-lists format."""
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # t=0: M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0

        # t=1: M2 = (A10+A11)*B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0

        # t=2: M3 = A00*(B01-B11)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0

        # t=3: M4 = A11*(B10-B00)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0

        # t=4: M5 = (A00+A01)*B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0

        # t=5: M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0

        # t=6: M7 = (A01-A11)*(B10+B11)
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

    # Verify rank 28
    if not verify(U_full, V_full, W_full, tol=1e-8):
        # Fallback schoolbook
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
        return to_lists(U, V, W, rank)

    # Try to reduce rank using optimization
    best_rank = full_rank
    best_result = to_lists(U_full, V_full, W_full, full_rank)

    if time.time() < deadline - 30:
        try:
            from scipy.optimize import minimize

            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def unflatten_params(params, r):
                size_u = n * m * r
                size_v = m * p * r
                size_w = p * n * r
                U = params[:size_u].reshape(n * m, r)
                V = params[size_u:size_u + size_v].reshape(m * p, r)
                W = params[size_u + size_v:].reshape(p * n, r)
                return U, V, W

            def objective(params, r):
                U, V, W = unflatten_params(params, r)
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                return np.sum((recon - T)**2)

            # Try rank 27 with multiple initializations
            for target_rank in [27, 26]:
                if time.time() >= deadline - 15:
                    break

                # Multiple random restarts
                for restart in range(5):
                    if time.time() >= deadline - 10:
                        break

                    if restart == 0:
                        # Start from rank-28 solution, drop one column
                        U_init = U_full[:, :target_rank].copy()
                        V_init = V_full[:, :target_rank].copy()
                        W_init = W_full[:, :target_rank].copy()
                    else:
                        # Random initialization with small perturbations
                        scale = 1.0
                        U_init = np.random.randn(n * m, target_rank) * scale
                        V_init = np.random.randn(m * p, target_rank) * scale
                        W_init = np.random.randn(p * n, target_rank) * scale

                    init_params = flatten_params(U_init, V_init, W_init)

                    try:
                        res = minimize(
                            objective,
                            init_params,
                            args=(target_rank,),
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
                        )

                        if res.success or res.fun < 1e-8:
                            U_opt, V_opt, W_opt = unflatten_params(res.x, target_rank)
                            if verify(U_opt, V_opt, W_opt, tol=1e-6):
                                # Polish with tighter tolerance
                                if time.time() < deadline - 5:
                                    res2 = minimize(
                                        objective,
                                        res.x,
                                        args=(target_rank,),
                                        method='L-BFGS-B',
                                        options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 5000}
                                    )
                                    U_pol, V_pol, W_pol = unflatten_params(res2.x, target_rank)
                                    if verify(U_pol, V_pol, W_pol, tol=1e-8):
                                        U_opt, V_opt, W_opt = U_pol, V_pol, W_pol

                                best_rank = target_rank
                                best_result = to_lists(U_opt, V_opt, W_opt, target_rank)
                                break  # Found solution for this rank, try lower

                    except Exception:
                        pass

        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
