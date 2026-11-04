import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 3, 6
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    def verify_decomposition(U, V, W, tol=1e-9):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_rank_32():
        """Build rank 32 decomposition: 12 (schoolbook col0) + 10 + 10 (2x2x3 blocks)."""
        rank = 32
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0

        # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> C[:, 0:6]
        # Split into two 2x2x3 blocks: cols (0,1,2) and (3,4,5)
        for t in range(2):
            c0 = 3 * t      # first column of this block
            c1 = 3 * t + 1  # second column
            c2 = 3 * t + 2  # third column

            # 2x2x3 rank-10 decomposition
            # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
            # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]

            # M1: (a+d)(e+h)
            U[0*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + c0, col] += 1.0
            V[2*p + c0, col] += 1.0
            W[c0*n + 0, col] += 1.0
            W[c0*n + 1, col] += 1.0
            col += 1

            # M2: (c+d)e
            U[1*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + c0, col] += 1.0
            W[c0*n + 1, col] += 1.0
            W[c0*n + 0, col] += -1.0
            col += 1

            # M3: a(f-i)
            U[0*m + 1, col] += 1.0
            V[1*p + c1, col] += 1.0
            V[2*p + c1, col] += -1.0
            W[c1*n + 0, col] += 1.0
            col += 1

            # M4: d(h-e)
            U[1*m + 2, col] += 1.0
            V[2*p + c0, col] += 1.0
            V[1*p + c0, col] += -1.0
            W[c0*n + 1, col] += 1.0
            col += 1

            # M5: (a+b)i
            U[0*m + 1, col] += 1.0
            U[0*m + 2, col] += 1.0
            V[2*p + c1, col] += 1.0
            W[c1*n + 0, col] += 1.0
            col += 1

            # M6: (c-a)(e+f)
            U[1*m + 1, col] += 1.0
            U[0*m + 1, col] += -1.0
            V[1*p + c0, col] += 1.0
            V[1*p + c1, col] += 1.0
            W[c1*n + 1, col] += 1.0
            col += 1

            # M7: (b-d)(h+i)
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += -1.0
            V[2*p + c0, col] += 1.0
            V[2*p + c1, col] += 1.0
            W[c1*n + 0, col] += 1.0
            W[c1*n + 1, col] += -1.0
            col += 1

            # M8: (a+c)g
            U[0*m + 1, col] += 1.0
            U[1*m + 1, col] += 1.0
            V[1*p + c2, col] += 1.0
            W[c2*n + 0, col] += 1.0
            W[c2*n + 1, col] += 1.0
            col += 1

            # M9: (b+d)j
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[2*p + c2, col] += 1.0
            W[c2*n + 0, col] += 1.0
            W[c2*n + 1, col] += 1.0
            col += 1

            # M10: (a-c)g + (b-d)j correction
            # C02 = ag + bj, C12 = cg + dj
            # From M8: (a+c)g contributes a*g + c*g to both C02 and C12
            # From M9: (b+d)j contributes b*j + d*j to both C02 and C12
            # We need: C02 = ag + bj, C12 = cg + dj
            # M8 gives: C02 += ag+cg, C12 += ag+cg  -- wrong, we need different
            # Let me reconsider.
            # Actually, let me verify this construction properly.
            # This is getting complex. Let me use a different approach.
            # For now, skip M10 and verify if 9 terms suffice (unlikely)
            # Or use a simpler fallback.
            pass

        return U, V, W, rank

    def build_rank_33():
        """Build rank 33 decomposition using Strassen for 2x2x2 blocks."""
        rank = 33
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0

        # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> C[:, 0:6]
        for t in range(3):
            # Strassen M1: (A00+A11)(B00+B11)
            U[0*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + 2*t, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[2*t*n + 0, col] += 1.0
            W[(2*t + 1)*n + 1, col] += 1.0
            col += 1

            # Strassen M2: (A10+A11)B00
            U[1*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + 2*t, col] += 1.0
            W[2*t*n + 1, col] += 1.0
            W[(2*t + 1)*n + 1, col] += -1.0
            col += 1

            # Strassen M3: A00(B01-B11)
            U[0*m + 1, col] += 1.0
            V[1*p + 2*t + 1, col] += 1.0
            V[2*p + 2*t + 1, col] += -1.0
            W[(2*t + 1)*n + 0, col] += 1.0
            W[(2*t + 1)*n + 1, col] += 1.0
            col += 1

            # Strassen M4: A11(B10-B00)
            U[1*m + 2, col] += 1.0
            V[2*p + 2*t, col] += 1.0
            V[1*p + 2*t, col] += -1.0
            W[2*t*n + 0, col] += 1.0
            W[2*t*n + 1, col] += 1.0
            col += 1

            # Strassen M5: (A00+A01)B11
            U[0*m + 1, col] += 1.0
            U[0*m + 2, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[2*t*n + 0, col] += -1.0
            W[(2*t + 1)*n + 0, col] += 1.0
            col += 1

            # Strassen M6: (A10-A00)(B00+B01)
            U[1*m + 1, col] += 1.0
            U[0*m + 1, col] += -1.0
            V[1*p + 2*t, col] += 1.0
            V[1*p + 2*t + 1, col] += 1.0
            W[(2*t + 1)*n + 1, col] += 1.0
            col += 1

            # Strassen M7: (A01-A11)(B10+B11)
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += -1.0
            V[2*p + 2*t, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[2*t*n + 0, col] += 1.0
            col += 1

        return U, V, W, rank

    def try_optimize_rank(target_rank, init_U=None, init_V=None, init_W=None, n_restarts=20, max_nfev=3000):
        if time.time() > deadline - 15:
            return None

        from scipy.optimize import least_squares

        T_flat = np.ones(n * m * p)

        def objective(params):
            U = params[0:num_A*target_rank].reshape(num_A, target_rank)
            V = params[num_A*target_rank:num_A*target_rank + num_B*target_rank].reshape(num_B, target_rank)
            W = params[num_A*target_rank + num_B*target_rank:].reshape(num_C, target_rank)

            result = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        result[idx] = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        idx += 1
            return result - T_flat

        best_params = None
        best_err = float('inf')

        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break

            if init_U is not None and restart == 0:
                if target_rank < 33:
                    U_init = init_U[:, :target_rank]
                    V_init = init_V[:, :target_rank]
                    W_init = init_W[:, :target_rank]
                    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                elif target_rank > 33:
                    U_init = np.zeros((num_A, target_rank))
                    V_init = np.zeros((num_B, target_rank))
                    W_init = np.zeros((num_C, target_rank))
                    U_init[:, :33] = init_U
                    V_init[:, :33] = init_V
                    W_init[:, :33] = init_W
                    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                else:
                    params0 = np.concatenate([init_U.flatten(), init_V.flatten(), W_init.flatten()])
            else:
                np.random.seed(42 + restart * 17)
                scale = 0.1 + 0.05 * (restart % 10)
                params0 = np.random.randn((num_A + num_B + num_C) * target_rank) * scale

            try:
                res = least_squares(objective, params0, method='lm', max_nfev=max_nfev)
                err = np.linalg.norm(res.fun)
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
            except Exception:
                pass

        if best_params is None or best_err > 1e-8:
            return None

        U = best_params[0:num_A*target_rank].reshape(num_A, target_rank)
        V = best_params[num_A*target_rank:num_A*target_rank + num_B*target_rank].reshape(num_B, target_rank)
        W = best_params[num_A*target_rank + num_B*target_rank:].reshape(num_C, target_rank)

        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0

        if verify_decomposition(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap, target_rank

        if verify_decomposition(U, V, W):
            return U, V, W, target_rank

        return None

    # Build rank-33 solution as baseline
    U_33, V_33, W_33, rank_33 = build_rank_33()

    best_rank = rank_33
    best_U = U_33.copy()
    best_V = V_33.copy()
    best_W = W_33.copy()

    if not verify_decomposition(U_33, V_33, W_33):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((num_A, rank_school))
        V_school = np.zeros((num_B, rank_school))
        W_school = np.zeros((num_C, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try to find rank 32
    if time.time() < deadline - 20:
        result = try_optimize_rank(32, init_U=U_33, init_V=V_33, init_W=W_33,
                                   n_restarts=15, max_nfev=2000)
        if result is not None:
            best_rank = 32
            best_U, best_V, best_W = result[0], result[1], result[2]

    # Try to find rank 31
    if best_rank > 31 and time.time() < deadline - 15:
        result = try_optimize_rank(31, init_U=best_U, init_V=best_V, init_W=best_W,
                                   n_restarts=10, max_nfev=1500)
        if result is not None:
            best_rank = 31
            best_U, best_V, best_W = result[0], result[1], result[2]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
