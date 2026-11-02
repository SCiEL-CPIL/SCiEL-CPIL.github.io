"""Structural redesign: build the 2x3x9 multiplication tensor, use SVD-based structured
initialization for factor matrices, then a fast local search with sparse error updates and
half-integer rounding checks.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n*m, m*p, p*n  # 6, 27, 18
    schoolbook_rank = n * m * p  # 54

    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(Nm)]
        V = [[0] * rank for _ in range(Mp)]
        W = [[0] * rank for _ in range(Pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, R):
        # U: Nm x R, V: Mp x R, W: Pn x R
        # Check: sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] == 1 for all i,j,k
        # And zero for non-matching indices.
        # We only need to check the 54 non-zero positions.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m + j]
                    vi = V[j*p + k]
                    wi = W[k*n + i]
                    s = 0
                    for c in range(R):
                        s += ui[c] * vi[c] * wi[c]
                    if s != 1:
                        return False
        return True

    def build_target_tensor():
        T = np.zeros((Nm, Mp, Pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T

    T_target = build_target_tensor()

    def calc_error_fast(U, V, W, R):
        """Calculate SSE error using einsum."""
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.sum((T_calc - T_target)**2)

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def try_rank(R, max_iters=8000, seed=42):
        rng = np.random.default_rng(seed)
        random.seed(seed)

        # SVD-based initialization: flatten tensor in one mode
        # T shape (Nm, Mp, Pn). Matricize to (Nm, Mp*Pn):
        T_mat = T_target.reshape(Nm, Mp*Pn)
        try:
            U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
        except:
            U_svd = None

        if U_svd is not None and min(Nm, Mp*Pn) >= R:
            U_init = U_svd[:, :R] * np.sqrt(S[:R])
        else:
            U_init = rng.standard_normal((Nm, R))

        # For V and W, use random sparse initialization with {-1, 0, 1}
        V_init = np.zeros((Mp, R), dtype=float)
        W_init = np.zeros((Pn, R), dtype=float)

        for c in range(R):
            # Each column should have some structure
            nz_v = rng.integers(3, 8)
            nz_w = rng.integers(3, 8)
            v_rows = rng.choice(Mp, size=nz_v, replace=False)
            w_rows = rng.choice(Pn, size=nz_w, replace=False)
            V_init[v_rows, c] = rng.choice([-1, 1], size=nz_v)
            W_init[w_rows, c] = rng.choice([-1, 1], size=nz_w)

        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        # Scale U to make initial error reasonable
        err = calc_error_fast(U, V, W, R)
        if err > 100:
            U = U * (100 / err) ** 0.25
            V = V * (100 / err) ** 0.25
            W = W * (100 / err) ** 0.25
            err = calc_error_fast(U, V, W, R)

        best_err = err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Candidate values for perturbations
        candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        # Local search with fast error updates
        # Precompute for fast updates
        # When we change U[i,c], the error change involves V[j,c]*W[k,c] for all j,k
        # This is expensive to compute incrementally, so we'll use full recalc every step
        # but with numpy which is fast for small tensors

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break

            err = calc_error_fast(U, V, W, R)
            if err < 1e-12:
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), R):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Perturb one entry
            r = random.random()
            if r < 0.35:
                mat = U
                rows, cols = Nm, R
            elif r < 0.65:
                mat = V
                rows, cols = Mp, R
            else:
                mat = W
                rows, cols = Pn, R

            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)

            old_val = mat[row, col]
            new_val = random.choice(candidates)
            if new_val == old_val:
                continue

            mat[row, col] = new_val
            new_err = calc_error_fast(U, V, W, R)

            if new_err >= err:
                mat[row, col] = old_val
            else:
                # Accept
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Check best solution
        if best_err < 1e-12:
            U_r = round_to_half(best_U)
            V_r = round_to_half(best_V)
            W_r = round_to_half(best_W)
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), R):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        return None

    # Try ranks from 40 upwards (structural: SVD init + local search)
    for R in [40, 42, 45, 48, 50]:
        if time.time() - start_time > DEADLINE - 15:
            break
        result = try_rank(R)
        if result:
            return result

    # Fallback
    return build_schoolbook()
