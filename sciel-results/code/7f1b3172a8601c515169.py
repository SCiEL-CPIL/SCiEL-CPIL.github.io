import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                continue
                            val = np.sum(U[i1*m+j1, :] * V[j2*p+k1, :] * W[k2*n+i2, :])
                            if abs(val) > tol:
                                return False
    return True

def _precompute_indices(n, m, p):
    """Precompute valid and invalid index triples."""
    valid = []
    invalid = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.append((i*m+j, j*p+k, k*n+i))
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                continue
                            invalid.append((i1*m+j1, j2*p+k1, k2*n+i2))
    return np.array(valid), np.array(invalid)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1.0
                V_school[j * p + k, c] = 1.0
                W_school[k * n + i, c] = 1.0
                c += 1

    valid_idx, invalid_idx = _precompute_indices(n, m, p)

    for target_rank in [39, 40, 41]:
        if time.time() > deadline:
            break

        for trial in range(3):
            if time.time() > deadline:
                break

            np.random.seed(trial * 104729 + 7)
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3

            best_obj = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            lr = 0.003
            for step in range(1500):
                if time.time() > deadline:
                    break

                va = valid_idx[:, 0]
                vb = valid_idx[:, 1]
                vc = valid_idx[:, 2]
                prod_valid = U[va] * V[vb] * W[vc]
                valid_vals = prod_valid.sum(axis=1)
                valid_err = valid_vals - 1.0

                ia = invalid_idx[:, 0]
                ib = invalid_idx[:, 1]
                ic = invalid_idx[:, 2]
                prod_invalid = U[ia] * V[ib] * W[ic]
                invalid_vals = prod_invalid.sum(axis=1)

                obj = np.sum(valid_err ** 2) + np.sum(invalid_vals ** 2)

                if obj < best_obj:
                    best_obj = obj
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if best_obj < 1e-14:
                    break

                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)

                valid_contrib_U = 2 * valid_err[:, None] * V[vb] * W[vc]
                valid_contrib_V = 2 * valid_err[:, None] * U[va] * W[vc]
                valid_contrib_W = 2 * valid_err[:, None] * U[va] * V[vb]

                invalid_contrib_U = 2 * invalid_vals[:, None] * V[ib] * W[ic]
                invalid_contrib_V = 2 * invalid_vals[:, None] * U[ia] * W[ic]
                invalid_contrib_W = 2 * invalid_vals[:, None] * U[ia] * V[ib]

                np.add.at(grad_U, va, valid_contrib_U)
                np.add.at(grad_V, vb, valid_contrib_V)
                np.add.at(grad_W, vc, valid_contrib_W)

                np.add.at(grad_U, ia, invalid_contrib_U)
                np.add.at(grad_V, ib, invalid_contrib_V)
                np.add.at(grad_W, ic, invalid_contrib_W)

                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

            if best_obj < 1e-10:
                if _check_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
