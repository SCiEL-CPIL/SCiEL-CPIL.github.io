import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 115  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def validate(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
            return False
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return bool(np.max(np.abs(T - 1.0)) < tol)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen 2x2x2 coefficient tables
    U_coeffs = [
        {0: 1, 3: 1},   # M1: a, d
        {2: 1, 3: 1},   # M2: c, d
        {0: 1},         # M3: a
        {3: 1},         # M4: d
        {0: 1, 1: 1},   # M5: a, b
        {2: 1, 0: -1},  # M6: c, -a
        {1: 1, 3: -1},  # M7: b, -d
    ]
    V_coeffs = [
        {0: 1, 3: 1},   # M1: e, h
        {0: 1},         # M2: e
        {1: 1, 3: -1},  # M3: f, -h
        {2: 1, 0: -1},  # M4: g, -e
        {3: 1},         # M5: h
        {0: 1, 1: 1},   # M6: e, f
        {2: 1, 3: 1},   # M7: g, h
    ]
    W_coeffs = [
        {0: 1, 3: 1},   # M1: C00, C11
        {2: 1, 3: -1},  # M2: C10, -C11
        {1: 1, 3: 1},   # M3: C01, C11
        {0: 1, 2: 1},   # M4: C00, C10
        {0: -1, 1: 1},  # M5: -C00, C01
        {3: 1},         # M6: C11
        {0: 1},         # M7: C00
    ]

    def build_strassen_embedding():
        rank = 68
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for g in range(4):
            k0 = 2 * g
            k1 = 2 * g + 1
            for l in range(7):
                for li, coeff in U_coeffs[l].items():
                    i_global = li // 2
                    j_global = li % 2
                    U[i_global * m + j_global, col] += coeff
                for lj, coeff in V_coeffs[l].items():
                    j_global = lj // 2
                    k_global = k0 + (lj % 2)
                    V[j_global * p + k_global, col] += coeff
                for lc, coeff in W_coeffs[l].items():
                    i_global = lc // 2
                    k_global = k0 + (lc % 2)
                    W[k_global * n + i_global, col] += coeff
                col += 1
            for i in range(2):
                for k in (k0, k1):
                    U[i * m + 2, col] = 1.0
                    V[2 * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
            for j in range(2):
                for k in (k0, k1):
                    U[2 * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + 2, col] = 1.0
                    col += 1
            for k in (k0, k1):
                U[2 * m + 2, col] = 1.0
                V[2 * p + k, col] = 1.0
                W[k * n + 2, col] = 1.0
                col += 1
        assert col == rank, (col, rank)
        return {"U": U, "V": V, "W": W}, rank

    # Try to reduce rank by removing one column and adjusting others
    def try_reduce_rank(factors, target_rank, budget_s=5.0):
        U0 = np.array(factors["U"], dtype=float)
        V0 = np.array(factors["V"], dtype=float)
        W0 = np.array(factors["W"], dtype=float)
        r0 = U0.shape[1]
        t_end = time.time() + budget_s
        
        # Try removing each column and see if we can compensate
        for c_remove in range(r0):
            if time.time() > t_end:
                break
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            # Remove column c_remove
            U = np.delete(U, c_remove, axis=1)
            V = np.delete(V, c_remove, axis=1)
            W = np.delete(W, c_remove, axis=1)
            r = r0 - 1
            if r != target_rank:
                continue
            # Compute residual
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            residual = 1.0 - T
            # Check if residual is rank-1 (can be fixed by adding one rank-1 term)
            # Flatten residual and check if it's a rank-1 tensor
            res_flat = residual.flatten()
            # A rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k]
            # Check if residual can be written as such
            # Try: for each nonzero entry (i,j,k) in residual, set a[i]=res[i,j,k], b[j]=1, c[k]=1
            # and check consistency
            nonzero = np.argwhere(np.abs(residual) > 1e-10)
            if len(nonzero) == 0:
                return {"U": U, "V": V, "W": W}, r
            if len(nonzero) == 1:
                i, j, k = nonzero[0]
                val = residual[i, j, k]
                U_new = np.zeros((nm, r + 1))
                V_new = np.zeros((mp, r + 1))
                W_new = np.zeros((pn, r + 1))
                U_new[:, :r] = U
                V_new[:, :r] = V
                W_new[:, :r] = W
                U_new[i * m + j, r] = val
                V_new[j * p + k, r] = 1.0
                W_new[k * n + i, r] = 1.0
                if validate(U_new, V_new, W_new):
                    return {"U": U_new, "V": V_new, "W": W_new}, r + 1
            # For multiple nonzero entries, check rank-1 structure
            if len(nonzero) >= 2:
                i0, j0, k0_ = nonzero[0]
                val0 = residual[i0, j0, k0_]
                if abs(val0) < 1e-10:
                    continue
                # Assume a[i0]=val0, b[j0]=1, c[k0_]=1
                a = np.zeros(n)
                b = np.zeros(m)
                c = np.zeros(p)
                a[i0] = val0
                b[j0] = 1.0
                c[k0_] = 1.0
                valid = True
                for i, j, k in nonzero[1:]:
                    val = residual[i, j, k]
                    if abs(a[i]) > 1e-10:
                        if abs(a[i] * b[j] * c[k] - val) > 1e-8:
                            valid = False
                            break
                    elif abs(b[j]) > 1e-10:
                        if abs(a[i] * b[j] * c[k] - val) > 1e-8:
                            valid = False
                            break
                    elif abs(c[k]) > 1e-10:
                        if abs(a[i] * b[j] * c[k] - val) > 1e-8:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                if valid:
                    U_new = np.zeros((nm, r + 1))
                    V_new = np.zeros((mp, r + 1))
                    W_new = np.zeros((pn, r + 1))
                    U_new[:, :r] = U
                    V_new[:, :r] = V
                    W_new[:, :r] = W
                    for i in range(n):
                        if abs(a[i]) > 1e-10:
                            for j in range(m):
                                if abs(b[j]) > 1e-10:
                                    U_new[i * m + j, r] = a[i] * b[j]
                    for j in range(m):
                        if abs(b[j]) > 1e-10:
                            for k in range(p):
                                if abs(c[k]) > 1e-10:
                                    V_new[j * p + k, r] = b[j] * c[k]
                    for k in range(p):
                        if abs(c[k]) > 1e-10:
                            for i in range(n):
                                if abs(a[i]) > 1e-10:
                                    W_new[k * n + i, r] = c[k] * a[i]
                    if validate(U_new, V_new, W_new):
                        return {"U": U_new, "V": V_new, "W": W_new}, r + 1
        return None

    # Main
    factors, rank = build_strassen_embedding()
    if not validate(factors["U"], factors["V"], factors["W"]):
        return schoolbook()

    best = (factors["U"], factors["V"], factors["W"], rank)

    # Try to reduce by 1
    if time_left() > 15.0:
        result = try_reduce_rank(factors, rank - 1, budget_s=min(10.0, time_left() - 12.0))
        if result is not None:
            U_new, V_new, W_new, r_new = result["U"], result["V"], result["W"], result[1]
            if validate(U_new, V_new, W_new) and r_new < best[3]:
                best = (U_new, V_new, W_new, r_new)

    U, V, W, r = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
