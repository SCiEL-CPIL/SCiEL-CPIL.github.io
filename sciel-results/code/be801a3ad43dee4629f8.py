"""Matrix multiplication 4,4,5 — half-integer rank search via iterative rank-1 peeling."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
RANK = n * m * p          # 80
M1 = n * m                # 16
M2 = m * p                # 20
M3 = p * n                # 20
DEADLINE = 140.0


def _tensor():
    T = np.zeros((M1, M2, M3), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _verify(U, V, W):
    """U,V,W have shape (M1,rank),(M2,rank),(M3,rank) with half-integer values.
    Check sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    T = _tensor()
    recon = np.zeros((M1, M2, M3), dtype=np.int64)
    rank = U.shape[1]
    for c in range(rank):
        u = U[:, c].astype(np.float64)
        v = V[:, c].astype(np.float64)
        w = W[:, c].astype(np.float64)
        recon += np.einsum('a,b,c->abc', u, v, w).astype(np.int64)
    return np.array_equal(recon, T)


def _half_round(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0


def _search_peeling(T, time_budget=120.0):
    """
    Search for low-rank half-integer decomposition.
    Returns (U, V, W) as lists of lists with half-integer values, or None.
    """
    start = time.time()
    best_rank = RANK
    best_result = None

    # Schoolbook baseline
    U0 = [[0] * RANK for _ in range(M1)]
    V0 = [[0] * RANK for _ in range(M2)]
    W0 = [[0] * RANK for _ in range(M3)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i][col] = 1
                V0[j][col] = 1
                W0[k][col] = 1
                col += 1
    best_result = (U0, V0, W0)

    # Try peeling with different seeds
    for seed in range(100):
        if time.time() - start > time_budget:
            break

        T_work = T.copy().astype(np.float64)
        U_list, V_list, W_list = [], [], []
        success = True

        for it in range(150):
            if np.all(np.abs(T_work) < 1e-9):
                break

            F = T_work.reshape(M1, M2 * M3)
            try:
                U_s, s, Vt = np.linalg.svd(F, full_matrices=False)
            except np.linalg.LinAlgError:
                success = False
                break

            u = U_s[:, 0]

            found = False
            # Try different scalings
            for t in np.linspace(0.5, 6.0, 120):
                u_try = u * t
                u_h = np.array([_half_round(x) for x in u_try])
                if np.all(u_h == 0):
                    continue

                # Compute rank-1 component
                proj = F @ u_h
                denom = float(u_h @ u_h)
                if denom < 1e-9:
                    continue
                comp_flat = np.outer(u_h, proj) / denom
                comp_3d = comp_flat.reshape(M1, M2, M3)

                # Round component to half-integers
                comp_rounded = np.array([[_half_round(x) for x in row] for row in comp_3d])

                # Check if residual is still "nice"
                new_residual = T_work - comp_rounded
                if np.linalg.norm(new_residual) < np.linalg.norm(T_work) - 1e-9:
                    T_work = new_residual
                    U_list.append(u_h.tolist())

                    # Extract v and w
                    # comp[i,j,k] = u_h[i] * v[j] * w[k]
                    # Find nonzero i
                    nonzero_i = np.where(u_h != 0)[0]
                    extracted = False
                    for i0 in nonzero_i:
                        u_val = u_h[i0]
                        if abs(u_val) < 1e-9:
                            continue
                        vw = comp_rounded[i0, :, :] / u_val
                        # Factor vw as v[j]*w[k]
                        # Try each k to get w[k] = vw[j,k]/v[j]
                        # This is a rank-1 matrix factorization
                        # Use SVD on vw
                        try:
                            U_vw, s_vw, Vt_vw = np.linalg.svd(vw, full_matrices=False)
                        except np.linalg.LinAlgError:
                            continue
                        if s_vw[0] < 1e-9:
                            continue
                        v_try = U_vw[:, 0] * np.sqrt(s_vw[0])
                        w_try = Vt_vw[0, :] * np.sqrt(s_vw[0])

                        # Round to half-integers
                        v_h = np.array([_half_round(x) for x in v_try])
                        w_h = np.array([_half_round(x) for x in w_try])

                        # Verify: v_h outer w_h should equal vw (approximately)
                        recon_vw = np.outer(v_h, w_h)
                        if np.allclose(recon_vw, vw, atol=0.01):
                            V_list.append(v_h.tolist())
                            W_list.append(w_h.tolist())
                            extracted = True
                            break

                    if not extracted:
                        success = False
                        break
                    found = True
                    break

            if not found:
                success = False
                break

            if time.time() - start > time_budget:
                success = False
                break

        if success and len(U_list) < best_rank:
            rank = len(U_list)
            U = [[0] * rank for _ in range(M1)]
            V = [[0] * rank for _ in range(M2)]
            W = [[0] * rank for _ in range(M3)]
            for c in range(rank):
                for i in range(M1):
                    U[i][c] = U_list[c][i]
                for j in range(M2):
                    V[j][c] = V_list[c][j]
                for k in range(M3):
                    W[k][c] = W_list[c][k]

            if _verify(U, V, W):
                best_rank = rank
                best_result = (U, V, W)

    return best_result


def construct():
    start = time.time()
    T = _tensor()

    try:
        result = _search_peeling(T, time_budget=DEADLINE - (time.time() - start) - 5.0)
    except Exception:
        result = None

    if result is None:
        # Schoolbook
        U = [[0] * RANK for _ in range(M1)]
        V = [[0] * RANK for _ in range(M2)]
        W = [[0] * RANK for _ in range(M3)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i][col] = 1
                    V[j][col] = 1
                    W[k][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    U, V, W = result
    return {"U": U, "V": V, "W": W}
