"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _round_halfint(X):
    # round to nearest half-integer
    return (2 * X).round() / 2.0

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    # Reconstruct tensor T of shape (n*m, m*p, p*n)
    rank = U.shape[1]
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build tensor T
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    
    # Build expected tensor of shape (n*m, m*p, p*n)
    expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                expected[u, v, w] = 1.0

    if np.max(np.abs(T - expected)) > tol:
        return False
    return True

def _als_decompose(n, m, p, rank, rng, max_iter=50, tol=1e-10):
    # Alternating least squares for rank-r tensor decomposition
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-15
    V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-15
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-15

    # Precompute valid indices for error checking
    valid_u = []
    valid_v = []
    valid_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_u.append(i * m + j)
                valid_v.append(j * p + k)
                valid_w.append(k * n + i)
    valid_u = np.array(valid_u)
    valid_v = np.array(valid_v)
    valid_w = np.array(valid_w)

    for it in range(max_iter):
        # Update U
        U_new = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, rank))
                for k in range(p):
                    B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        U = U_new

        # Update V
        V_new = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        V = V_new

        # Update W
        W_new = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                B = np.zeros((m, rank))
                for j in range(m):
                    B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        W = W_new

        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_valid = T[valid_u, valid_v, valid_w]
        err = np.max(np.abs(T_valid - 1.0))
        if err < tol:
            break

    return U, V, W, err

def _try_round_and_check(U, V, W, n, m, p):
    # Round to half-integers and check exactness
    Ur = _round_halfint(U)
    Vr = _round_halfint(V)
    Wr = _round_halfint(W)
    if _check_exact(Ur, Vr, Wr, n, m, p, tol=1e-9):
        return True, Ur, Vr, Wr
    # Also try rounding to integers
    Ui = U.round()
    Vi = V.round()
    Wi = W.round()
    if _check_exact(Ui, Vi, Wi, n, m, p, tol=1e-9):
        return True, Ui, Vi, Wi
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s kill)
    n, m, p = DIMENSIONS

    best = _schoolbook()
    best_rank = n * m * p

    rng = np.random.default_rng(42)

    # Try ranks from a known lower bound upward
    # Known lower bound for 5x5x5 is 55, but let's start from a reasonable range
    # Try a few ranks below schoolbook
    for rank in [100, 95, 90, 85, 80, 75, 70, 65, 60, 55]:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        # Multiple restarts per rank
        for restart in range(20):
            if time.time() > deadline:
                break
            U, V, W, err = _als_decompose(n, m, p, rank, rng, max_iter=30, tol=1e-12)
            if err < 1e-8:
                ok, Ur, Vr, Wr = _try_round_and_check(U, V, W, n, m, p)
                if ok:
                    # Convert to list of lists
                    U_list = [[float(x) for x in row] for row in Ur]
                    V_list = [[float(x) for x in row] for row in Vr]
                    W_list = [[float(x) for x in row] for row in Wr]
                    # Clean up: make half-integers exact
                    U_list = [[(int(2 * x) / 2.0) for x in row] for row in U_list]
                    V_list = [[(int(2 * x) / 2.0) for x in row] for row in V_list]
                    W_list = [[(int(2 * x) / 2.0) for x in row] for row in W_list]
                    return {"U": U_list, "V": V_list, "W": W_list}
            if time.time() > deadline:
                break

    return best
