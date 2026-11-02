import time
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = 2, 3, 8

def _verify(U, V, W, tol=1e-9):
    """Verify U, V, W (lists of lists) exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                e = abs(T[a, b, c] - 1.0)
                if e > max_err:
                    max_err = e
    # Check non-target entries are zero
    target_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_mask[i * m + j, j * p + k, k * n + i] = True
    non_target_err = float(np.max(np.abs(T[~target_mask]))) if (~target_mask).any() else 0.0
    return max_err <= tol and non_target_err <= tol

def _strassen_block(block):
    """Return (U,V,W) local factor matrices (6x7, 24x7, 16x7) for a 2x2*2x2
    Strassen block acting on B columns (k0,k1), A cols 0,1."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 7))
    V = np.zeros((24, 7))
    W = np.zeros((16, 7))
    U[0, 0] += 1; U[4, 0] += 1
    V[k0, 0] += 1; V[8 + k1, 0] += 1
    W[2 * k0, 0] += 1; W[2 * k1 + 1, 0] += 1
    U[3, 1] += 1; U[4, 1] += 1
    V[k0, 1] += 1
    W[2 * k0 + 1, 1] += 1; W[2 * k1 + 1, 1] += -1
    U[0, 2] += 1
    V[k1, 2] += 1; V[8 + k1, 2] += -1
    W[2 * k1, 2] += 1; W[2 * k1 + 1, 2] += 1
    U[4, 3] += 1
    V[8 + k0, 3] += 1; V[k0, 3] += -1
    W[2 * k0, 3] += 1; W[2 * k0 + 1, 3] += 1
    U[0, 4] += 1; U[1, 4] += 1
    V[8 + k1, 4] += 1
    W[2 * k0, 4] += -1; W[2 * k1, 4] += 1
    U[3, 5] += 1; U[0, 5] += -1
    V[k0, 5] += 1; V[k1, 5] += 1
    W[2 * k1 + 1, 5] += 1
    U[1, 6] += 1; U[4, 6] += -1
    V[8 + k0, 6] += 1; V[8 + k1, 6] += 1
    W[2 * k0, 6] += 1
    return U, V, W

def _outer_block(block):
    """4 rank-1 terms for A_right (col 2) * B_bottom (row 2)."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 4))
    V = np.zeros((24, 4))
    W = np.zeros((16, 4))
    t = 0
    for ii in range(2):
        for kk in range(2):
            kv = k0 if kk == 0 else k1
            U[ii * 3 + 2, t] = 1
            V[16 + kv, t] = 1
            W[kv * 2 + ii, t] = 1
            t += 1
    return U, V, W

def _build_rank44_decomposition():
    """Explicit rank-44 decomposition: 4 blocks x (Strassen 7 + outer 4)."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))
    col = 0
    for block in range(4):
        Us, Vs, Ws = _strassen_block(block)
        Uo, Vo, Wo = _outer_block(block)
        U[:, col:col + 7] = Us
        V[:, col:col + 7] = Vs
        W[:, col:col + 7] = Ws
        col += 7
        U[:, col:col + 4] = Uo
        V[:, col:col + 4] = Vo
        W[:, col:col + 4] = Wo
        col += 4
    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()

def _build_target_tensor():
    """Build the target tensor T[a,b,c] = 1 if (a,b,c) corresponds to (i,j,k)
    with a=i*m+j, b=j*p+k, c=k*n+i, else 0."""
    n, m, p = N, M, P
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_norm_err(U, V, W, T):
    """Compute max absolute error between tensor decomposition and target."""
    n, m, p = N, M, P
    rank = U.shape[1]
    T_approx = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return float(np.max(np.abs(T_approx - T)))

def _als_decompose(U0, V0, W0, T, max_iter=200, tol=1e-12, deadline=None):
    """ALS optimization. Returns (U, V, W, err)."""
    n, m, p = N, M, P
    U = U0.copy().astype(float)
    V = V0.copy().astype(float)
    W = W0.copy().astype(float)
    rank = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Update U
        for a in range(n * m):
            design = np.zeros((m * p * p * n, rank))
            for r in range(rank):
                design[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            target_vec = T[a].flatten()
            try:
                sol, _, _, _ = np.linalg.lstsq(design, target_vec, rcond=None)
                U[a, :] = sol
            except np.linalg.LinAlgError:
                pass
        
        # Update V
        for b in range(m * p):
            design = np.zeros((n * m * p * n, rank))
            for r in range(rank):
                design[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            target_vec = T[:, b, :].flatten()
            try:
                sol, _, _, _ = np.linalg.lstsq(design, target_vec, rcond=None)
                V[b, :] = sol
            except np.linalg.LinAlgError:
                pass
        
        # Update W
        for c in range(p * n):
            design = np.zeros((n * m * m * p, rank))
            for r in range(rank):
                design[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            target_vec = T[:, :, c].flatten()
            try:
                sol, _, _, _ = np.linalg.lstsq(design, target_vec, rcond=None)
                W[c, :] = sol
            except np.linalg.LinAlgError:
                pass
        
        err = _tensor_norm_err(U, V, W, T)
        if err < tol:
            break
    
    return U, V, W, err

def _round_and_verify(U, V, W, T, tol=1e-9):
    """Round entries to nearest half-integer and verify."""
    U_r = np.round(2 * U) / 2.0
    V_r = np.round(2 * V) / 2.0
    W_r = np.round(2 * W) / 2.0
    err = _tensor_norm_err(U_r, V_r, W_r, T)
    if err <= tol:
        return U_r, V_r, W_r, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    U44, V44, W44 = _build_rank44_decomposition()
    U44_arr = np.array(U44, dtype=float)
    V44_arr = np.array(V44, dtype=float)
    W44_arr = np.array(W44, dtype=float)

    T = _build_target_tensor()

    best_U, best_V, best_W = U44_arr, V44_arr, W44_arr
    best_rank = 44

    for target_rank in range(43, 20, -1):
        if time.time() > deadline:
            break

        for trial in range(3):
            if time.time() > deadline:
                break

            rng = np.random.RandomState(42 + trial * 1000 + target_rank)
            U_init = rng.randn(n * m, target_rank) * 0.5
            V_init = rng.randn(m * p, target_rank) * 0.5
            W_init = rng.randn(p * n, target_rank) * 0.5

            U_out, V_out, W_out, err = _als_decompose(
                U_init, V_init, W_init, T,
                max_iter=100, tol=1e-12, deadline=deadline
            )

            if time.time() > deadline:
                break

            U_r, V_r, W_r, ok = _round_and_verify(U_out, V_out, W_out, T, tol=1e-9)
            if ok:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break

        if target_rank < 44 and time.time() < deadline:
            U_init = U44_arr[:, :target_rank].copy()
            V_init = V44_arr[:, :target_rank].copy()
            W_init = W44_arr[:, :target_rank].copy()

            U_out, V_out, W_out, err = _als_decompose(
                U_init, V_init, W_init, T,
                max_iter=100, tol=1e-12, deadline=deadline
            )

            if time.time() > deadline:
                break

            U_r, V_r, W_r, ok = _round_and_verify(U_out, V_out, W_out, T, tol=1e-9)
            if ok:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break

    if best_rank == 44:
        return {"U": U44, "V": V44, "W": W44}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
