import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-10):
    T = np.einsum('ar,br,cr->abc', U, V, W)
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False
    return True

def _get_rank27():
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    return U, V, W

def _build_target():
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
    return M

_TARGET = _build_target()

def _als_decompose(rank, max_iters=300, tol=1e-10, init_U=None, init_V=None, init_W=None,
                   round_to_half=True, use_polar=False):
    if init_U is not None:
        U = init_U.astype(float).copy()
        V = init_V.astype(float).copy()
        W = init_W.astype(float).copy()
    else:
        U = np.random.randn(9, rank)
        V = np.random.randn(9, rank)
        W = np.random.randn(9, rank)

    M = _TARGET
    prev_err = np.inf

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break

        # Update U
        B_mat = np.einsum('br,cr->bcr', V, W).reshape(81, rank)
        M_mat = M.reshape(9, 81)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat.T, M_mat.T, rcond=None)
            U = U_new.T
        except:
            pass

        # Update V
        M_mat2 = M.transpose(1, 0, 2).reshape(9, 81)
        B_mat2 = np.einsum('ar,cr->acr', U, W).reshape(81, rank)
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat2.T, M_mat2.T, rcond=None)
            V = V_new.T
        except:
            pass

        # Update W
        M_mat3 = M.transpose(2, 0, 1).reshape(9, 81)
        B_mat3 = np.einsum('ar,br->abr', U, V).reshape(81, rank)
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat3.T, M_mat3.T, rcond=None)
            W = W_new.T
        except:
            pass

        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - M)
        if err < tol:
            break
        if err > prev_err * 1.01:
            # Diverging, stop early
            break
        prev_err = err

    if round_to_half:
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        # Try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if _verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
    return None

def _try_rank(rank, n_attempts=5, max_iters=200):
    U27, V27, W27 = _get_rank27()
    for attempt in range(n_attempts):
        if time.time() > _DEADLINE - 10:
            break
        # Initialize by selecting random subset of rank-27 columns
        if rank < 27:
            cols = np.random.choice(27, rank, replace=False)
            U_init = U27[:, cols]
            V_init = V27[:, cols]
            W_init = W27[:, cols]
            res = _als_decompose(rank, max_iters=max_iters, init_U=U_init, init_V=V_init, init_W=W_init)
        else:
            res = _als_decompose(rank, max_iters=max_iters)
        if res:
            return res
    return None

def _try_random_init(rank, n_attempts=5, max_iters=200):
    for attempt in range(n_attempts):
        if time.time() > _DEADLINE - 10:
            break
        res = _als_decompose(rank, max_iters=max_iters)
        if res:
            return res
    return None

def construct():
    # Try rank 23 first (optimal)
    for r in [23, 24, 25, 26]:
        if time.time() > _DEADLINE - 15:
            break
        res = _try_rank(r, n_attempts=8, max_iters=250)
        if res:
            return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
        res = _try_random_init(r, n_attempts=3, max_iters=200)
        if res:
            return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}

    # Fallback: rank 27
    U27, V27, W27 = _get_rank27()
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
