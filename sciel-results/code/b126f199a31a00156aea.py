import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135  # safety margin before 150s kill


def _build_rank_33(n, m, p):
    """
    Build a rank-33 decomposition: three disjoint 2×2×3 sub-tensors,
    each using rank-11 (Strassen-7 + schoolbook-4 for the 3rd column).
    """
    rank = 33
    U = [[0] * rank for _ in range(n * m)]      # 4 x 33
    V = [[0] * rank for _ in range(m * p)]      # 18 x 33
    W = [[0] * rank for _ in range(p * n)]      # 18 x 33

    col = 0

    def add_2x2x3_block(k0, k1, k2):
        nonlocal col
        # Strassen for columns (k0, k1) - 7 terms
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1

        # M2: (A10+A11)*B00
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1

        # M3: A00*(B11-B01)
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1

        # M4: A11*(B00-B01)
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1

        # M5: (A00+A01)*B11
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

        # Schoolbook for column k2 - 4 terms
        U[0][col] = 1; V[0*p + k2][col] = 1; W[k2*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k2][col] = 1; W[k2*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k2][col] = 1; W[k2*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k2][col] = 1; W[k2*n + 1][col] = 1; col += 1

    add_2x2x3_block(0, 1, 2)
    add_2x2x3_block(3, 4, 5)
    add_2x2x3_block(6, 7, 8)

    return U, V, W


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1

        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1

        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1

        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1

        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p):
    """Try to remove terms from the decomposition if they are linearly dependent."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_removal(indices_to_remove):
        mask = [i for i in range(rank) if i not in indices_to_remove]
        if len(mask) == 0:
            return None
        U_sub = Ua[:, mask]
        V_sub = Va[:, mask]
        W_sub = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
        if np.allclose(T, T_recon, atol=1e-10):
            new_rank = len(mask)
            return (
                [[U_sub[i][j] for j in range(new_rank)] for i in range(n * m)],
                [[V_sub[i][j] for j in range(new_rank)] for i in range(m * p)],
                [[W_sub[i][j] for j in range(new_rank)] for i in range(p * n)],
            )
        return None

    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        result = check_removal([r])
        if result is not None:
            return result

    for r1 in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        for r2 in range(r1 + 1, rank):
            result = check_removal([r1, r2])
            if result is not None:
                return result

    if time.time() - _START_TIME < _TIMEOUT - 15:
        for r1 in range(rank):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                return None
            for r2 in range(r1 + 1, rank):
                if time.time() - _START_TIME > _TIMEOUT - 5:
                    return None
                for r3 in range(r2 + 1, rank):
                    if time.time() - _START_TIME > _TIMEOUT - 5:
                        return None
                    result = check_removal([r1, r2, r3])
                    if result is not None:
                        return result

    return None


def _try_rank_30_search(n, m, p, budget_sec=60.0):
    """
    Search for a rank-30 decomposition by optimizing over 30 terms
    initialized from the rank-33 structure with 3 terms removed.
    Uses smooth L2 objective on the tensor reconstruction error.
    """
    if time.time() - _START_TIME > _TIMEOUT - 20:
        return None

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat = T.ravel()
    target_size = T_flat.size

    best_U = None
    best_V = None
    best_W = None
    best_obj = np.inf

    deadline = time.time() + budget_sec

    # Initialize from rank-33, remove 3 terms (one from each block's schoolbook part)
    U33, V33, W33 = _build_rank_33(n, m, p)
    U33a = np.array(U33, dtype=float)
    V33a = np.array(V33, dtype=float)
    W33a = np.array(W33, dtype=float)

    # Remove terms at positions: 10 (block 1 schoolbook), 21 (block 2 schoolbook), 32 (block 3 schoolbook)
    # Actually let's try removing one term from each block
    # Block 1: terms 0-10, Block 2: terms 11-21, Block 3: terms 22-32
    # Remove one schoolbook term from each block
    remove_indices = [10, 21, 32]
    keep_indices = [i for i in range(33) if i not in remove_indices]

    U0 = U33a[:, keep_indices].copy()  # 4 x 30
    V0 = V33a[:, keep_indices].copy()  # 18 x 30
    W0 = W33a[:, keep_indices].copy()  # 18 x 30

    def compute_obj(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T_recon - T) ** 2)

    def compute_grad(U, V, W):
        """Compute gradient of sum of squared errors."""
        # T_recon[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # d/dU[i,r] = 2 * sum_{j,k} (T_recon[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        E = T_recon - T  # error tensor

        # grad_U[i,r] = 2 * sum_{j,k} E[i,j,k] * V[j,r] * W[k,r]
        # = 2 * einsum('ijk,jr,kr->ir', E, V, W)
        gU = 2.0 * np.einsum('ijk,jr,kr->ir', E, V, W)
        gV = 2.0 * np.einsum('ijk,ir,kr->jr', E, U, W)
        gW = 2.0 * np.einsum('ijk,ir,jr->kr', E, U, V)
        return gU, gV, gW

    # L-BFGS style optimization using scipy
    from scipy.optimize import minimize

    def flatten_params(U, V, W):
        return np.concatenate([U.ravel(), V.ravel(), W.ravel()])

    def unflatten_params(x):
        nu = U0.size
        nv = V0.size
        U = x[:nu].reshape(U0.shape)
        V = x[nu:nu+nv].reshape(V0.shape)
        W = x[nu+nv:].reshape(W0.shape)
        return U, V, W

    def objective(x):
        U, V, W = unflatten_params(x)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T_recon - T) ** 2)

    def gradient(x):
        U, V, W = unflatten_params(x)
        gU, gV, gW = compute_grad(U, V, W)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    x0 = flatten_params(U0, V0, W0)
    obj0 = objective(x0)

    # Try L-BFGS-B
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 2000})

    if result.fun < best_obj:
        best_obj = result.fun
        U, V, W = unflatten_params(result.x)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # If L-BFGS didn't converge well, try with different initializations
    if time.time() < deadline - 10:
        # Try removing different combinations of 3 terms from rank-33
        # Block structure: [0-10], [11-21], [22-32]
        block_ranges = [(0, 11), (11, 22), (22, 33)]
        attempts = 0
        for b1 in block_ranges:
            for b2 in block_ranges:
                if b2 <= b1:
                    continue
                for b3 in block_ranges:
                    if b3 <= b2:
                        continue
                    # Pick one term from each block to remove
                    for i1 in range(b1[0], b1[1]):
                        for i2 in range(b2[0], b2[1]):
                            for i3 in range(b3[0], b3[1]):
                                if time.time() > deadline - 5:
                                    break
                                attempts += 1
                                rem = [i1, i2, i3]
                                if len(set(rem)) != 3:
                                    continue
                                keep = [i for i in range(33) if i not in rem]
                                U_t = U33a[:, keep].copy()
                                V_t = V33a[:, keep].copy()
                                W_t = W33a[:, keep].copy()
                                x_t = flatten_params(U_t, V_t, W_t)
                                res = minimize(objective, x_t, jac=gradient, method='L-BFGS-B',
                                               options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 500})
                                if res.fun < best_obj:
                                    best_obj = res.fun
                                    U, V, W = unflatten_params(res.x)
                                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                if attempts > 50:
                                    break
                        if attempts > 50:
                            break
                    if attempts > 50:
                        break
                if attempts > 50:
                    break
            if attempts > 50:
                break
        if time.time() > deadline - 5:
            break

    # Check if we achieved exact decomposition
    if best_U is not None and best_obj < 1e-12:
        # Round to nearest integers/half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        # Try without rounding
        if _validate(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()

    return None


def _try_rank_31_search(n, m, p, budget_sec=30.0):
    """
    Search for a rank-31 decomposition by removing 2 terms from rank-33
    and optimizing the remaining 31 terms.
    """
    if time.time() - _START_TIME > _TIMEOUT - 25:
        return None

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    U33, V33, W33 = _build_rank_33(n, m, p)
    U33a = np.array(U33, dtype=float)
    V33a = np.array(V33, dtype=float)
    W33a = np.array(W33, dtype=float)

    best_obj = np.inf
    best_result = None
    deadline = time.time() + budget_sec

    from scipy.optimize import minimize

    def compute_obj_grad(x, U_shape, V_shape, W_shape):
        nu = U_shape[0] * U_shape[1]
        nv = V_shape[0] * V_shape[1]
        U = x[:nu].reshape(U_shape)
        V = x[nu:nu+nv].reshape(V_shape)
        W = x[nu+nv:].reshape(W_shape)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        obj = np.sum((T_recon - T) ** 2)
        E = T_recon - T
        gU = 2.0 * np.einsum('ijk,jr,kr->ir', E, V, W)
        gV = 2.0 * np.einsum('ijk,ir,kr->jr', E, U, W)
        gW = 2.0 * np.einsum('ijk,ir,jr->kr', E, U, V)
        return obj, np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    # Try removing pairs of terms from different blocks
    block_ranges = [(0, 11), (11, 22), (22, 33)]
    attempts = 0
    for b1 in block_ranges:
        for b2 in block_ranges:
            if b2 <= b1:
                continue
            for i1 in range(b1[0], b1[1]):
                for i2 in range(b2[0], b2[1]):
                    if time.time() > deadline - 3:
                        break
                    attempts += 1
                    rem = [i1, i2]
                    keep = [i for i in range(33) if i not in rem]
                    U_t = U33a[:, keep].copy()
                    V_t = V33a[:, keep].copy()
                    W_t = W33a[:, keep].copy()
                    x_t = np.concatenate([U_t.ravel(), V_t.ravel(), W_t.ravel()])

                    def obj_fn(x):
                        o, g = compute_obj_grad(x, U_t.shape, V_t.shape, W_t.shape)
                        return o
                    def grad_fn(x):
                        o, g = compute_obj_grad(x, U_t.shape, V_t.shape, W_t.shape)
                        return g

                    res = minimize(obj_fn, x_t, jac=grad_fn, method='L-BFGS-B',
                                   options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 1000})
                    if res.fun < best_obj:
                        best_obj = res.fun
                        nu = U_t.size
                        nv = V_t.size
                        U_r = res.x[:nu].reshape(U_t.shape)
                        V_r = res.x[nu:nu+nv].reshape(V_t.shape)
                        W_r = res.x[nu+nv:].reshape(W_t.shape)
                        best_result = (U_r, V_r, W_r)
                    if attempts > 30:
                        break
                if attempts > 30:
                    break
            if attempts > 30:
                break
        if attempts > 30:
            break

    if best_result is not None and best_obj < 1e-12:
        U_r, V_r, W_r = best_result
        U_round = np.round(U_r * 2) / 2
        V_round = np.round(V_r * 2) / 2
        W_round = np.round(W_r * 2) / 2
        if _validate(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        if _validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()

    return None


def construct():
    n, m, p = DIMENSIONS

    # First, try to find rank-30 via smooth optimization
    result_30 = _try_rank_30_search(n, m, p, budget_sec=55.0)
    if result_30 is not None:
        U, V, W = result_30
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Try rank-31
    result_31 = _try_rank_31_search(n, m, p, budget_sec=25.0)
    if result_31 is not None:
        U, V, W = result_31
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: rank-32 with term removal
    U, V, W = _build_rank_32(n, m, p)
    if _validate(U, V, W, n, m, p):
        reduced = _try_remove_terms(U, V, W, n, m, p)
        if reduced is not None:
            U2, V2, W2 = reduced
            if _validate(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
